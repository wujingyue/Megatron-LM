# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MTP main-token normalization adapted from NVIDIA-NeMo/nv-mistralai-megatron,
# revision 3ffc55243c888171e13022fc3b35c4ed9a342012, BSD-3-Clause (see LICENSE):
# https://github.com/NVIDIA-NeMo/nv-mistralai-megatron/tree/3ffc55243c888171e13022fc3b35c4ed9a342012
# Focused regression cases retained from the measured NT4 recipe's MCore tests.
import pytest
import torch

from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    process_mtp_loss,
)
from megatron.core.transformer.transformer_config import TransformerConfig

_SEED = 42


class TestMultiTokenPredictionLayer:
    def setup_method(self):
        MTPLossLoggingHelper.tracker = {}

    @pytest.mark.parametrize(
        ("normalize_by_main_tokens", "expected_valid_gradient"), [(False, 4.0 / 3.0), (True, 1.0)]
    )
    def test_process_mtp_loss_main_token_normalization_is_opt_in(
        self, normalize_by_main_tokens, expected_valid_gradient
    ):
        """The opt-in path leaves MTP as a sum for global main-token finalization."""
        config = TransformerConfig(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=1.0,
            mtp_normalize_loss_by_main_tokens=normalize_by_main_tokens,
            calculate_per_token_loss=True,
            num_layers=2,
            hidden_size=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        seq_len = 4
        hidden_states = torch.ones(2 * seq_len, 1, 1, requires_grad=True)
        mtp_module.MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))

        result = process_mtp_loss(
            hidden_states=hidden_states,
            labels=torch.arange(seq_len).unsqueeze(0),
            loss_mask=torch.ones(1, seq_len),
            output_layer=lambda hidden, **kwargs: (hidden, None),
            output_weight=None,
            is_training=False,
            compute_language_model_loss=lambda labels, logits: logits.squeeze(-1).transpose(0, 1),
            config=config,
        )
        result.sum().backward()

        mtp_grad = hidden_states.grad[seq_len:, 0, 0]
        expected = torch.tensor(
            [expected_valid_gradient] * (seq_len - 1) + [0.0], dtype=mtp_grad.dtype
        )
        assert torch.allclose(mtp_grad, expected)

    def test_main_token_mtp_normalization_requires_per_token_loss(self):
        with pytest.raises(
            ValueError,
            match="mtp_normalize_loss_by_main_tokens=True requires calculate_per_token_loss=True",
        ):
            TransformerConfig(
                mtp_num_layers=1,
                mtp_normalize_loss_by_main_tokens=True,
                calculate_per_token_loss=False,
                num_layers=2,
                hidden_size=1,
                num_attention_heads=1,
                use_cpu_initialization=True,
            )

    @pytest.mark.parametrize(
        ("calculate_per_token_loss", "normalize_by_main_tokens"),
        [(False, False), (True, False), (True, True)],
    )
    def test_process_mtp_loss_early_backward_matches_deferred_path(
        self, monkeypatch, calculate_per_token_loss, normalize_by_main_tokens
    ):
        """mtp_loss_early_backward must reproduce the deferred path's output and gradients
        (including the MTP loss scale) while running each head's projection exactly once and
        accumulating the output-weight gradient already during the forward pass."""
        monkeypatch.setattr(MTPLossLoggingHelper, "save_metrics_to_tracker", lambda *a, **k: None)
        monkeypatch.setattr(MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(0.5))
        seq_len, batch_size, vocab_size, hidden_size, num_heads = 5, 2, 16, 8, 2
        torch.manual_seed(_SEED)
        base_hidden = torch.randn((1 + num_heads) * seq_len, batch_size, hidden_size)
        base_weight = torch.randn(vocab_size, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        loss_mask[0, -2:] = 0.0

        def run(early_backward):
            config = TransformerConfig(
                mtp_num_layers=num_heads,
                num_layers=2,
                hidden_size=hidden_size,
                num_attention_heads=2,
                use_cpu_initialization=True,
                calculate_per_token_loss=calculate_per_token_loss,
                mtp_normalize_loss_by_main_tokens=normalize_by_main_tokens,
                mtp_loss_early_backward=early_backward,
            )
            hidden_states = base_hidden.clone().requires_grad_(True)
            output_weight = torch.nn.Parameter(base_weight.clone())
            calls = {"output_layer": 0}

            def output_layer(hidden, weight=None, runtime_gather_output=None):
                calls["output_layer"] += 1
                return torch.matmul(hidden, weight.t()), None

            def compute_language_model_loss(labels, logits):
                return (
                    torch.nn.functional.cross_entropy(
                        logits.float().view(-1, vocab_size),
                        labels.transpose(0, 1).reshape(-1),
                        reduction="none",
                    )
                    .view(seq_len, batch_size)
                    .transpose(0, 1)
                )

            result = process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=output_layer,
                output_weight=output_weight,
                is_training=True,
                compute_language_model_loss=compute_language_model_loss,
                config=config,
                metric_avg_group=object(),
            )
            weight_grad_after_forward = (
                None if output_weight.grad is None else output_weight.grad.clone()
            )
            assert hidden_states.grad is None
            # Main-loss backward: unit upstream gradient on the returned hidden states.
            result.backward(torch.ones_like(result))
            return (
                result.detach(),
                hidden_states.grad,
                output_weight.grad,
                weight_grad_after_forward,
                calls["output_layer"],
            )

        deferred = run(early_backward=False)
        early = run(early_backward=True)

        torch.testing.assert_close(early[0], deferred[0])
        torch.testing.assert_close(early[1], deferred[1])
        torch.testing.assert_close(early[2], deferred[2])
        # Deferred: weight grad only exists after the main backward.
        assert deferred[3] is None
        # Early: the full MTP contribution to the weight grad is already there after forward.
        torch.testing.assert_close(early[3], deferred[2])
        # No recompute in either mode: one vocab projection per head.
        assert deferred[4] == num_heads and early[4] == num_heads


class TestMTPLossLoggingHelper:
    def setup_method(self):
        MTPLossLoggingHelper.tracker = {}

    def test_track_mtp_loss_adds_opt_in_token_weighted_metrics(self):
        """Raw sums produce exact global-style metrics while retaining the legacy loss."""
        layer_number = 0
        loss_scale = torch.tensor(0.5)
        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=torch.tensor(10.0),
            correct=torch.tensor(1.0),
            total=torch.tensor(2.0),
            loss_sum=torch.tensor(20.0),
            loss_total=torch.tensor(2.0),
            main_total=torch.tensor(4.0),
            layer_number=layer_number,
            num_layers=1,
        )
        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=torch.tensor(2.0),
            correct=torch.tensor(3.0),
            total=torch.tensor(10.0),
            loss_sum=torch.tensor(20.0),
            loss_total=torch.tensor(10.0),
            main_total=torch.tensor(12.0),
            layer_number=layer_number,
            num_layers=1,
        )

        class DummyWriter:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, iteration):
                self.scalars[name] = value

        writer = DummyWriter()
        total_loss_dict = {}
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale, iteration=1, writer=writer, total_loss_dict=total_loss_dict
        )

        assert torch.isclose(torch.as_tensor(writer.scalars["mtp_1 loss"]), torch.tensor(6.0))
        assert torch.isclose(
            torch.as_tensor(writer.scalars["mtp_1 token-weighted loss"]), torch.tensor(40.0 / 12.0)
        )
        assert torch.isclose(
            torch.as_tensor(writer.scalars["mtp_1 valid fraction"]), torch.tensor(12.0 / 16.0)
        )
        assert "mtp_1 token-weighted loss" in total_loss_dict
        assert "mtp_1 valid fraction" in total_loss_dict

    def test_token_weighted_metrics_share_one_sum_collective(self, monkeypatch):
        """The opt-in metrics add payload but not an additional collective."""
        avg_group = object()
        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss=torch.tensor(3.0),
            correct=torch.tensor(1.0),
            total=torch.tensor(2.0),
            loss_sum=torch.tensor(4.0),
            loss_total=torch.tensor(2.0),
            main_total=torch.tensor(5.0),
            layer_number=0,
            num_layers=1,
            avg_group=avg_group,
        )
        calls = []

        def fake_all_reduce(tensor, *, group, op=None):
            calls.append((tensor.shape, group, op))
            if op == torch.distributed.ReduceOp.SUM:
                tensor.mul_(2)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        MTPLossLoggingHelper.reduce_metrics_in_tracker()

        assert calls == [
            (torch.Size([1]), avg_group, torch.distributed.ReduceOp.AVG),
            (torch.Size([5, 1]), avg_group, torch.distributed.ReduceOp.SUM),
        ]
        tracker = MTPLossLoggingHelper.tracker
        assert torch.equal(tracker["loss_values"], torch.tensor([3.0], device="cuda"))
        assert torch.equal(tracker["loss_sum_values"], torch.tensor([8.0], device="cuda"))
        assert torch.equal(tracker["loss_total_values"], torch.tensor([4.0], device="cuda"))
        assert torch.equal(tracker["correct_values"], torch.tensor([2.0], device="cuda"))
        assert torch.equal(tracker["total_values"], torch.tensor([4.0], device="cuda"))
        assert torch.equal(tracker["main_total_values"], torch.tensor([10.0], device="cuda"))
