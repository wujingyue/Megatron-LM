# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.transformer.moe import fused_a2a


@pytest.mark.parametrize("reserve,expected", [(None, 8), (4, 8), (8192, 8192)])
def test_initial_capacity_and_backward(monkeypatch, reserve, expected):
    """Reservation affects allocation, while dispatch shapes and gradients stay intact."""
    allocations = []

    class Buffer:
        def dispatch_with_permute(self, hidden, probs, **kwargs):
            return hidden.clone(), probs.clone(), None, torch.tensor([8]), None

        def combine_with_unpermute(self, hidden, probs, **kwargs):
            return hidden, probs

    def initialize(*args):
        allocations.append(args[2])
        monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", Buffer())

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    monkeypatch.setattr(fused_a2a, "init_hybrid_ep_buffer", initialize)
    x = torch.randn(8, 2, requires_grad=True)
    probs = torch.ones(8, 1, requires_grad=True)
    out, p, *_ = fused_a2a.HybridEPDispatch.apply(
        x,
        torch.ones(8, 1, dtype=torch.bool),
        probs,
        None,
        1,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        108,
        None,
        None,
        reserve,
    )
    assert allocations == [expected]
    torch.testing.assert_close(out, x)
    (out.sum() + p.sum()).backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(probs.grad, torch.ones_like(probs))
