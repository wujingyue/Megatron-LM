# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Uneven fan-in gradient split adapted from NVIDIA-NeMo/nv-mistralai-megatron,
# revision 3ffc55243c888171e13022fc3b35c4ed9a342012, BSD-3-Clause (see LICENSE):
# https://github.com/NVIDIA-NeMo/nv-mistralai-megatron/tree/3ffc55243c888171e13022fc3b35c4ed9a342012

from collections import deque

import torch

from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator


class TestUnevenFanIn:
    @staticmethod
    def _bridge():
        bridge = BridgeCommunicator.__new__(BridgeCommunicator)
        bridge.tensor_ndim = 2
        bridge.dim_mapping = {'b': 0}
        bridge._received_forward_shapes = deque()
        return bridge

    def test_eight_uneven_image_shards_preserve_projector_gradient_and_fifo(self):
        bridge = self._bridge()
        torch.manual_seed(41)
        weight = torch.randn(3, 4, dtype=torch.float64)
        reference_weight = weight.clone().requires_grad_()
        global_tokens = 73
        all_sizes = [[0, 1, 0, 3, 2, 0, 4, 1], [2, 0, 1, 0, 3, 1, 0, 2], [0] * 8]
        batches = []
        reference_loss = reference_weight.sum() * 0
        for sizes in all_sizes:
            features = torch.randn(sum(sizes), 4, dtype=torch.float64)
            upstream = torch.randn(sum(sizes), 3, dtype=torch.float64)
            reference_loss = reference_loss + ((features @ reference_weight.T) * upstream).sum()
            bridge._received_forward_shapes.append([(size, 3) for size in sizes])
            batches.append((sizes, features, upstream))
        (reference_loss / global_tokens).backward()
        # Encoder DDP performs SUM for per-token loss, then divides once by the
        # language global token count. Image shards are not independent samples.
        shard_gradient_sum = torch.zeros_like(weight)
        for sizes, features, upstream in batches:
            returned_gradients = bridge._split_backward_gradient(upstream, 8)
            for local_features, returned_gradient, size in zip(
                torch.split(features, sizes), returned_gradients, sizes, strict=True
            ):
                assert returned_gradient.shape == (size, 3)
                if size == 0:
                    continue
                local_weight = weight.clone().requires_grad_()
                output = local_features @ local_weight.T
                output.backward(returned_gradient)
                shard_gradient_sum += local_weight.grad
        torch.testing.assert_close(shard_gradient_sum / global_tokens, reference_weight.grad)
        assert not bridge._received_forward_shapes
