"""Minimal repro for https://github.com/NVIDIA/Megatron-LM/pull/4467.

Wraps a single MambaLayer with random weights in MegatronFSDP using
optim_grads_params + the fine-grained per-module param-gather hook. The fused
mamba conv path (mamba_split_conv1d_scan_combined) reads conv1d.weight/bias
directly instead of calling self.conv1d(...), so the per-module gather hook on
conv1d never fires. With MambaLayer registered as the FSDP unit, conv1d's
weights stay sharded across iterations -> illegal memory access on iter 1+.

Prerequisites (already applied locally):
  * MambaLayer added to fsdp_unit_modules in mcore_fsdp_adapter.py
  * enable_fine_grained_param_gather_hook hard-coded to True in mcore_fsdp_adapter.py

Launch:
  torchrun --nproc_per_node=2 repro_mamba_layer_fsdp.py
"""

import os

import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def main():
    # CP=2, DP=1: with WORLD_SIZE=2 the dp_cp group has size 2, so FSDP still
    # shards params; and cp_size=2 makes MambaContextParallel.get_conv1d_weight()
    # actually slice the conv1d params on each forward, which exercises the
    # buggy direct-read path on a sharded weight.
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )
    model_parallel_cuda_manual_seed(123)

    transformer_config = TransformerConfig(
        hidden_size=1024,
        num_layers=1,
        num_attention_heads=1,
        context_parallel_size=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8='hybrid',
        fp8_recipe='mxfp8',
        fp8_param=True,
    )

    # 'tp' is required by MambaMixer (reads pg_collection.tp.size() and uses it
    # as the tp_group for in_proj/out_proj). 'cp' is required because
    # MambaMixer.__init__ unconditionally constructs MambaContextParallel(cp_group=...).
    # dp comes from parallel_state when pg_collection is omitted on the FSDP wrapper.
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'cp']
    )

    assert isinstance(
        hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules
    )
    torch.cuda.memory._record_memory_history()
    with get_fp8_context(transformer_config, is_init=True):
        layer = MambaLayer(
            transformer_config,
            hybrid_stack_spec.submodules.mamba_layer.submodules,
            pg_collection=pg_collection,
        ).cuda()

    # Wrap MambaLayer in a thin parent so that MegatronFSDP's root is not itself
    # an FSDP unit. Otherwise MegatronFSDP.forward calls `self.module.forward(...)`
    # directly, which bypasses the forward hooks (including _post_forward,
    # pre-backward, post-backward) registered on the FSDP unit.
    class _Wrapper(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x)

    root = _Wrapper(layer).cuda()

    ddp_config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        use_megatron_fsdp=True,
        fp8_param_gather=True,
    )
    fsdp_layer = FullyShardedDataParallel(
        config=transformer_config,
        ddp_config=ddp_config,
        module=root,
    )

    rank = torch.distributed.get_rank()

    # Print bucket -> param-name + shape mapping (only on rank 0).
    if rank == 0:
        buf = fsdp_layer.param_and_grad_buffer
        name_by_id = {id(p): n for n, p in fsdp_layer.module.raw_param.items()}
        for gid, group in enumerate(buf.parameter_groups):
            entries = [
                f"{name_by_id.get(id(p), '?')}{tuple(p.shape)}" for p in group.params
            ]
            print(f"[rank {rank}] bucket {gid}: {entries}", flush=True)

    seq_len = 128
    micro_batch = 2
    conv1d = layer.mixer.conv1d
    in_proj = layer.mixer.in_proj
    mixer = layer.mixer

    def describe(name, p):
        local = p.to_local() if hasattr(p, "to_local") else p
        return (
            f"{name} type={type(p).__name__} global_shape={tuple(p.shape)} "
            f"local_shape={tuple(local.shape)} local_numel={local.numel()} "
            f"data_ptr={local.data_ptr():x} "
            f"nbytes={local.untyped_storage().nbytes()}"
        )

    for it in range(3):
        print(f"[rank {rank}] iter {it} pre-fwd {describe('conv1d.weight', conv1d.weight)}", flush=True)
        x = torch.randn(
            seq_len,
            micro_batch,
            transformer_config.hidden_size,
            device='cuda',
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with get_fp8_context(transformer_config):
            out = fsdp_layer(x)
            out.sum().backward()
        torch.cuda.synchronize()
        print(f"[rank {rank}] iter {it} ok", flush=True)
    torch.cuda.memory._dump_snapshot(f"mem_rank{rank}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
