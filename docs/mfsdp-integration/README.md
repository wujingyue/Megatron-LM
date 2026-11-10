# MIMO SFT integration on current main

This integration refresh is based on upstream Megatron-LM `main` at
`874044ee8c0061dfea25799b10dab7b7c20ce655` (November10,2026 snapshot). The previous
measured integration remains pinned at `12bf304f14793dce5fa21d42a5bd094d537ce825`,
based on upstream `364eef70a30aacdc622e75f6f437b50c2ef18278`. Historical measurements
do not describe this refreshed source until new measurements pass review.

## Pending changes included

| PR | Purpose | Pinned source head |
| --- | --- | --- |
| [#7322](https://github.com/NVIDIA/Megatron-LM/pull/7322) | Ordinary local optimizer parameters and gradients; DTensors at checkpoint boundaries | `393fd9f816972f9c626f42471ffe85675dae9ddf` |
| [#8422](https://github.com/NVIDIA/Megatron-LM/pull/8422) | Residual replay within an execution layer | `10218923c3d712ae05bb3d1862701de9cd5d2d71` |
| [#8609](https://github.com/NVIDIA/Megatron-LM/pull/8609) | MIMO optimizer construction and module-local gradient statistics | `79e261599f32d1c81b7aede09c18110fbdda48f3` |
| [#8605](https://github.com/NVIDIA/Megatron-LM/pull/8605) | Historical distributed-checkpoint names for residual connections | `fde642669387c3a93cfb21cb4b15b0841609eea7` |
| [#9330](https://github.com/NVIDIA/Megatron-LM/pull/9330) | Stage each completed local gradient into its communication buffer | `199556b1d38ddf73982c165967d9759b405e399f` |

#7322 is adapted to main's extracted `mcore_adapter_v2.py` and separate inner
and outer reductions. Runtime `.grad` views match local optimizer shards even
while a larger partial buffer accumulates across the outer data-parallel axis.
Checkpoint DTensor views retain global and uneven-shard metadata. #8605 uses
main's explicit wide-residual layer classes. The MFSDP replay gate allows only
layer-local replay (`residual_stream_recompute_num_layers=1`).

Current main already includes the corrected #8312 ownership implementation,
MTP normalization by main-loss token count, token-aligned packed-vision RoPE,
GDP context-parallel startup warmup, and GTP. Their older overlays are omitted.
The dev-targeted GTP PR #6882 is not needed for this main-based integration.

## Shared compatibility and historical recipe

The integration retains configurable initial HybridEP reservation (original
`58cbbd104`) and uneven bridge-gradient slices/ordering (`3db309a97`). These
support the MIMO recipe across DDP, MFSDP and GTP.

The optional historical patches preserve the previous measurements' BF16 vision
RoPE frequencies and token-ID MTP embeddings. They do not establish the correct
pretrained semantics; the branch itself retains upstream choices. They apply
to the refreshed source without changing the patch files:

```bash
git apply --check docs/mfsdp-integration/vision-historical-bf16.patch \
    docs/mfsdp-integration/mtp-historical-token-input.patch
git apply docs/mfsdp-integration/vision-historical-bf16.patch \
    docs/mfsdp-integration/mtp-historical-token-input.patch
```

SHA-256, in the same order:

```text
4098da5c802c3cdacf54c09e707c06ccb5efd35fa515fd0058aa2a5e4312024b
643a754d619fb4d9d622811f5a74c5604bc3818593a54cf8525236a34fec4c6c
```

## Precision and external integration

The recorded MFSDP recipe inherits stored-gradient dtype from each compute
parameter and uses FP32 communication. BF16 parameters keep BF16 gradients;
FP32 parameters keep FP32 gradients. On current main, retain that inheritance
and disable the legacy validation flag:

```python
DistributedDataParallelConfig(
    use_megatron_fsdp=True,
    megatron_fsdp_version=2,
    grad_reduce_in_fp32=False,
    megatron_fsdp_main_params_dtype=torch.float32,
    megatron_fsdp_main_grads_dtype=None,  # Inherit each parameter group's dtype.
    megatron_fsdp_grad_comm_dtype=torch.float32,
)
```

Merged #8922 validates `grad_reduce_in_fp32=True` against FP32 main-gradient
storage and communication. It does not implement the earlier experiment's
flag-to-communication mapping while retaining inherited gradient storage. Vision remains DDP
with distributed Adam and `grad_reduce_in_fp32=True` in every method. The
recorded DDP/GTP language recipes also retain FP32 main gradients.

Bridge/NeMo-RL still require wide-residual decoder specs with ordinary-width
MTP, checkpoint-metadata compatibility, MIMO initialization and process groups,
explicit dtype forwarding, and per-role optimizer/DDP configuration. A Core
checkout alone is not a complete reproduction.

## Validation and measurement scope

Fresh measurement scope is576 GPUs (512 language+64 vision), CP=EP64, DP8,
global batch32, microbatch1 and four microbatches per replica,100 updates per
method. DDP/GTP retain their recorded four-layer replay; MFSDP uses layer-local
replay plus early staging. Native latency and allocator-memory measurements are
separate. Earlier results remain archived.

Focused source/lint and four-GPU validation are being recorded in
`nt4-mimo-main-refresh-576-20261110`. All-rank source, precision, ownership,
replay, workload and numerical evidence must pass review before publication.
Norm discrepancies remain unresolved; scalar agreement is not full-gradient
equivalence. Instrumented memory timing is not native latency.
