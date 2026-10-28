# MFSDP integration branch

`wujingyue/Megatron-LM:mfsdp` integrates the Core changes used by the October 28,
2026 NT4 Nano MIMO retest. Its base is upstream `main` at
`364eef70a30aacdc622e75f6f437b50c2ef18278`, not the moving tip of `main`.

The branch retains upstream vision-frequency precision and multimodal MTP input
selection. The two optional patches below restore the historical choices used
in the retest. With those patches applied, every Core entry in the completed
run's source manifest matches. Bridge and NeMo-RL integration is still required;
checking out this Core branch alone is not a complete experiment setup.

## Included changes

| Change | Integrated commit | Source revision |
| --- | --- | --- |
| [#7322: local optimizer tensors](https://github.com/NVIDIA/Megatron-LM/pull/7322) | `b133cddcd` | `393fd9f816972f9c626f42471ffe85675dae9ddf` |
| [#8312: ShortcutMoEBlock FSDP ownership](https://github.com/NVIDIA/Megatron-LM/pull/8312) | `42e0a4312` | `95a1d24dd092bb122d0dda5f44ba8a981c6c40cd` |
| [#8605: distributed checkpoint residual keys](https://github.com/NVIDIA/Megatron-LM/pull/8605) | `4c0ce5256` | `fde642669387c3a93cfb21cb4b15b0841609eea7` |
| [#8609: MIMO optimizer integration](https://github.com/NVIDIA/Megatron-LM/pull/8609) | `c9329f874` | `79e261599f32d1c81b7aede09c18110fbdda48f3` |
| Opt-in MTP normalization by main-loss token count | `407b7b7ad` | Retest compatibility patch |
| Configurable initial HybridEP capacity | `58cbbd104` | Retest compatibility patch |
| Preserve uneven bridge-gradient slices and forward/backward ordering | `3db309a97` | Retest compatibility patch |
| Token-aligned RoPE dispatch for packed vision inputs | `d644ca795` | Forward diagnostic 1512249 |
| GDP CP boundary compilation during startup | `e5d286d3c` | Operator diagnostic 1513723 |

The integrated PR revisions are pinned. Future changes to those PR branches are
not automatically included. Optional layer-local replay
[#8422](https://github.com/NVIDIA/Megatron-LM/pull/8422) and registration ordering
[#8505](https://github.com/NVIDIA/Megatron-LM/pull/8505) are not included.

## Historical reproduction choices

Apply these only when reproducing the recorded NT4 runs. They select historical
behavior; the diagnostics do not establish the correct pretrained semantics.
From the repository root:

```bash
git apply --check docs/mfsdp-integration/vision-historical-bf16.patch \
    docs/mfsdp-integration/mtp-historical-token-input.patch
git apply docs/mfsdp-integration/vision-historical-bf16.patch \
    docs/mfsdp-integration/mtp-historical-token-input.patch
```

- `vision-historical-bf16.patch` casts vision RoPE frequencies to BF16.
- `mtp-historical-token-input.patch` selects token-ID embeddings for MTP while
  keeping the composed multimodal embeddings for the main decoder.

These files are unchanged copies of the measured experiment patches. Their
SHA-256 values are, respectively:

```text
4098da5c802c3cdacf54c09e707c06ccb5efd35fa515fd0058aa2a5e4312024b
643a754d619fb4d9d622811f5a74c5604bc3818593a54cf8525236a34fec4c6c
```

The recorded MFSDP recipe removes `residual_stream` from `recompute_modules` and
clears `residual_stream_recompute_num_layers`; it retains `moe_act`, `gdp_in_proj`,
and `gdp_qkv`. The DDP recipe retains residual-stream replay across four physical
layers, so the measured DDP/MFSDP comparison changes recomputation as well as
sharding. MTP main-token normalization is enabled, and HybridEP initially
reserves 8192 tokens per rank in that recipe.

Outside Core, reproduction also needs the recorded Bridge wide-residual decoder
specs with ordinary-width MTP, checkpoint metadata compatibility, MIMO setup and
initialization fixes, and NeMo-RL worker integration. Those changes belong in
their respective repositories and are not included here.

## Validation and limits

The completed native job 1515098 and separate memory job 1516107 each ran DDP
BF16, MFSDP BF16, and DDP MXFP8 for 100 updates per mode, on 64 language plus
8 vision GB300 GPUs, CP=EP64, DP1, and four microbatches per update. All rank
receipts were audited. These runs used both optional reproduction patches and
the external integration described above.

First-update main/MTP losses match the historical runs exactly at logged
precision. Later curves differ, and gradient-norm spikes remain unexplained.
This is an integration branch, not a claim of full numerical equivalence.
The branch without the two reproduction patches has not had a new 100-update
full-model run. Memory instrumentation supplies no latency claim.

Branch assembly checks include Python parsing, formatting/lint checks for the
two newly committed runtime files, and comparison against all 3,141 Core source
manifest entries from the completed run. Only the two optional overrides differ
before applying the patches; all entries match afterward. Assembly does not
change the frozen experiment sources or constitute another training run.
