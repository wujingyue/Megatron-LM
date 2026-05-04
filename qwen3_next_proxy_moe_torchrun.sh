#!/bin/bash

DATA_CACHE_PATH="/tmp/benchmark_cache_qwen3_next_80b_a3b"

# Torchrun launch config
GPUS_PER_NODE=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-6000}

# Environment
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
unset CUDA_DEVICE_MAX_CONNECTIONS

# Benchmark args
PRETRAIN_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 2
    --context-parallel-size 1
    --num-distributed-optimizer-instances 1
    --micro-batch-size 1
    --global-batch-size 32
    --seq-length 4096
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
)

# Extra args (CLI passthrough)
PRETRAIN_ARGS+=(
    --recompute-granularity selective
    --recompute-modules moe_act
    --moe-router-force-load-balancing
)

# Architecture args
# Qwen3-Next-80B-A3B: hybrid layout 12*(3*DeltaNet + 1*GatedAttn), all MoE FFN
PRETRAIN_ARGS+=(
    --num-layers 8
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --max-position-embeddings 32768
    --make-vocab-size-divisible-by 1187
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --swiglu
    --qk-layernorm
    --apply-wd-to-qk-layernorm
    --attention-output-gate
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --rotary-seq-len-interpolation-factor 1
    --no-rope-fusion
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 32
    --num-experts 16
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    # FIXME(@cspades): Shared gate weight not un-sharded during pre-backward. Why?
    # --moe-shared-expert-gate
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 10
    --moe-aux-loss-coeff 1e-3
)

# Training args
PRETRAIN_ARGS+=(
    --sequence-parallel
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-timeout-minutes 60
    --use-flash-attn
    --disable-bias-linear
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --init-method-std 0.02
    --use-mcore-models
    --train-samples 268554688
    --exit-duration-in-mins 230
    --manual-gc
    --manual-gc-interval 10
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-fusion
    --moe-router-force-load-balancing
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 0.00012
    --min-lr 1.2e-05
    --lr-decay-style cosine
    --lr-decay-samples 255126953
    --lr-warmup-samples 162761
    --bf16
    --fp8-format e4m3
    --fp8-recipe mxfp8
    --fp8-param-gather
    --fp8-format hybrid
    --enable-experimental
    --mock-data
    --data-cache-path $DATA_CACHE_PATH
    --split 99,1,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 6
    --tokenizer-type NullTokenizer
    --vocab-size 128256
    --tiktoken-pattern v2
    --eval-iters 32
    --eval-interval 100
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
    --log-throughput
    --log-interval 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --outer-dp-sharding-strategy no_shard
)

# Backend args (megatron_fsdp)
PRETRAIN_ARGS+=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --init-model-with-meta-device
    --calculate-per-token-loss
    --grad-reduce-in-bf16
    # --fsdp-double-buffer
    # --use-nccl-ub
    # --fsdp-manual-registration
    --ckpt-format fsdp_dtensor
)

set -x
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_gpt.py \
    "${PRETRAIN_ARGS[@]}"
