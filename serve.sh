export NCCL_P2P_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export NCCL_DEBUG=INFO
# export VLLM_TRACE_FUNCTION=1
 # --batch-request-timeout-seconds 36000 \
CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_server.py \
    --model 'jan-hq/Qwen3-4B-v0.3-deepresearch-100-step' \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 40960 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --batch-request-timeout-seconds 36000 \
    --enable-prefix-caching \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000
