export NCCL_P2P_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_V1=1
export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_IGNORE_DISABLED_P2P=1
python verifiers/inference/vllm_server.py \
    --model 'jan-hq/Qwen3-4B-v0.3-deepresearch-100-step' \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --max-model-len 40960 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --batch-request-timeout-seconds 36000 \
    --enable-prefix-caching \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000
# trl vllm-serve --model jan-hq/Qwen3-4B-v0.3-deepresearch-100-step --gpu_memory_utilization 0.90 --port 8000 --tensor-parallel-size 2 --dtype bfloat16 --host 0.0.0.0 --enforce-eager false