# export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=2,3,4,5
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --config-file configs/zero3.yaml --num_processes 4 verifiers/examples/trl_no_think_v2.py \
    --async_generation_timeout 3600 --vllm_server_host inference --num_generations 4
    