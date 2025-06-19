# export NCCL_P2P_DISABLE=0
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 6 verifiers/examples/trl_no_think_v2.py \
    --async_generation_timeout 3600 --vllm_server_host inference  
    