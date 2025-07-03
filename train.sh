export NCCL_P2P_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 4 verifiers/examples/trl_stage_3.py \
    --async_generation_timeout 3600 --vllm_server_host 0.0.0.0 --num_generations 4 --model_name Menlo/Jan-nano \
    --max_completion_length 40960 --hub_model_id jan-hq/Jan-nano-v0.1 --wandb_project jan-nano-v0.1-cherry-prompt --max_steps_env 30 \
    --run_name jan-nano-128k
