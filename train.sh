# export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 8 verifiers/examples/trl_stage_3.py \
    --async_generation_timeout 3600 --vllm_server_host 10.200.108.158 --num_generations 8 --model_name Menlo/Jan-nano \
    --max_completion_length 40960 --hub_model_id jan-hq/Jan-nano-v0.1 --wandb_project jan-nano-v0.1-cherry-prompt --max_steps_env 30 \
    --run_name jan-nano-128k
    