export NCCL_P2P_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export RAG_SERVER_URL=http://10.200.108.198:2223
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 4 verifiers/examples/trl_stage_3.py \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --async_generation_timeout 3600 --vllm_server_host 10.200.108.158 --num_generations 16 --model_name /mnt/nas/alex/models/Qwen/Qwen3-1.7B \
    --max_completion_length 4096 --hub_model_id jan-hq/Jan-pico-1.7B-thinking-v0.2 --wandb_project noname-1.7B-thinking-v0.3 --max_steps_env 25 \
    --run_name noname-1.7B-128k-thinking-v0.3-1207 --save_steps 40 \
    --temperature 0.7 --top_p 0.95 --top_k 20 --min_p 0  --reward_correct_answer 1
