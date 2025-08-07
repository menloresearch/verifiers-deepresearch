export NCCL_P2P_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAG_SERVER_URL=http://10.200.108.198:2223

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 8 verifiers/examples/trl_stage_3_think.py \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --async_generation_timeout 3600 --vllm_server_host 10.200.108.158 --num_generations 4 --model_name /mnt/nas/alex/models/Qwen/Qwen3-4B-Thinking-2507 \
    --max_completion_length 40960 --max_seq_len 40960 --max_tokens 4096 --hub_model_id Menlo/jan-v1-4B-thinking-alpha-reward --wandb_project jan-v1 --max_steps_env 25 \
    --run_name jan-v1-4B-0708-base-qwen-2507-think-alpha-reward --save_steps 20 \
    --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0  --reward_correct_answer 1 --warmup_steps 32 --max_steps 640 --num_train_epochs 2

#/mnt/nas/alex/deep-research/verifiers-deepresearch/outputs/jan-v1-4B-3107-base-jan-nano/checkpoint-100
# /mnt/nas/alex/models/Qwen/Qwen3-4B-nothink