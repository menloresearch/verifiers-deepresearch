export CUDA_VISIBLE_DEVICES=2,3

# baseline: bs=64, num_generations=8
accelerate launch --config-file configs/zero3.yaml --num_processes 2 qwen3_think.py \
    --num_iterations 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
    --async_generation_timeout 3600 --num_generations 4 --model_name Qwen/Qwen3-4B \
    --max_seq_len 4096 --max_tokens 2048 --hub_model_id Menlo/noname-1.7B-thinking-v0.3 --wandb_project lucy --max_steps_env 8 \
    --run_name debug --save_steps 50 \
    # --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0 \  # non-thinking mode
    --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0 \  # thinking mode
    --warmup_steps 32 --max_steps 640 --num_train_epochs 2 \
    --optim adamw_8bit \
    > stdout.log 2> stderr.log
