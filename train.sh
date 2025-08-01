export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --config-file configs/zero3.yaml --num_processes 2 qwen3_think.py \
    --num_iterations 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --async_generation_timeout 3600 --num_generations 8 --model_name Qwen/Qwen3-4B \
    --max_tokens 4096 --hub_model_id Menlo/noname-1.7B-thinking-v0.3 --wandb_project lucy --max_steps_env 8 \
    --run_name debug --save_steps 100 \
    --temperature 0.7 --top_p 0.9 --top_k 20 --min_p 0 --reward_correct_answer 1 --warmup_steps 32 --max_steps 640 --num_train_epochs 2
