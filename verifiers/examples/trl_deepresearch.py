import os
import verifiers as vf
from verifiers.parsers import XMLParser
from trl import GRPOConfig
from datasets import concatenate_datasets
from verifiers.tools.search import search_rag
from verifiers.utils import preprocess_dataset

"""
Multi-GPU training (single node, 2 training + 6 inference)
# Qwen/Qwen3-30B-A3B or Qwen/Qwen3-32B or Qwen/Qwen3-14B or Qwen/Qwen3-8B
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen3-8B' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 4 verifiers/examples/think_rag.py
"""

TOOL_PROMPT = """\
Your primary purpose is to help users with tasks that require extensive online research tool:

You have access to these tools:

{tool_descriptions}

When reasoning, think step-by-step inside <think>...</think> tags:
- Break the question into smaller parts.

If a tool is needed, call it using JSON inside <tool>...</tool> tags:
	• "name": the tool name
	• "args": the arguments required by the tool

Tool results will appear inside <tool_response>...</tool_response> tags. You can call tools multiple times if the search results don't contain context to answer the question.

Always put your final answer inside <answer>...</answer> tags.

⸻

# Example:

## User: When was the first McDonald's established?
## Assistant:
<think>
Let me analyze the question:
- It's asking when the first McDonald's was established.
- This is a factual, date-based question about a historical event.
I don't recall the exact year, so I'll need to use a tool to retrieve it.

• Can I answer this using my knowledge?
No. I need to use a search tool.

I don't know the exact year offhand, so I'll search for it.
</think>
<tool>
{{"name": "search_rag", "args": {{"query": "first McDonald's establishment date", "num_results": 3}}}}
</tool>
## User: This context is onlu returned when the tool is used
<tool_response>
"Title: McDonald  
Context: The original McDonald's was opened by Richard and Maurice McDonald in 1940 in San Bernardino, California."
</tool_response>
## Assistant:
<think>
Based on the result, I now know that the first McDonald's was opened in 1940.
</think>
<answer>
1940
</answer>
"""

# Data
train_dataset = preprocess_dataset(name="qa", split="train")
# train_dataset = train_dataset.select(range(1000))
print(train_dataset)
print(train_dataset[0])

# eval_dataset = preprocess_dataset(name="qa", split="test")
# eval_dataset = eval_dataset.select(range(100))

vf_env = vf.ToolEnv(
    dataset=train_dataset,
    # eval_dataset=eval_dataset,
    system_prompt=TOOL_PROMPT,
    llm_fields=["think", ("tool", "answer")],
    env_fields=["tool_response"],
    few_shot=[],
    tools=[search_rag],
    max_steps=5
)

# print(vf_env.system_prompt)

# model_name = Qwen/Qwen3-30B-A3B or Qwen/Qwen3-32B or Qwen/Qwen3-14B or Qwen/Qwen3-8B
model_name = "Qwen/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "Qwen3-4B-v0.1-deepresearch" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=3e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=30,
    num_train_epochs=1,
    temperature=0.6,
    max_steps=300, # 1 epoch = 139 steps
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=4,
    beta=0.01,
    max_prompt_length=2048,
    max_completion_length=2048,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_generations=8,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    # eval_strategy="steps",
    # eval_steps=50,
    # eval_accumulation_steps=1,
    # eval_on_start=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0",
    vllm_server_port=8000,
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=vf_env.get_reward_weights(),
    scale_rewards=False,
    epsilon_high=0.28,
    mask_truncated_completions=True,
    # push_to_hub=True,
    # hub_model_id="Qwen3-8B-v0.1-deepresearch",
    # use_liger_loss=True,
    loss_type="dr_grpo"
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset(),
    eval_dataset=vf_env.get_eval_dataset()
)
trainer.train() 