import os
from trl import GRPOConfig

import verifiers as vf
from verifiers.tools.search_visit_rag import web_search, visit_tool
from verifiers.utils import preprocess_dataset

os.environ["WANDB_PROJECT"] = "DeepResearch-v0.2-visit-site-no-think"
"""
Multi-GPU training (single node, 2 training + 6 inference)
# Qwen/Qwen3-30B-A3B or Qwen/Qwen3-32B or Qwen/Qwen3-14B or Qwen/Qwen3-8B
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'jan-hq/Qwen3-14B-v0.2-deepresearch-200-step' \
    --tensor_parallel_size 2 --data_parallel_size 2 --max_model_len 16384 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 4 verifiers/examples/trl_visit_site_no_think.py
"""

TOOL_PROMPT = """
Your primary purpose is to help users with tasks that require extensive online research.

Available tools:
{tool_descriptions}

When handling user queries:

1. When you need to search for information, call the web_search tool using this exact XML format:
<tool>
{{"name": "web_search", "args": {{"query": "your search query here", "num_results": 5}}}}
</tool>

2. If search results show promising URLs/documents but you need more detailed information, use the visit_tool tool:
<tool>
{{"name": "visit_tool", "args": {{"url": "doc_1 or specific URL from search results"}}}}
</tool>

3. Tool results will appear inside <result>...</result> tags

4. You can call tools multiple times with refined queries if initial results don't contain sufficient information

5. After gathering all necessary information, provide your final answer inside <answer>...</answer> tags

Example query and response flow:
User: "When was McDonald's founded and who was its founder?"

This question has two parts:
1. The founding date of McDonald's
2. The founder(s) of McDonald's
I'll search for this information first, then visit specific pages if needed.

<tool>
{{"name": "web_search", "args": {{"query": "McDonald's founding date founder history", "num_results": 3}}}}
</tool>

<result>
Result 1:
Title: McDonald's Corporation History
URL: doc_1
Preview: McDonald's was founded in 1940 by Richard and Maurice McDonald in San Bernardino, California...

Result 2:
Title: Ray Kroc and McDonald's Expansion
URL: doc_2
Preview: Ray Kroc joined McDonald's in 1955 and transformed it into a global franchise...
</result>

<tool>
{{"name": "visit_tool", "args": {{"url": "doc_1"}}}}
</tool>

<result>
Title: McDonald's Corporation History
URL: doc_1

Full Content:
McDonald's was founded on May 15, 1940, in San Bernardino, California by brothers Richard and Maurice McDonald...
</result>

<answer>
McDonald's was founded on May 15, 1940, in San Bernardino, California. The original McDonald's restaurant was opened by brothers Richard and Maurice McDonald. However, the McDonald's Corporation as we know it today was created by Ray Kroc, who joined the company in 1955 as a franchise agent and later purchased the chain from the McDonald brothers.
</answer>
/no_think
"""

# Data
train_dataset = preprocess_dataset(name="qa", split="train")
# train_dataset = train_dataset.select(range(1000))
print(train_dataset)
print(train_dataset[0])

eval_dataset = preprocess_dataset(name="qa", split="test")

vf_env = vf.ToolEnv(
    dataset=train_dataset,
    # eval_dataset=eval_dataset,
    system_prompt=TOOL_PROMPT,
    llm_fields=["think", ("tool", "answer")],
    few_shot=[],
    tools=[web_search, visit_tool],
    max_steps=5,
)

# print(vf_env.system_prompt)

# model_name = Qwen/Qwen3-30B-A3B or Qwen/Qwen3-32B or Qwen/Qwen3-14B or Qwen/Qwen3-8B
model_name = "jan-hq/Qwen3-14B-v0.2-deepresearch-200-step"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "Qwen3-14B-v0.2-deepresearch-no-think" + model_name.split("/")[-1].lower()

training_args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=3e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    num_train_epochs=5,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_steps=500,  # 1 epoch = 139 steps
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=4,
    beta=0.01,
    max_prompt_length=2048,
    max_completion_length=16384,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
    push_to_hub=True,
    hub_model_id="Qwen3-14B-v0.2-deepresearch-no-think",
    # use_liger_loss=True,
    loss_type="dr_grpo",
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset(),
    eval_dataset=vf_env.get_eval_dataset(),
)
trainer.train()