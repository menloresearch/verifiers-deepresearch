import os
from trl import GRPOConfig

import verifiers as vf
from verifiers.tools.search_visit_rag import web_search, visit_tool
from verifiers.utils import load_example_dataset

os.environ["WANDB_PROJECT"] = "DeepResearch-v0.4-visit-site-no-think-test"
"""
Multi-GPU training (single node, 2 training + 6 inference)
# Qwen/Qwen3-30B-A3B or Qwen/Qwen3-32B or Qwen/Qwen3-14B or Qwen/Qwen3-8B
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_server.py \
    --model 'jan-hq/Qwen3-4B-v0.3-deepresearch-100-step' \
    --tensor-parallel-size 2 \
    --data_parallel_size 2 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num_processes 4 verifiers/examples/trl_no_think.py
"""

TOOL_PROMPT = """
Your primary purpose is to help users with tasks that require extensive online research.

Available tools:
{tool_descriptions}

When handling user queries:

Think step-by-step about the query inside :
   - Break complex questions into smaller, searchable parts
   - Identify key search terms and parameters
   - Consider what information is needed to provide a complete answer

1. When you need to search for information, call the web_search tool using this exact XML format:
<tool>
{{"name": "web_search", "args": {{"query": "your search query here"}}}}
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
{{"name": "web_search", "args": {{"query": "McDonald's founding date founder history"}}}}
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
...
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
train_dataset = load_example_dataset(name="qa", split="train")
print(train_dataset)
print(train_dataset[0])

eval_dataset = load_example_dataset(name="qa", split="test")

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
model_name = "jan-hq/Qwen3-4B-no-think"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "Qwen3-4B-v0.4-deepresearch-no-think-4" 

training_args=vf.grpo_defaults(run_name=run_name)
training_args.output_dir = f"outputs/{run_name}"
training_args.learning_rate = 3e-6
training_args.lr_scheduler_type = "constant_with_warmup"
training_args.warmup_steps = 10
training_args.num_train_epochs = 10
training_args.temperature = 0.7
training_args.top_p = 0.8
training_args.top_k = 20
training_args.min_p = 0
training_args.max_steps = 2000
training_args.bf16 = True
training_args.max_grad_norm = 0.1
training_args.num_iterations = 4
training_args.beta = 0.01
training_args.max_prompt_length = 2048
training_args.max_completion_length = 4096
training_args.per_device_train_batch_size = 1
# training_args.per_device_eval_batch_size = 1
training_args.num_generations = 6
training_args.gradient_accumulation_steps = 4
training_args.gradient_checkpointing = True
training_args.save_strategy = "steps"
training_args.save_steps = 100
training_args.save_only_model = True
training_args.use_vllm = True
training_args.vllm_server_host = "0.0.0.0" #10.200.108.158
training_args.vllm_server_port = 8000
training_args.vllm_gpu_memory_utilization = 0.9
training_args.logging_steps = 1
training_args.log_on_each_node = False
training_args.log_completions = True
training_args.report_to = "wandb"
training_args.scale_rewards = False
training_args.epsilon_high = 0.28
training_args.mask_truncated_completions = True
training_args.push_to_hub = True
training_args.hub_model_id = "Qwen3-4B-v0.4-deepresearch-no-think-4"
training_args.loss_type = "dr_grpo"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()