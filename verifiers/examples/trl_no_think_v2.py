import os
from trl import GRPOConfig

import verifiers as vf
from verifiers.tools.search_visit_rag import web_search, visit_tool
from verifiers.utils import load_example_dataset
import argparse

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
"""
# /no_think
def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepResearch training with configurable parameters")
    
    # Model and training configuration
    parser.add_argument("--model_name", type=str, default="jan-hq/Qwen3-4B-no-think",
                      help="Model name or path")
    parser.add_argument("--run_name", type=str, default="Qwen3-4B-v0.4-deepresearch-no-think-4",
                      help="Name for the training run")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory path")
    parser.add_argument("--learning_rate", type=float, default=3e-6,
                      help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                      choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                      help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=10,
                      help="Number of warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=2000,
                      help="Maximum number of training steps")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8,
                      help="Top-p sampling value")
    parser.add_argument("--top_k", type=int, default=20,
                      help="Top-k sampling value")
    parser.add_argument("--min_p", type=float, default=0,
                      help="Minimum probability threshold")
    parser.add_argument("--wandb_project", type=str, default=None,
                      help="wandb project name")
    
    # Hardware/performance settings
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                      help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                      help="Enable gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=0.1,
                      help="Maximum gradient norm")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                      help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=4096,
                      help="Maximum completion length")
    
    # Generation and RL settings
    parser.add_argument("--num_generations", type=int, default=6,
                      help="Number of generations per prompt")
    parser.add_argument("--num_iterations", type=int, default=4,
                      help="Number of PPO iterations")
    parser.add_argument("--beta", type=float, default=0.01,
                      help="KL penalty coefficient")
    parser.add_argument("--scale_rewards", action="store_true", default=False,
                      help="Scale rewards during training")
    parser.add_argument("--epsilon_high", type=float, default=0.28,
                      help="High threshold for advantage estimation")
    parser.add_argument("--mask_truncated_completions", action="store_true", default=True,
                      help="Mask truncated completions")
    parser.add_argument("--loss_type", type=str, default="dr_grpo",
                      help="Type of loss function")
    
    # vLLM server settings
    parser.add_argument("--use_vllm", action="store_true", default=True,
                      help="Whether to use vLLM server")
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0",
                      help="vLLM server host")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                      help="vLLM server port")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                      help="GPU memory utilization for vLLM")
    parser.add_argument("--async_generation_timeout", type=int, default=3600,
                      help="Timeout for async generation")
    
    # Dataset and environment settings
    parser.add_argument("--train_dataset", type=str, default="qa",
                      help="Training dataset name")
    parser.add_argument("--max_steps_env", type=int, default=5,
                      help="Maximum environment steps (for ToolEnv)")
    
    # Logging and saving
    parser.add_argument("--save_steps", type=int, default=100,
                      help="Number of steps between saves")
    parser.add_argument("--save_only_model", action="store_true", default=True,
                      help="Save only model weights")
    parser.add_argument("--logging_steps", type=int, default=1,
                      help="Number of steps between logging")
    parser.add_argument("--log_on_each_node", action="store_true", default=False,
                      help="Log on each node in distributed training")
    parser.add_argument("--log_completions", action="store_true", default=True,
                      help="Log completions during training")
    parser.add_argument("--report_to", type=str, default="wandb",
                      help="Where to report metrics")
    
    # Hub settings
    parser.add_argument("--push_to_hub", action="store_true", default=True,
                      help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                      help="Hugging Face Hub model ID")
    
    # Precision
    parser.add_argument("--bf16", action="store_true", default=True,
                      help="Use bfloat16 precision")
    
    return parser.parse_args()
# Data




def main():
    args = parse_args()
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    
    train_dataset = load_example_dataset(name=args.train_dataset, split="train")
    
    vf_env = vf.ToolEnv(
        dataset=train_dataset,
        system_prompt=TOOL_PROMPT,
        llm_fields=["think", ("tool", "answer")],
        few_shot=[],
        tools=[web_search, visit_tool],
        max_steps=args.max_steps_env,
    )
    
    model, tokenizer = vf.get_model_and_tokenizer(args.model_name)
    
    # Set output dir based on run name if not specified
    output_dir = args.output_dir if args.output_dir else f"outputs/{args.run_name}"
    
    training_args = vf.grpo_defaults(run_name=args.run_name)
    
    
    # Map all arguments to training_args[]
    for arg_name, arg_value in vars(args).items():
        if hasattr(training_args, arg_name):
            setattr(training_args, arg_name, arg_value)
    training_args.output_dir = output_dir
    # Set hub model ID if not specified
    if args.push_to_hub and not training_args.hub_model_id:
        training_args.hub_model_id = args.run_name
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    main()