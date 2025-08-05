import argparse
import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import jinja2
import verifiers as vf
import dotenv
from verifiers.tools.search_visit_rag import visit_tool, web_search
from verifiers.utils.data_utils import load_example_dataset
from verifiers.utils.tool_utils import convert_func_to_oai_tool

dotenv.load_dotenv(override=True)

QWEN3_TOOLS_TEMPLATE = jinja2.Template(
    r"""
{{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
{%- for tool in tools %}
    {{- "\n" }}
    {{- tool | tojson }}
{%- endfor %}
{{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n" }}
""".strip()
)

SYSTEM_PROMPT = """
Your primary purpose is to help users with tasks that require extensive online research.

When handling user queries:
1. Think step-by-step about the query inside <think>...</think> tags:
   - Break complex questions into smaller, searchable parts
   - Identify key search terms and parameters
   - Consider what information is needed to provide a complete answer

2. When you need to search for information, call the "web_search" tool using this exact XML format:
<tool_call>
{{"name": "web_search", "arguments": {{"query": "your search query here"}}}}
</tool_call>

3. If search results show promising URLs/documents but you need more detailed information, use the "visit_tool" tool:
<tool_call>
{{"name": "visit_tool", "arguments": {{"url": "doc_1 or specific URL from search results"}}}}
</tool_call>

4. Tool response results will appear inside <tool_response>...</tool_response> tags

5. After gathering all necessary information, provide your final answer inside <answer>...</answer> tags

{tool_descriptions}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepResearch training with configurable parameters"
    )

    # Model and training configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="jan-hq/Qwen3-4B-no-think",
        help="Model name or path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="Qwen3-4B-v0.4-deepresearch-no-think-4",
        help="Name for the training run",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory path"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.5e-6, help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="warmup_stable_decay",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "warmup_stable_decay",
        ],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=20, help="Number of warmup steps"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=2000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling value"
    )
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling value")
    parser.add_argument(
        "--min_p", type=float, default=0, help="Minimum probability threshold"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="wandb project name"
    )
    parser.add_argument(
        "--reward_correct_answer",
        type=float,
        default=1.0,
        help="Reward weight for correct answer",
    )
    parser.add_argument(
        "--reward_tool_execution",
        type=float,
        default=0.2,
        help="Reward weight for tools execution",
    )
    parser.add_argument(
        "--reward_format",
        type=float,
        default=0.2,
        help="Reward weight for format parser",
    )

    # Hardware/performance settings
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per device train batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.1, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=2048, help="Maximum prompt length"
    )
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens per vLLM response")
    parser.add_argument(
        "--max_seq_len", type=int, default=4096
    )  # will truncate if total tokens exceed this
    parser.add_argument(
        "--optim", default="adamw_torch_fused"
    )  # see transformers.training_args.OptimizerNames

    # Generation and RL settings
    parser.add_argument(
        "--num_generations",
        type=int,
        default=6,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1, help="Number of PPO iterations"
    )
    parser.add_argument(
        "--beta", type=float, default=0.01, help="KL penalty coefficient"
    )
    parser.add_argument(
        "--scale_rewards",
        action="store_true",
        default=False,
        help="Scale rewards during training",
    )
    parser.add_argument(
        "--epsilon_high",
        type=float,
        default=0.28,
        help="High threshold for advantage estimation",
    )
    parser.add_argument(
        "--loss_type", type=str, default="dr_grpo", help="Type of loss function"
    )

    # vLLM server settings
    parser.add_argument(
        "--vllm_server_host", type=str, default="0.0.0.0", help="vLLM server host"
    )
    parser.add_argument(
        "--vllm_server_port", type=int, default=8000, help="vLLM server port"
    )
    parser.add_argument(
        "--async_generation_timeout",
        type=int,
        default=3600,
        help="Timeout for async generation",
    )

    # Dataset and environment settings
    parser.add_argument(
        "--train_dataset", type=str, default="qa", help="Training dataset name"
    )
    parser.add_argument(
        "--max_steps_env",
        type=int,
        default=5,
        help="Maximum environment steps (for ToolEnv)",
    )

    # Logging and saving
    parser.add_argument(
        "--save_steps", type=int, default=100, help="Number of steps between saves"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Number of steps between logging"
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb", help="Where to report metrics"
    )

    # Hub settings
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None, help="Hugging Face Hub model ID"
    )

    return parser.parse_args()


# Data


def main():
    args = parse_args()
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    train_dataset = load_example_dataset(name=args.train_dataset, split="train")

    tools = [web_search, visit_tool]
    oai_tools = [convert_func_to_oai_tool(tool) for tool in tools]
    TOOL_PROMPT = QWEN3_TOOLS_TEMPLATE.render(tools=oai_tools)

    vf_env = vf.OldToolEnv(
        dataset=train_dataset,
        system_prompt=SYSTEM_PROMPT.format(tool_descriptions=TOOL_PROMPT),
        few_shot=[],
        tools=tools,
        format_prompt=False,
        max_turns=args.max_steps_env,
        max_seq_len=args.max_seq_len,
    )
    vf_env.rubric.reward_weights = [
        args.reward_correct_answer,
        args.reward_tool_execution,
        args.reward_format,
        0.2,
        0.2,
        0.2,
        0.0,
    ]  #

    model, tokenizer = vf.get_model_and_tokenizer(args.model_name)

    # Set output dir based on run name if not specified
    output_dir = args.output_dir if args.output_dir else f"outputs/{args.run_name}"

    # check if output_dir is not empty
    if Path(output_dir).exists():
        assert len(list(Path(output_dir).iterdir())) == 0

    training_args = vf.grpo_defaults(run_name=args.run_name)

    # Map all arguments to training_args[]
    for arg_name, arg_value in vars(args).items():
        if hasattr(training_args, arg_name):
            setattr(training_args, arg_name, arg_value)

    training_args.output_dir = output_dir

    # Set hub model ID if not specified
    if args.push_to_hub and not training_args.hub_model_id:
        training_args.hub_model_id = args.run_name
    if args.lr_scheduler_type == "warmup_stable_decay":
        print("Using warmup_stable_decay")
        setattr(training_args, "lr_scheduler_kwargs", {"num_decay_steps": 128})

    print(training_args)

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
