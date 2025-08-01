import os

from openai import OpenAI
import verifiers as vf
from verifiers.tools import python
from verifiers.utils import load_example_dataset 

"""
Evaluating multi-turn reasoning before/after training.

CUDA_VISIBLE_DEVICES=0,1 vllm serve 'Qwen/Qwen2.5-7B-Instruct' --tensor_parallel_size 2 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching \
    --host 0.0.0.0 --port 8001

uv run verifiers/examples/math_eval.py
"""

TOOL_PROMPT = """

"""

dataset = load_example_dataset("gsm8k", split="train")
vf_env = vf.ToolEnv(
    eval_dataset=dataset,
    system_prompt=TOOL_PROMPT,
    llm_fields=["think", ("tool_call", "answer")],
    env_fields=["result"],
    few_shot=[],
    tools=[python],
    max_steps=30
)

def main(api: str, num_samples: int, max_tokens: int, save_dataset: bool = False):
    # collect V3/R1 rollouts from API
    if api == "deepseek":
        base_url = "https://api.deepseek.com"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model_name = "deepseek-chat" # DeepSeek V3-0324
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif api == "openai":
        # just for testing :) not for distillation :)
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4.1" 
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Invalid API: {api}")
    sampling_args = {
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    # columns = ['prompt', 'completion', 'answer', 'reward']
    # use deepseek-chat for multiturn rollouts (V3-0324)
    results = vf_env.evaluate(
        client=client, model=model_name, 
        sampling_args=sampling_args, num_samples=num_samples)
    print("Rewards:")
    for k, v in results.items():
        if 'reward' in k:
            print(k, '-', v)
    if save_dataset:
        dataset_dsv3 = vf_env.make_dataset(results)
        # filter to top half of rows by rewards
        dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True).select(range(len(dataset_dsv3) // 2))
        # save to hub
        dataset_dsv3.push_to_hub("V3-math-python-test")

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--api", "-a", type=str, default="openai")
    argparser.add_argument("--num-samples", "-n", type=int, default=10)
    argparser.add_argument("--max-tokens", "-t", type=int, default=2048)
    argparser.add_argument("--save-dataset", "-s", action="store_true")
    args = argparser.parse_args()
    main(args.api, args.num_samples, args.max_tokens, args.save_dataset)