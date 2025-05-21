from importlib.util import find_spec
from typing import Dict, Any, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from .model_utils import get_model, get_tokenizer

def is_peft_available() -> bool:
    return find_spec("peft") is not None

def get_lora_model_and_tokenizer(
    model_name: str,
    lora_config: LoraConfig,
    model_kwargs: Union[Dict[str, Any], None] = None
) -> Tuple[Any, Any]:
    """
    Get a model with LoRA adapters applied and its corresponding tokenizer.
    
    Args:
        model_name (str): Name/path of the base model to load
        lora_config (LoraConfig): PEFT LoraConfig specifying LoRA parameters
        model_kwargs (Dict[str, Any], optional): Additional arguments for model loading
        
    Returns:
        Tuple[Any, Any]: (lora_model, tokenizer) tuple
        
    Raises:
        ImportError: If PEFT library is not available
        ValueError: If tokenizer doesn't have required attributes
    """
    if not is_peft_available():
        raise ImportError("PEFT library is required for LoRA functionality. Install with: pip install peft")
    
    # Load the base model
    base_model = get_model(model_name, model_kwargs)
    lora_model = get_peft_model(base_model, lora_config)
    tokenizer = get_tokenizer(model_name)
    
    return lora_model, tokenizer

if __name__ == "__main__":
    # Example LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # Rank of adaptation
        lora_alpha=32,           # LoRA scaling parameter
        lora_dropout=0.1,        # LoRA dropout
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],  # Target modules
        bias="none",             # Bias type
        use_rslora=False,        # Use rank-stabilized LoRA
        use_dora=False,          # Use DoRA (Weight-Decomposed Low-Rank Adaptation)
    )
    
    # Get LoRA model and tokenizer
    model, tokenizer = get_lora_model_and_tokenizer(
        model_name="Qwen/Qwen3-14B",
        lora_config=lora_config
    )
    model.print_trainable_parameters()