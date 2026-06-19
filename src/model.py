import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def load_model_and_tokenizer(model_name: str, use_peft: bool = False, 
                             lora_r: int = 8, lora_alpha: int = 32, 
                             lora_dropout: float = 0.1):
    """
    Loads a sequence-to-sequence model and tokenizer, optionally wrapping it
    with PEFT LoRA adapters (EXP-07).
    """
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading Seq2Seq model for: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Configure Parameter-Efficient Fine-Tuning (LoRA)
    if use_peft:
        print(f"Wrapping model with PEFT LoRA (r={lora_r}, alpha={lora_alpha})...")
        
        # Determine target modules based on model architecture
        target_modules = ["q", "v"]  # Default for T5 models
        if "mbart" in model_name.lower():
            target_modules = ["q_proj", "v_proj"]
            
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    return model, tokenizer

def load_peft_model_for_inference(base_model_name: str, adapter_path: str):
    """
    Loads a pre-trained base model and its fine-tuned PEFT LoRA adapters
    for model deployment or test submission inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer
