import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge():
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = os.path.join("outputs", "grpo_checkpoints", "best")
    save_path = os.path.join("outputs", "merged_model")
    
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path not found: {adapter_path}")
        return

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu", # Merge on CPU to save VRAM for dashboard
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {save_path}")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("SUCCESS! Merged model saved to outputs/merged_model")

if __name__ == "__main__":
    merge()
