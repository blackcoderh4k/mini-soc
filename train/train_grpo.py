"""
Mini SOC — GRPO Training Script (Official TRL)
==============================================
Trains a language model to be a SOC analyst using HuggingFace TRL's GRPOTrainer.
This replaces the old manual loop and is optimized for Colab / High-VRAM GPUs.

Usage:
  python train/train_grpo.py --steps 200 --model Qwen/Qwen2.5-1.5B-Instruct --group-size 4
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from train.reward_wrapper import compute_reward, sample_task, build_prompt_for_task
from server.mini_soc_environment import SocEnvironment

def parse_args():
    parser = argparse.ArgumentParser(description="Mini SOC GRPO Trainer (TRL)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to train")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--group-size", type=int, default=2, help="GRPO num_generations (K)")
    parser.add_argument("--output", type=str, default="outputs/grpo_checkpoints", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device train batch size")
    return parser.parse_args()

def generate_training_dataset(num_samples: int) -> Dataset:
    """
    Generate a dataset of random tasks and scenarios for training.
    GRPOTrainer requires a dataset with a 'prompt' column.
    """
    print(f"[DATASET] Generating {num_samples} training samples...", flush=True)
    env = SocEnvironment()
    data = {"prompt": [], "task_id": [], "scenario_id": []}

    for _ in range(num_samples):
        task_id, scenario_id = sample_task()
        
        # We must initialize the environment to generate the exact prompt state
        env.reset(task_id=task_id, scenario_id=scenario_id)
        obs_dict = env.state().model_dump()
        prompt = build_prompt_for_task(task_id, scenario_id, obs_dict)

        data["prompt"].append(prompt)
        data["task_id"].append(task_id)
        data["scenario_id"].append(scenario_id or "mixed")

    return Dataset.from_dict(data)

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=== Mini SOC GRPO Training (TRL) ===")
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}")
    print(f"Group Size (K): {args.group_size}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model with LoRA
    print("[MODEL] Loading model with PEFT/LoRA...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Create Dataset
    # We generate enough samples for the requested number of steps
    total_samples = args.steps * args.batch_size
    train_dataset = generate_training_dataset(total_samples)

    # 4. GRPO Configuration
    training_args = GRPOConfig(
        output_dir=args.output,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.group_size, # This is 'K' in GRPO
        max_prompt_length=1024,
        max_completion_length=256,
        save_steps=50,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False, # Important: keep task_id and scenario_id for reward func
        report_to="none" # Set to 'wandb' if wandb is available
    )

    # 5. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=compute_reward,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 6. Train!
    print("[TRAIN] Starting GRPOTrainer.train()...", flush=True)
    trainer.train()

    # 7. Save Final Model
    final_dir = os.path.join(args.output, "final_lora")
    print(f"[SAVE] Saving final adapter to {final_dir}...", flush=True)
    trainer.save_model(final_dir)
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
