"""
Mini SOC — GRPO Training Script
=================================
Trains a language model to be a SOC analyst using Group Relative Policy
Optimization (GRPO) via HuggingFace TRL.

Architecture:
  - Base model: Qwen/Qwen2.5-1.5B-Instruct (fits Colab T4 16GB)
  - LoRA adapter: r=16, alpha=32, targeting all attention projections
  - GRPO group size: K=4 candidate sequences per prompt
  - Reward: Mini SOC environment graded score (0.0 → 1.0)
  - Training: 200 steps (demo) or 500 steps (full)

Usage:
  # Local (requires GPU with >= 16GB VRAM):
  python train/train_grpo.py --steps 200 --task all

  # With specific task focus:
  python train/train_grpo.py --steps 100 --task incident_investigation

  # Colab: see train/train_colab.ipynb

Environment variables:
  WANDB_API_KEY    — WandB logging (optional but recommended)
  HF_TOKEN         — Push checkpoints to HF Hub (optional)
  WANDB_PROJECT    — WandB project name (default: mini-soc-rl)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports — wrapped in try/except for clear error messages
# ---------------------------------------------------------------------------

try:
    import torch
    print(f"[INIT] PyTorch {torch.__version__} -- CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        _gpu_props = torch.cuda.get_device_properties(0)
        _gpu_mem = getattr(_gpu_props, 'total_memory', getattr(_gpu_props, 'total_mem', 0))
        print(f"[INIT] GPU: {torch.cuda.get_device_name(0)} ({_gpu_mem / 1e9:.1f} GB)", flush=True)
    else:
        _gpu_mem = 0
except ImportError:
    print("[FATAL] PyTorch not installed. Run: pip install torch", flush=True)
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except ImportError:
    print("[FATAL] transformers/peft not installed. Run: pip install transformers peft", flush=True)
    sys.exit(1)

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    print("[WARN] TRL not installed. Will use manual GRPO loop. Run: pip install trl>=0.12.0", flush=True)
    TRL_AVAILABLE = False

# Optional WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Mini SOC imports
from train.reward_wrapper import (
    build_prompt_for_task,
    parse_actions_from_text,
    execute_episode,
    sample_task,
    SYSTEM_PROMPTS,
    TASK_SAMPLE_WEIGHTS,
)
from server.mini_soc_environment import SocEnvironment, TASK_CONFIG


# ---------------------------------------------------------------------------
# Configuration (matches PRD §12.3)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "learning_rate": 2e-5,
    "grpo_group_size": 2,
    "max_seq_length": 512,  # Extreme reduction for 4GB
    "training_steps": 100,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 5,
    "save_every": 50,
    "eval_every": 100, # Skip eval to save memory
    "wandb_project": "mini-soc-rl",
    "output_dir": str(PROJECT_ROOT / "outputs" / "grpo_checkpoints"),
    "max_new_tokens": 128,  # Enough for JSON actions
    "temperature": 0.7,             # Higher for diverse GRPO group
    "kl_coeff": 0.05,               # KL penalty coefficient
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load base model with LoRA adapter, optimized for available hardware."""
    model_name = config["base_model"]
    print(f"[TRAIN] Loading base model: {model_name}", flush=True)

    # Determine device and dtype first
    device = "cpu" if config.get("force_cpu") else ("auto" if torch.cuda.is_available() else "cpu")
    
    gpu_mem = 0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    
    use_4bit = torch.cuda.is_available() and gpu_mem < 20e9 and device != "cpu"
    use_fp16 = torch.cuda.is_available() and gpu_mem < 8e9 and device != "cpu"
    dtype = torch.float32 if device == "cpu" else (torch.float16 if use_fp16 else torch.bfloat16)

    if use_4bit:
        print(f"[TRAIN] Using 4-bit quantization (QLoRA) for {gpu_mem/1e9:.1f}GB GPU", flush=True)
        compute_dtype = torch.float16 if use_fp16 else torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif device == "cpu":
        print(f"[TRAIN] Training on CPU (Stability Mode)", flush=True)
        bnb_config = None
    else:
        print(f"[TRAIN] Training on GPU (Standard Mode)", flush=True)
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[TRAIN] LoRA adapter: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)", flush=True)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt / episode generation
# ---------------------------------------------------------------------------

def generate_prompt_and_episode_config() -> Dict[str, Any]:
    """Generate a training prompt by sampling a task and building the observation."""
    task_id, scenario_id = sample_task()
    env = SocEnvironment()
    reset_result = env.reset(task_id=task_id, scenario_id=scenario_id)
    obs_dict = reset_result.observation.model_dump()

    prompt = build_prompt_for_task(task_id, scenario_id, obs_dict)

    return {
        "prompt": prompt,
        "task_id": task_id,
        "scenario_id": scenario_id,
    }


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    k: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[str]:
    """Generate K candidate action sequences from the policy model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the generated tokens (strip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for seq in outputs:
        generated = seq[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append(text)

    return completions


def score_completions(
    completions: List[str],
    task_id: str,
    scenario_id: Optional[str] = None,
) -> List[float]:
    """Execute each completion in the environment and return scores."""
    env = SocEnvironment()
    scores = []
    for completion in completions:
        try:
            actions = parse_actions_from_text(completion)
            if not actions:
                scores.append(0.001)
                continue
            score, _, _, _ = execute_episode(env, task_id, actions, scenario_id)
            scores.append(score)
        except Exception as e:
            print(f"[TRAIN] Error scoring completion: {e}", flush=True)
            scores.append(0.001)
    return scores


def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO advantage computation: A_i = (r_i - mean(r)) / std(r)
    Group-relative normalization — key innovation of GRPO.
    """
    import numpy as np
    rewards_np = np.array(rewards, dtype=np.float64)
    mean_r = rewards_np.mean()
    std_r = rewards_np.std()
    if std_r < 1e-8:
        # All rewards identical → zero advantage
        return [0.0] * len(rewards)
    advantages = ((rewards_np - mean_r) / std_r).tolist()
    return advantages


# ---------------------------------------------------------------------------
# Manual GRPO training loop (fallback when TRL not available)
# ---------------------------------------------------------------------------

def train_manual_grpo(config: Dict[str, Any]):
    """
    Manual GRPO training loop that doesn't require TRL.
    Implements the core GRPO algorithm directly.
    """
    print("[TRAIN] === Manual GRPO Training Loop ===", flush=True)

    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    # Learning rate scheduler — warmup then constant (better for short runs than cosine)
    warmup = config["warmup_steps"]
    total_steps_sched = config["training_steps"]
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        # Stay at max LR after warmup (constant) — cosine decays too fast for short runs
        return 1.0
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # WandB init
    if WANDB_AVAILABLE and os.getenv("WANDB_API_KEY"):
        wandb.init(
            project=config["wandb_project"],
            config=config,
            name=f"grpo-{config['base_model'].split('/')[-1]}-{time.strftime('%m%d-%H%M')}",
        )

    # Metrics tracking
    metrics_history = []
    os.makedirs(config["output_dir"], exist_ok=True)

    # Training
    K = config["grpo_group_size"]
    total_steps = config["training_steps"]
    best_mean_score = 0.0

    print(f"[TRAIN] Starting training: {total_steps} steps, K={K} group size", flush=True)
    print(f"[TRAIN] Task sampling weights: {TASK_SAMPLE_WEIGHTS}", flush=True)

    for step in range(1, total_steps + 1):
        step_start = time.time()

        # 1. Sample a training episode configuration
        episode_config = generate_prompt_and_episode_config()
        prompt = episode_config["prompt"]
        task_id = episode_config["task_id"]
        scenario_id = episode_config["scenario_id"]

        # 2. Generate K candidate action sequences
        try:
            completions = generate_completions(
                model, tokenizer, prompt,
                k=K,
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
            )
        except Exception as e:
            print(f"[TRAIN] Generation error at step {step}: {e}", flush=True)
            continue

        # 3. Score each completion in the environment
        scores = score_completions(completions, task_id, scenario_id)

        # 4. Compute GRPO advantages
        advantages = compute_grpo_advantages(scores)

        # 5. Policy gradient update
        # For each completion, compute log probability under current policy
        # then multiply by advantage for the GRPO objective
        optimizer.zero_grad()
        total_loss = 0.0

        try:
            for completion, advantage in zip(completions, advantages):
                if abs(advantage) < 1e-8:
                    continue  # Skip zero-advantage samples

                # Tokenize the full sequence (prompt + completion)
                full_text = prompt + "\n" + completion
                tokens = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config["max_seq_length"],
                )
                tokens = {k: v.to(model.device) for k, v in tokens.items()}

                # Forward pass to get log probabilities
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(**tokens, labels=tokens["input_ids"])
                    # GRPO loss: -advantage * log_prob (maximize advantage-weighted log likelihood)
                    loss = -advantage * outputs.loss
                    loss = loss / K  # Average over group

                loss.backward()
                total_loss += loss.item()
        except Exception as e:
            print(f"[TRAIN] Error during backward pass at step {step}: {e}", flush=True)
            optimizer.zero_grad() # Clear gradients if an error occurs to avoid applying broken gradients
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # KL penalty (approximate via loss regularization)
        # In full GRPO, this uses the reference model. Here we use weight decay as proxy.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Free CUDA memory between steps (critical for 4GB GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        step_time = time.time() - step_start
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)

        # Log metrics
        metrics = {
            "step": step,
            "task_id": task_id,
            "scenario_id": scenario_id or "mixed",
            "mean_score": round(mean_score, 4),
            "max_score": round(max_score, 4),
            "min_score": round(min(scores), 4),
            "loss": round(total_loss, 6),
            "lr": scheduler.get_last_lr()[0],
            "step_time_s": round(step_time, 2),
            "advantages_std": round(sum(a**2 for a in advantages) / len(advantages), 4) ** 0.5 if advantages else 0,
        }
        metrics_history.append(metrics)

        # Console logging (every step for short runs, every 5 for long)
        if step % 5 == 0 or step == 1 or total_steps <= 20:
            vram_str = ""
            if torch.cuda.is_available():
                used_mb = torch.cuda.memory_allocated() / 1e6
                total_mb = _gpu_mem / 1e6
                vram_str = f" vram={used_mb:.0f}/{total_mb:.0f}MB"
            print(
                f"[STEP {step:>4}/{total_steps}] "
                f"task={task_id:<24} "
                f"mean={mean_score:.4f} max={max_score:.4f} "
                f"loss={total_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
                f"{vram_str} ({step_time:.1f}s)",
                flush=True,
            )

        # WandB logging
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(metrics)

        # Evaluation
        if step % config["eval_every"] == 0:
            eval_scores = run_evaluation(model, tokenizer, config)
            print(f"[EVAL  step={step}] {eval_scores}", flush=True)
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({f"eval/{k}": v for k, v in eval_scores.items()})

        # Save checkpoint
        if step % config["save_every"] == 0 or step == total_steps:
            ckpt_dir = os.path.join(config["output_dir"], f"checkpoint-{step}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[SAVE] Checkpoint saved to {ckpt_dir}", flush=True)

            # Track best
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_dir = os.path.join(config["output_dir"], "best")
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"[SAVE] New best model (score={mean_score:.4f}) saved to {best_dir}", flush=True)

    # Save metrics history
    metrics_path = os.path.join(config["output_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"[DONE] Training complete. Metrics saved to {metrics_path}", flush=True)

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()

    return metrics_history


# ---------------------------------------------------------------------------
# TRL-based GRPO training (when TRL is available)
# ---------------------------------------------------------------------------

def train_trl_grpo(config: Dict[str, Any]):
    """
    GRPO training using HuggingFace TRL's GRPOTrainer.
    This is the preferred path when TRL >= 0.12.0 is installed.
    """
    print("[TRAIN] === TRL GRPOTrainer ===", flush=True)

    model, tokenizer = load_model_and_tokenizer(config)

    # Build training dataset (prompts)
    from datasets import Dataset
    prompts = []
    task_ids = []
    scenario_ids = []

    num_prompts = config["training_steps"] * 2  # Oversample to allow shuffling
    print(f"[TRAIN] Generating {num_prompts} training prompts...", flush=True)

    for _ in range(num_prompts):
        ep = generate_prompt_and_episode_config()
        prompts.append(ep["prompt"])
        task_ids.append(ep["task_id"])
        scenario_ids.append(ep["scenario_id"] or "")

    train_dataset = Dataset.from_dict({
        "prompt": prompts,
        "task_id": task_ids,
        "scenario_id": scenario_ids,
    })

    # Reward function for TRL
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """TRL-compatible reward function."""
        batch_task_ids = kwargs.get("task_id", ["alert_triage"] * len(completions))
        batch_scenario_ids = kwargs.get("scenario_id", [""] * len(completions))
        rewards = []
        env = SocEnvironment()
        for comp, tid, sid in zip(completions, batch_task_ids, batch_scenario_ids):
            try:
                actions = parse_actions_from_text(comp)
                if not actions:
                    rewards.append(0.001)
                    continue
                score, _, _, _ = execute_episode(env, tid, actions, sid or None)
                rewards.append(score)
            except Exception:
                rewards.append(0.001)
        return rewards

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=1,
        max_steps=config["training_steps"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=5,
        save_steps=config["save_every"],
        eval_steps=config["eval_every"],
        max_completion_length=config["max_new_tokens"],
        num_generations=config["grpo_group_size"],
        temperature=config["temperature"],
        report_to="wandb" if (WANDB_AVAILABLE and os.getenv("WANDB_API_KEY")) else "none",
        run_name=f"grpo-minisoc-{time.strftime('%m%d-%H%M')}",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    # Train
    print("[TRAIN] Starting GRPOTrainer.train()...", flush=True)
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(config["output_dir"], "final"))
    print(f"[DONE] TRL training complete. Model saved to {config['output_dir']}/final", flush=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model, tokenizer, config: Dict[str, Any]) -> Dict[str, float]:
    """Run one episode per task and return scores."""
    eval_scores = {}
    env = SocEnvironment()

    for task_id in ["alert_triage", "incident_investigation", "threat_response"]:
        scenarios = TASK_CONFIG[task_id].get("scenarios", [None])
        scenario_id = scenarios[0]  # Use default scenario for eval

        ep = generate_prompt_and_episode_config.__wrapped__() if hasattr(generate_prompt_and_episode_config, '__wrapped__') else None
        # Build eval prompt
        reset_result = env.reset(task_id=task_id, scenario_id=scenario_id)
        obs_dict = reset_result.observation.model_dump()
        prompt = build_prompt_for_task(task_id, scenario_id, obs_dict)

        # Generate single best completion (greedy)
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=False,  # Greedy for eval
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            actions = parse_actions_from_text(completion)

            if actions:
                score, _, _, _ = execute_episode(env, task_id, actions, scenario_id)
            else:
                score = 0.001
        except Exception as e:
            print(f"[EVAL] Error on {task_id}: {e}", flush=True)
            score = 0.001

        eval_scores[task_id] = round(score, 4)

    eval_scores["overall"] = round(sum(eval_scores.values()) / 3, 4)
    return eval_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mini SOC — GRPO Training Script")
    parser.add_argument("--model", default=DEFAULT_CONFIG["base_model"],
                        help=f"Base model (default: {DEFAULT_CONFIG['base_model']})")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG["training_steps"],
                        help=f"Training steps (default: {DEFAULT_CONFIG['training_steps']})")
    parser.add_argument("--task", default="all", choices=["all", "alert_triage", "incident_investigation", "threat_response"],
                        help="Focus on specific task or all (default: all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})")
    parser.add_argument("--group-size", type=int, default=DEFAULT_CONFIG["grpo_group_size"],
                        help=f"GRPO group size K (default: {DEFAULT_CONFIG['grpo_group_size']})")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"],
                        help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})")
    parser.add_argument("--use-trl", action="store_true", default=False,
                        help="Force use of TRL GRPOTrainer (requires trl>=0.12.0)")
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable WandB logging")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU training (slower but stable)")

    args = parser.parse_args()

    # Build config
    config = dict(DEFAULT_CONFIG)
    config["base_model"] = args.model
    config["training_steps"] = args.steps
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["grpo_group_size"] = args.group_size
    config["output_dir"] = args.output_dir
    config["force_cpu"] = args.cpu

    if args.no_wandb:
        os.environ.pop("WANDB_API_KEY", None)

    # Task filtering
    if args.task != "all":
        # Override sampling to focus on one task
        from train import reward_wrapper
        reward_wrapper.TASK_SAMPLE_WEIGHTS = {args.task: 1.0}
        for t in ["alert_triage", "incident_investigation", "threat_response"]:
            if t != args.task:
                reward_wrapper.TASK_SAMPLE_WEIGHTS[t] = 0.0

    print("=" * 70, flush=True)
    print("[TRAIN] Mini SOC — GRPO Training Pipeline", flush=True)
    print(f"[TRAIN] Model:      {config['base_model']}", flush=True)
    print(f"[TRAIN] Steps:      {config['training_steps']}", flush=True)
    print(f"[TRAIN] LoRA:       r={config['lora_r']}, alpha={config['lora_alpha']}", flush=True)
    print(f"[TRAIN] Group size: K={config['grpo_group_size']}", flush=True)
    print(f"[TRAIN] LR:         {config['learning_rate']}", flush=True)
    print(f"[TRAIN] Output:     {config['output_dir']}", flush=True)
    print("=" * 70, flush=True)

    # Select training backend
    if args.use_trl and TRL_AVAILABLE:
        train_trl_grpo(config)
    else:
        if args.use_trl and not TRL_AVAILABLE:
            print("[WARN] TRL requested but not installed. Falling back to manual loop.", flush=True)
        train_manual_grpo(config)


if __name__ == "__main__":
    main()
