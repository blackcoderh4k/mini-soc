"""
Mini SOC — Reward Curve Plotter
=================================
Generates before/after reward improvement charts from training metrics.
Produces publication-quality plots for the judging submission.

Usage:
  python train/plot_rewards.py                          # Use default metrics.json
  python train/plot_rewards.py --metrics path/to.json   # Custom metrics file
  python train/plot_rewards.py --demo                   # Generate demo chart with synthetic data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed. Run: pip install matplotlib numpy", flush=True)


# ---------------------------------------------------------------------------
# Color palette (matches PRD / dark premium aesthetic)
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0f1117",
    "card_bg": "#1a1d29",
    "text": "#e0e0e0",
    "text_dim": "#8888aa",
    "grid": "#2a2d3a",
    "alert_triage": "#00d4aa",       # teal
    "incident_investigation": "#6366f1",  # indigo
    "threat_response": "#f43f5e",    # rose
    "overall": "#f59e0b",            # amber
    "accent": "#818cf8",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}

TASK_LABELS = {
    "alert_triage": "Task 1: Alert Triage",
    "incident_investigation": "Task 2: Investigation",
    "threat_response": "Task 3: Threat Response",
}


def smooth(values: List[float], window: int = 10) -> List[float]:
    """Exponential moving average smoothing."""
    if len(values) < 2:
        return values
    result = [values[0]]
    alpha = 2.0 / (window + 1)
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Load metrics from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def generate_demo_metrics(steps: int = 200) -> List[Dict[str, Any]]:
    """
    Generate realistic synthetic training metrics for demo purposes.
    Simulates the expected GRPO training curve from PRD §12.4.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    metrics = []
    tasks = ["alert_triage", "incident_investigation", "threat_response"]
    scenarios = {
        "alert_triage": [None],
        "incident_investigation": ["brute_force_ssh_001", "ransomware_001", "insider_threat_001"],
        "threat_response": ["phishing_lateral_001", "supply_chain_001", "multi_stage_apt_001"],
    }

    # Baseline scores and target scores (from PRD)
    baseline = {"alert_triage": 0.15, "incident_investigation": 0.08, "threat_response": 0.04}
    target_200 = {"alert_triage": 0.52, "incident_investigation": 0.35, "threat_response": 0.18}

    for step in range(1, steps + 1):
        task = random.choice(tasks)
        scenario = random.choice(scenarios[task])

        # Simulate learning curve: logarithmic growth with noise
        progress = step / steps
        base = baseline[task]
        target = target_200[task]

        # Logarithmic growth with some plateaus and jumps
        growth = np.log1p(progress * 3) / np.log1p(3)  # 0 → 1
        score = base + (target - base) * growth

        # Add realistic noise (decreasing over time)
        noise = np.random.normal(0, 0.05 * (1 - progress * 0.5))
        score = max(0.001, min(0.999, score + noise))

        # Loss decays
        loss = 0.5 * np.exp(-progress * 2) + np.random.normal(0, 0.02)

        metrics.append({
            "step": step,
            "task_id": task,
            "scenario_id": scenario or "mixed",
            "mean_score": round(score, 4),
            "max_score": round(min(score + abs(np.random.normal(0.05, 0.03)), 0.999), 4),
            "min_score": round(max(score - abs(np.random.normal(0.08, 0.04)), 0.001), 4),
            "loss": round(max(loss, 0.001), 6),
            "lr": round(2e-5 * (1 - progress * 0.5), 8),
            "step_time_s": round(random.uniform(8, 25), 2),
        })

    return metrics


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_reward_curves(metrics: List[Dict[str, Any]], output_path: str, title: str = "Mini SOC — GRPO Training"):
    """
    Generate the main reward improvement chart.
    Shows per-task smoothed reward curves and overall average.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=COLORS["bg"])
    fig.suptitle(title, fontsize=20, fontweight="bold", color=COLORS["text"], y=0.98)

    # Organize metrics by task
    task_metrics = {"alert_triage": [], "incident_investigation": [], "threat_response": []}
    all_scores = []

    for m in metrics:
        tid = m["task_id"]
        if tid in task_metrics:
            task_metrics[tid].append(m)
        all_scores.append(m)

    # Plot 1-3: Per-task reward curves
    for idx, (task_id, task_data) in enumerate(task_metrics.items()):
        ax = axes[idx // 2][idx % 2]
        ax.set_facecolor(COLORS["card_bg"])

        if not task_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    color=COLORS["text_dim"], ha="center", fontsize=14)
            continue

        steps = [m["step"] for m in task_data]
        scores = [m["mean_score"] for m in task_data]
        max_scores = [m.get("max_score", m["mean_score"]) for m in task_data]
        min_scores = [m.get("min_score", m["mean_score"]) for m in task_data]
        smoothed = smooth(scores, window=15)

        color = COLORS[task_id]

        # Raw scores (faded)
        ax.scatter(steps, scores, alpha=0.15, s=8, color=color, zorder=2)

        # Smoothed curve
        ax.plot(steps, smoothed, color=color, linewidth=2.5, label="Smoothed reward", zorder=3)

        # Min-max band
        smoothed_max = smooth(max_scores, window=15)
        smoothed_min = smooth(min_scores, window=15)
        ax.fill_between(steps, smoothed_min, smoothed_max, alpha=0.12, color=color, zorder=1)

        # Baseline reference line
        baseline_map = {"alert_triage": 0.15, "incident_investigation": 0.08, "threat_response": 0.04}
        baseline = baseline_map.get(task_id, 0.1)
        ax.axhline(y=baseline, color=COLORS["text_dim"], linestyle="--", linewidth=1, alpha=0.7)
        ax.text(steps[-1] * 0.02, baseline + 0.02, f"Random baseline: {baseline:.2f}",
                color=COLORS["text_dim"], fontsize=9)

        # Final score annotation
        final_score = smoothed[-1] if smoothed else 0
        ax.annotate(
            f"{final_score:.3f}",
            xy=(steps[-1], final_score),
            xytext=(steps[-1] - len(steps) * 0.15, final_score + 0.08),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            fontsize=12, fontweight="bold", color=color,
        )

        ax.set_title(TASK_LABELS.get(task_id, task_id), fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
        ax.set_xlabel("Training Step", fontsize=11, color=COLORS["text_dim"])
        ax.set_ylabel("Episode Score", fontsize=11, color=COLORS["text_dim"])
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(colors=COLORS["text_dim"])
        ax.grid(True, alpha=0.2, color=COLORS["grid"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    # Plot 4: Overall combined curve
    ax = axes[1][1]
    ax.set_facecolor(COLORS["card_bg"])

    if all_scores:
        # Compute rolling overall average across all tasks
        window = 20
        rolling_scores = []
        for i in range(len(all_scores)):
            start = max(0, i - window)
            window_scores = [m["mean_score"] for m in all_scores[start:i + 1]]
            rolling_scores.append(sum(window_scores) / len(window_scores))

        all_steps = [m["step"] for m in all_scores]
        overall_smooth = smooth([m["mean_score"] for m in all_scores], window=20)

        # Per-task colored dots
        for task_id, color_key in [("alert_triage", "alert_triage"),
                                    ("incident_investigation", "incident_investigation"),
                                    ("threat_response", "threat_response")]:
            t_steps = [m["step"] for m in all_scores if m["task_id"] == task_id]
            t_scores = [m["mean_score"] for m in all_scores if m["task_id"] == task_id]
            ax.scatter(t_steps, t_scores, alpha=0.2, s=6, color=COLORS[color_key],
                      label=TASK_LABELS.get(task_id, task_id))

        # Overall smoothed line
        ax.plot(all_steps, overall_smooth, color=COLORS["overall"], linewidth=3,
                label="Overall (smoothed)", zorder=5)

        # Baseline
        overall_baseline = 0.09
        ax.axhline(y=overall_baseline, color=COLORS["text_dim"], linestyle="--", linewidth=1, alpha=0.7)

        final_overall = overall_smooth[-1] if overall_smooth else 0
        improvement = ((final_overall - overall_baseline) / overall_baseline) * 100

        ax.set_title(
            f"Overall Performance  (↑{improvement:.0f}% from baseline)",
            fontsize=14, fontweight="bold", color=COLORS["overall"], pad=10
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.3,
                 labelcolor=COLORS["text"], facecolor=COLORS["card_bg"], edgecolor=COLORS["grid"])

    ax.set_xlabel("Training Step", fontsize=11, color=COLORS["text_dim"])
    ax.set_ylabel("Episode Score", fontsize=11, color=COLORS["text_dim"])
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors=COLORS["text_dim"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[PLOT] Reward curves saved to {output_path}", flush=True)


def plot_before_after(metrics: List[Dict[str, Any]], output_path: str):
    """
    Generate a before/after comparison bar chart.
    Shows random agent vs trained agent scores per task.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["card_bg"])

    tasks = ["alert_triage", "incident_investigation", "threat_response"]
    task_labels = [TASK_LABELS[t] for t in tasks]
    baseline_scores = [0.15, 0.08, 0.04]

    # Get final trained scores from metrics (last 20 episodes per task)
    trained_scores = []
    for task_id in tasks:
        task_data = [m for m in metrics if m["task_id"] == task_id]
        if task_data:
            last_n = task_data[-20:]
            avg = sum(m["mean_score"] for m in last_n) / len(last_n)
            trained_scores.append(avg)
        else:
            trained_scores.append(baseline_scores[tasks.index(task_id)])

    x = np.arange(len(tasks))
    width = 0.35

    bars_before = ax.bar(x - width / 2, baseline_scores, width,
                         label="Random Agent (Before)", color="#4a4a6a", edgecolor=COLORS["grid"], linewidth=1)
    bars_after = ax.bar(x + width / 2, trained_scores, width,
                        label="GRPO-Trained (After)", color=COLORS["accent"], edgecolor=COLORS["grid"], linewidth=1)

    # Value labels
    for bar, score in zip(bars_before, baseline_scores):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{score:.2f}", ha="center", va="bottom", fontsize=11, color=COLORS["text_dim"])

    for bar, score in zip(bars_after, trained_scores):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{score:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=COLORS["text"])

    # Improvement arrows
    for i, (before, after) in enumerate(zip(baseline_scores, trained_scores)):
        if after > before:
            pct = ((after - before) / before) * 100
            ax.annotate(
                f"+{pct:.0f}%",
                xy=(x[i] + width / 2, after),
                xytext=(x[i] + width / 2 + 0.2, after + 0.06),
                fontsize=10, fontweight="bold", color=COLORS["overall"],
                arrowprops=dict(arrowstyle="->", color=COLORS["overall"], lw=1),
            )

    ax.set_title("Mini SOC — Before vs After GRPO Training", fontsize=16, fontweight="bold",
                 color=COLORS["text"], pad=15)
    ax.set_ylabel("Episode Score", fontsize=12, color=COLORS["text_dim"])
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11, color=COLORS["text"])
    ax.set_ylim(0, max(max(trained_scores), max(baseline_scores)) * 1.3)
    ax.legend(fontsize=11, framealpha=0.3, labelcolor=COLORS["text"],
              facecolor=COLORS["card_bg"], edgecolor=COLORS["grid"])
    ax.tick_params(colors=COLORS["text_dim"])
    ax.grid(True, axis="y", alpha=0.2, color=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[PLOT] Before/After chart saved to {output_path}", flush=True)


def plot_loss_curve(metrics: List[Dict[str, Any]], output_path: str):
    """Plot training loss over steps."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["card_bg"])

    steps = [m["step"] for m in metrics]
    losses = [m.get("loss", 0) for m in metrics]
    smoothed_loss = smooth(losses, window=15)

    ax.plot(steps, losses, alpha=0.25, color=COLORS["accent"], linewidth=1)
    ax.plot(steps, smoothed_loss, color=COLORS["accent"], linewidth=2.5, label="Smoothed loss")

    ax.set_title("Training Loss", fontsize=16, fontweight="bold", color=COLORS["text"], pad=10)
    ax.set_xlabel("Training Step", fontsize=12, color=COLORS["text_dim"])
    ax.set_ylabel("Loss", fontsize=12, color=COLORS["text_dim"])
    ax.tick_params(colors=COLORS["text_dim"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    ax.legend(fontsize=10, framealpha=0.3, labelcolor=COLORS["text"],
              facecolor=COLORS["card_bg"], edgecolor=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[PLOT] Loss curve saved to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mini SOC — Reward Curve Plotter")
    parser.add_argument("--metrics", default=str(PROJECT_ROOT / "outputs" / "grpo_checkpoints" / "metrics.json"),
                        help="Path to metrics.json from training")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "plots"),
                        help="Directory to save plots")
    parser.add_argument("--demo", action="store_true", default=False,
                        help="Generate demo charts with synthetic data")

    args = parser.parse_args()

    if not HAS_MPL:
        print("[FATAL] matplotlib is required. Install: pip install matplotlib numpy", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate metrics
    if args.demo:
        print("[PLOT] Generating demo charts with synthetic data...", flush=True)
        metrics = generate_demo_metrics(steps=200)
        # Save synthetic metrics for reference
        demo_path = os.path.join(args.output_dir, "demo_metrics.json")
        with open(demo_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[PLOT] Demo metrics saved to {demo_path}", flush=True)
    else:
        if not os.path.exists(args.metrics):
            print(f"[ERROR] Metrics file not found: {args.metrics}", flush=True)
            print("[TIP] Run training first, or use --demo for synthetic charts.", flush=True)
            sys.exit(1)
        metrics = load_metrics(args.metrics)

    print(f"[PLOT] Loaded {len(metrics)} training steps", flush=True)

    # Generate all plots
    plot_reward_curves(
        metrics,
        os.path.join(args.output_dir, "reward_curves.png"),
        title="Mini SOC — GRPO Training Reward Curves"
    )
    plot_before_after(
        metrics,
        os.path.join(args.output_dir, "before_after.png"),
    )
    plot_loss_curve(
        metrics,
        os.path.join(args.output_dir, "loss_curve.png"),
    )

    print(f"\n[DONE] All plots saved to {args.output_dir}/", flush=True)
    print(f"  -> reward_curves.png   (4-panel per-task + overall)", flush=True)
    print(f"  -> before_after.png    (bar chart comparison)", flush=True)
    print(f"  -> loss_curve.png      (training loss)", flush=True)


if __name__ == "__main__":
    main()
