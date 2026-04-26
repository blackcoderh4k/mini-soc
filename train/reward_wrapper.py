"""
Mini SOC — Reward Wrapper for TRL GRPO
========================================
Adapts the SocEnvironment into a reward function compatible with
HuggingFace TRL's GRPOTrainer.

The wrapper:
  1. Takes generated text (action sequences) from the LLM policy
  2. Parses them into environment actions
  3. Executes a full episode in the Mini SOC environment
  4. Returns the final graded score as the reward signal

This is the bridge between the language model's text output and the
environment's numerical reward.
"""
from __future__ import annotations

import json
import re
import sys
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.mini_soc_environment import SocEnvironment, TASK_CONFIG
from server.simulator.attack_seeds import ATTACK_SCENARIOS


# ---------------------------------------------------------------------------
# System prompts (reused from inference.py but tuned for GRPO generation)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "alert_triage": textwrap.dedent("""\
        You are a Tier-1 SOC analyst. Classify every alert in the queue.
        Output ONLY raw JSON lines — no prose, no markdown, no explanation.
        One JSON object per line, exactly this format:
        {"action_type": "classify_alert", "parameters": {"alert_id": "ALT-XXX", "classification": "benign|suspicious|critical", "priority": "P1|P2|P3|P4"}}

        Classification guide:
          critical/P1 = confirmed active attack or compromise (brute force success, C2 beacon, lateral movement)
          suspicious/P2 = anomalous behaviour requiring investigation (encoded commands, Tor connection)
          suspicious/P3 = low-confidence anomaly (after-hours login with explanation)
          benign/P4 = confirmed authorized or scheduled activity (IT scanner, backup, helpdesk reset)

        EXAMPLE OUTPUT (do exactly this):
        {"action_type": "classify_alert", "parameters": {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}}
        {"action_type": "classify_alert", "parameters": {"alert_id": "ALT-002", "classification": "suspicious", "priority": "P2"}}
        {"action_type": "classify_alert", "parameters": {"alert_id": "ALT-020", "classification": "benign", "priority": "P4"}}"""),

    "incident_investigation": textwrap.dedent("""\
        You are a SOC analyst investigating a security incident.
        Output ONLY raw JSON lines — no prose, no markdown, no explanation.
        One JSON object per line.

        Available actions:
          {"action_type": "query_logs", "parameters": {"log_source": "auth"}}
          {"action_type": "query_logs", "parameters": {"log_source": "firewall"}}
          {"action_type": "close_incident", "parameters": {"incident_id": "INC-XXXX", "verdict": "true_positive", "attack_type": "brute_force", "attacker_ip": "1.2.3.4"}}

        Strategy: query auth logs, then firewall logs, then close the incident with your verdict.

        EXAMPLE OUTPUT (do exactly this):
        {"action_type": "query_logs", "parameters": {"log_source": "auth"}}
        {"action_type": "query_logs", "parameters": {"log_source": "firewall"}}
        {"action_type": "close_incident", "parameters": {"incident_id": "INC-abc123", "verdict": "true_positive", "attack_type": "brute_force", "attacker_ip": "185.220.101.47"}}"""),

    "threat_response": textwrap.dedent("""\
        You are a senior SOC analyst responding to an active threat.
        Output ONLY raw JSON lines — no prose, no markdown, no explanation.
        One JSON object per line.

        Available actions:
          {"action_type": "query_logs", "parameters": {"log_source": "process"}}
          {"action_type": "isolate_asset", "parameters": {"hostname": "WS-HR-03"}}
          {"action_type": "block_ip", "parameters": {"ip_address": "1.2.3.4"}}
          {"action_type": "write_report", "parameters": {"report": {"summary": "...", "attack_type": "lateral_movement", "affected_assets": ["WS-HR-03"], "attacker_ip": "1.2.3.4", "timeline": "..."}}}
          {"action_type": "close_incident", "parameters": {"incident_id": "INC-XXXX", "verdict": "true_positive", "attack_type": "lateral_movement", "attacker_ip": "1.2.3.4"}}

        CRITICAL: Only isolate assets you have confirmed are compromised. Never isolate healthy servers.
        Strategy: query process/network/auth/dns logs → isolate compromised host → block attacker IP → write report → close.

        EXAMPLE OUTPUT (do exactly this):
        {"action_type": "query_logs", "parameters": {"log_source": "process"}}
        {"action_type": "query_logs", "parameters": {"log_source": "network"}}
        {"action_type": "isolate_asset", "parameters": {"hostname": "WS-HR-03"}}
        {"action_type": "block_ip", "parameters": {"ip_address": "94.102.49.190"}}
        {"action_type": "write_report", "parameters": {"report": {"summary": "Phishing to C2 beacon", "attack_type": "lateral_movement", "affected_assets": ["WS-HR-03"], "attacker_ip": "94.102.49.190", "timeline": "09:45 phishing detected"}}}
        {"action_type": "close_incident", "parameters": {"incident_id": "INC-abc123", "verdict": "true_positive", "attack_type": "lateral_movement", "attacker_ip": "94.102.49.190"}}"""),
}


def build_prompt_for_task(task_id: str, scenario_id: Optional[str], obs_dict: Dict[str, Any]) -> str:
    """
    Build a full prompt string from environment observation.
    This is what the LLM policy sees as input.
    """
    system = SYSTEM_PROMPTS[task_id]
    ctx = obs_dict.get("task_context", {}) or {}
    current_alert = obs_dict.get("current_alert") or {}
    alert_queue = obs_dict.get("alert_queue", [])
    logs = obs_dict.get("available_logs", [])
    assets = obs_dict.get("asset_inventory", [])
    incidents = obs_dict.get("open_incidents", [])
    message = obs_dict.get("message", "")

    # Alert queue summary
    queue_lines = []
    for a in alert_queue:
        q = a if isinstance(a, dict) else a.model_dump() if hasattr(a, "model_dump") else {}
        queue_lines.append(
            f"  [{q.get('severity', '?').upper()}] {q.get('alert_id', '?')}: "
            f"{q.get('alert_type', '')} — {q.get('description', '')[:100]}"
        )

    # Asset summary
    asset_lines = []
    for a in assets:
        ad = a if isinstance(a, dict) else a.model_dump() if hasattr(a, "model_dump") else {}
        asset_lines.append(
            f"  {ad.get('hostname', '?')} ({ad.get('ip_address', '?')}) "
            f"type={ad.get('asset_type', '?')} criticality={ad.get('criticality', '?')} "
            f"isolated={ad.get('is_isolated', False)}"
        )

    # Incident summary
    inc_lines = []
    for inc in incidents:
        idict = inc if isinstance(inc, dict) else inc.model_dump() if hasattr(inc, "model_dump") else {}
        inc_lines.append(f"  {idict.get('incident_id', '?')}: status={idict.get('status', '?')}")

    prompt = f"""{system}

--- ENVIRONMENT STATE ---
Task: {task_id} | Scenario: {scenario_id or 'mixed'}
Objective: {ctx.get('objective', '')}
Message: {message}

ALERT QUEUE ({len(alert_queue)} alerts):
{chr(10).join(queue_lines) if queue_lines else '  (empty)'}

ASSET INVENTORY ({len(assets)} assets):
{chr(10).join(asset_lines) if asset_lines else '  (empty)'}

OPEN INCIDENTS:
{chr(10).join(inc_lines) if inc_lines else '  None'}

--- YOUR ACTIONS (one JSON per line) ---"""

    return prompt.strip()


def parse_actions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM-generated text into a list of action dicts.
    Handles:
      - One JSON per line
      - JSON wrapped in markdown code fences
      - Partial/malformed JSON (best effort)
    """
    actions = []

    # Strip markdown code fences
    clean = text.strip()
    if clean.startswith("```"):
        # Remove first and last ``` lines
        lines = clean.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines)

    # Try to find JSON objects
    # Strategy 1: one JSON per line
    for line in clean.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        # Find JSON substring in the line
        match = re.search(r'\{.*\}', line)
        if match:
            try:
                obj = json.loads(match.group())
                if "action_type" in obj:
                    actions.append(obj)
                elif "parameters" in obj:
                    # Might be just parameters without action_type
                    continue
            except json.JSONDecodeError:
                continue

    # Strategy 2: if no line-by-line matches, try the whole text as a JSON array
    if not actions:
        try:
            arr = json.loads(clean)
            if isinstance(arr, list):
                actions = [a for a in arr if isinstance(a, dict) and "action_type" in a]
            elif isinstance(arr, dict) and "action_type" in arr:
                actions = [arr]
        except json.JSONDecodeError:
            pass

    return actions


def execute_episode(
    env: SocEnvironment,
    task_id: str,
    actions: List[Dict[str, Any]],
    scenario_id: Optional[str] = None,
) -> Tuple[float, float, int, List[float]]:
    """
    Execute a full episode given a list of parsed actions.

    Returns:
        (final_score, total_step_reward, steps_taken, per_step_rewards)
    """
    from models import Action, ActionType

    # Reset environment
    reset_result = env.reset(task_id=task_id, scenario_id=scenario_id)

    per_step_rewards = []
    total_step_reward = 0.0
    steps_taken = 0

    for action_dict in actions:
        action_type_str = action_dict.get("action_type", "request_info")
        parameters = action_dict.get("parameters", {})

        # Validate action type
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.REQUEST_INFO
            parameters = {}

        action = Action(action_type=action_type, parameters=parameters)
        step_result = env.step(action)

        reward = step_result.reward
        per_step_rewards.append(reward)
        total_step_reward += reward
        steps_taken += 1

        if step_result.done:
            break

    # Get final graded score
    state = env.state()
    final_score = state.ground_truth  # This is the ground truth dict
    # Actually compute final score from grader
    from server.mini_soc_environment import TASK_CONFIG
    grader = TASK_CONFIG[task_id]["grader"]
    grader_state = {
        "agent_classifications": env._agent_classifications,
        "agent_queried_log_ids": env._agent_queried_log_ids,
        "agent_queried_sources": env._agent_queried_sources,
        "agent_isolated_assets": env._agent_isolated_assets,
        "agent_blocked_ips": env._agent_blocked_ips,
        "agent_verdict": env._agent_verdict,
        "agent_attack_type": env._agent_attack_type,
        "agent_attacker_ip": env._agent_attacker_ip,
        "agent_report": env._agent_report,
        "steps_taken": steps_taken,
        "max_steps": TASK_CONFIG[task_id]["max_steps"],
        "ground_truth": env._build_ground_truth(),
    }
    final_score = grader.grade(grader_state)

    return final_score, total_step_reward, steps_taken, per_step_rewards


def compute_reward(
    prompts: List[str],
    completions: List[str],
    task_ids: List[str],
    scenario_ids: Optional[List[str]] = None,
) -> List[float]:
    """
    TRL-compatible reward function.

    Args:
        prompts: The input prompts given to the model (one per sample).
        completions: The generated action sequences (one per sample).
        task_ids: Which task each sample belongs to.
        scenario_ids: Optional scenario override per sample.

    Returns:
        List of float rewards (one per sample).
    """
    rewards = []
    env = SocEnvironment()

    for i, (prompt, completion, task_id) in enumerate(zip(prompts, completions, task_ids)):
        scenario_id = scenario_ids[i] if scenario_ids else None
        try:
            actions = parse_actions_from_text(completion)
            if not actions:
                # No valid actions parsed → minimum reward
                rewards.append(0.001)
                continue

            final_score, _, _, _ = execute_episode(env, task_id, actions, scenario_id)
            rewards.append(final_score)
        except Exception as e:
            print(f"[REWARD] Error computing reward for sample {i}: {e}", flush=True)
            rewards.append(0.001)

    return rewards


def compute_reward_single(
    completion: str,
    task_id: str,
    scenario_id: Optional[str] = None,
) -> float:
    """Compute reward for a single completion. Convenience wrapper."""
    return compute_reward(
        prompts=[""],
        completions=[completion],
        task_ids=[task_id],
        scenario_ids=[scenario_id] if scenario_id else None,
    )[0]


# ---------------------------------------------------------------------------
# Task sampler (weighted by inverse baseline score)
# ---------------------------------------------------------------------------

# Baseline scores from random agent (PRD §12.4)
BASELINE_SCORES = {
    "alert_triage": 0.15,
    "incident_investigation": 0.08,
    "threat_response": 0.04,
}

# Inverse-weighted sampling probabilities (harder tasks sampled more)
_inv_scores = {k: 1.0 / max(v, 0.01) for k, v in BASELINE_SCORES.items()}
_total_inv = sum(_inv_scores.values())
TASK_SAMPLE_WEIGHTS = {k: v / _total_inv for k, v in _inv_scores.items()}


def sample_task() -> Tuple[str, Optional[str]]:
    """
    Sample a (task_id, scenario_id) pair weighted by inverse baseline score.
    Returns (task_id, scenario_id or None).
    """
    import random
    tasks = list(TASK_SAMPLE_WEIGHTS.keys())
    weights = [TASK_SAMPLE_WEIGHTS[t] for t in tasks]
    task_id = random.choices(tasks, weights=weights, k=1)[0]

    # Pick a scenario for the task
    scenarios = TASK_CONFIG[task_id].get("scenarios", [None])
    scenario_id = random.choice(scenarios)

    return task_id, scenario_id


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Reward Wrapper Self-Test ===")

    # Test parsing
    test_text = """
    {"action_type": "query_logs", "parameters": {"log_source": "auth"}}
    {"action_type": "query_logs", "parameters": {"log_source": "firewall"}}
    {"action_type": "close_incident", "parameters": {"verdict": "true_positive", "attack_type": "brute_force", "attacker_ip": "185.220.101.47"}}
    """
    actions = parse_actions_from_text(test_text)
    print(f"Parsed {len(actions)} actions from test text")
    assert len(actions) == 3, f"Expected 3 actions, got {len(actions)}"

    # Test episode execution
    env = SocEnvironment()
    score, total_reward, steps, rewards = execute_episode(
        env, "incident_investigation", actions, "brute_force_ssh_001"
    )
    print(f"Episode: score={score:.4f}, total_reward={total_reward:.4f}, steps={steps}")
    print(f"Per-step rewards: {rewards}")

    # Test compute_reward
    rewards_list = compute_reward(
        prompts=["test"],
        completions=[test_text],
        task_ids=["incident_investigation"],
        scenario_ids=["brute_force_ssh_001"],
    )
    print(f"compute_reward result: {rewards_list}")

    # Test task sampling
    for _ in range(5):
        tid, sid = sample_task()
        print(f"Sampled: task={tid}, scenario={sid}")

    print("\n=== All self-tests passed ===")
