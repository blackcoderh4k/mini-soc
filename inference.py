"""
Mini SOC — Baseline Inference Script
=====================================
Runs the LLM agent against all 3 tasks and emits the required STDOUT format.

Required env vars:
  API_BASE_URL        LLM API endpoint (default: HF router)
  MODEL_NAME          Model identifier
  HF_TOKEN            API key
  LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image)
"""
import os
import json
import textwrap
import asyncio
import httpx
from typing import List, Optional, Dict, Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen2.5-0.5B-SOC-Expert")
BENCHMARK = "mini-soc"
MAX_TOKENS = 300
TEMPERATURE = 0.2  # low for deterministic SOC decisions

# Base URL for the running SOC environment
SOC_ENV_URL = os.getenv("SOC_ENV_URL", "http://localhost:8000")

TASKS = ["alert_triage", "incident_investigation", "threat_response"]
MAX_STEPS = {"alert_triage": 15, "incident_investigation": 20, "threat_response": 30}
SUCCESS_THRESHOLD = {"alert_triage": 0.6, "incident_investigation": 0.4, "threat_response": 0.3}


# ---------------------------------------------------------------------------
# STDOUT logging (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = httpx.post(f"{SOC_ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(
        f"{SOC_ENV_URL}/step",
        json={"action_type": action_type, "parameters": parameters},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "alert_triage": textwrap.dedent("""
        You are a Tier-1 SOC analyst. Your job is to classify security alerts.

        Available action:
          classify_alert: {"alert_id": str, "classification": "benign"|"suspicious"|"critical", "priority": "P1"|"P2"|"P3"|"P4"}

        Classification guide:
          - critical/P1: Active attack, confirmed compromise, immediate threat
          - suspicious/P2: Anomalous behaviour requiring investigation
          - suspicious/P3: Low-confidence anomaly
          - benign/P4: Known-good, scheduled, or authorized activity

        Respond with ONLY valid JSON for one action, e.g.:
        {"action_type": "classify_alert", "parameters": {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}}
    """).strip(),

    "incident_investigation": textwrap.dedent("""
        You are a Tier-1 SOC analyst investigating a security incident.

        Available actions:
          query_logs: {"log_source": "auth"|"firewall"|"dns"|"process"|"network", "filter_ip": str (optional)}
          classify_alert: {"alert_id": str, "classification": str, "priority": str}
          escalate: {"alert_id": str, "reason": str}
          close_incident: {"incident_id": str, "verdict": "true_positive"|"false_positive", "attack_type": str, "attacker_ip": str}

        Strategy:
          1. Query auth and firewall logs first
          2. Filter by suspicious IPs you find
          3. Correlate the timeline across log sources
          4. Close the incident with your verdict

        Respond with ONLY valid JSON for one action.
    """).strip(),

    "threat_response": textwrap.dedent("""
        You are a senior SOC analyst responding to an active threat.

        Available actions:
          query_logs: {"log_source": "auth"|"firewall"|"dns"|"process"|"network", "filter_ip": str (optional)}
          isolate_asset: {"hostname": str}
          block_ip: {"ip_address": str}
          write_report: {"report": {"summary": str, "attack_type": str, "affected_assets": [str], "attacker_ip": str, "timeline": str}}
          close_incident: {"incident_id": str, "verdict": str, "attack_type": str}

        CRITICAL: Only isolate assets you have evidence are compromised.
        Isolating a healthy critical asset (like DC-01) causes major damage.

        Strategy:
          1. Query process and network logs first
          2. Identify attacker IP and compromised hosts
          3. Isolate ONLY confirmed compromised assets
          4. Block attacker IPs
          5. Write a report then close the incident

        Respond with ONLY valid JSON for one action.
    """).strip(),
}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def build_user_prompt(task_id: str, step: int, obs: Dict[str, Any], last_reward: float, history: List[str]) -> str:
    ctx = obs.get("task_context", {}) or {}
    current_alert = obs.get("current_alert") or {}
    alert_queue = obs.get("alert_queue", [])
    logs = obs.get("available_logs", [])
    assets = obs.get("asset_inventory", [])
    open_incidents = obs.get("open_incidents", [])
    message = obs.get("message", "")

    classified_ids = [h.split("classify_alert:")[1] if "classify_alert:" in h else "" for h in history]

    queue_summary = []
    for a in alert_queue[:5]:
        if a.get("alert_id") not in classified_ids:
            queue_summary.append(f"  [{a.get('severity','?').upper()}] {a.get('alert_id')}: {a.get('description','')[:80]}")

    log_summary = []
    for l in logs[-8:]:
        log_summary.append(f"  [{l.get('log_source')}] {l.get('timestamp','')[:16]} {l.get('event_type')} src={l.get('source_ip','?')} user={l.get('user','?')}")

    incidents_summary = [f"  {i.get('incident_id')}: {i.get('status')}" for i in open_incidents]

    return textwrap.dedent(f"""
        Step {step} | Last reward: {last_reward:+.2f} | Task: {task_id}
        Objective: {ctx.get('objective', '')}
        Environment message: {message}

        CURRENT ALERT:
          ID: {current_alert.get('alert_id', 'none')} | {current_alert.get('alert_type', '')}
          Severity: {current_alert.get('severity', '')} | {current_alert.get('description', '')}
          Source IP: {current_alert.get('source_ip', 'N/A')}

        ALERT QUEUE (unclassified, showing first 5):
        {chr(10).join(queue_summary) if queue_summary else '  All alerts classified.'}

        RETRIEVED LOGS ({len(logs)} total, last 8):
        {chr(10).join(log_summary) if log_summary else '  No logs retrieved yet. Use query_logs.'}

        OPEN INCIDENTS:
        {chr(10).join(incidents_summary) if incidents_summary else '  None'}

        ACTIONS TAKEN: {len(history)}
        Previous: {history[-3:] if history else 'None'}

        Respond with exactly ONE JSON action.
    """).strip()


async def get_llm_decision(prompt: str) -> str:
    """Calls the local FastAPI /inference endpoint for decisions."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{SOC_ENV_URL}/inference",
                json={"prompt": prompt, "max_tokens": 128},
                timeout=60
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
    except Exception as e:
        print(f"  [ERROR] Local inference failed: {e}")
        return ""


def get_agent_action(task_id: str, step: int, obs: Dict[str, Any],
                      last_reward: float, history: List[str]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(task_id, step, obs, last_reward, history)
    try:
        raw = asyncio.run(get_llm_decision(SYSTEM_PROMPTS[task_id] + "\n" + user_prompt))
        print(f"  [DEBUG] Raw LLM Output: {raw}", flush=True)
        
        # Improved Cleaning: Extract JSON from between { and }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            json_str = raw[start:end+1]
            return json.loads(json_str)
            
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"  [DEBUG] LLM output was NOT valid JSON: {exc}", flush=True)
        # Return a 'nothing' action so we don't cheat with fallbacks
        return {"action_type": "request_info", "parameters": {}}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    """Run one full episode for a task. Returns final normalized score."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env_reset(task_id)
        obs = result.get("observation", {})
        last_reward = 0.0
        
        while steps_taken < MAX_STEPS[task_id]:
            steps_taken += 1
            # Get agent action
            action_data = get_agent_action(task_id, steps_taken, obs, last_reward, history)
            action_type = action_data.get("action_type", "request_info")
            parameters = action_data.get("parameters", {})

            # Execute in environment
            step_result = env_step(action_type, parameters)
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            error = step_result.get("info", {}).get("error")
            obs = step_result.get("observation", {})

            rewards.append(reward)
            last_reward = reward
            history.append(f"step={steps_taken} {action_type}:{str(parameters)}")

            log_step(step=steps_taken, action=f"{action_type}({str(parameters)})",
                     reward=reward, done=done, error=error)

            if done:
                score = step_result.get("info", {}).get("final_score", 0.0)
                break

        if not score and rewards:
            score = sum(r for r in rewards if r > 0) / max(len(rewards), 1)

        score = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_THRESHOLD[task_id]

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[INFO] Mini SOC Inference — Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Environment URL: {SOC_ENV_URL}", flush=True)

    all_scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Running task: {task_id}", flush=True)
        score = run_episode(task_id)
        all_scores[task_id] = score
        print(f"[SCORE] {task_id}: {score:.3f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY] Final scores per task:", flush=True)
    for task_id, score in all_scores.items():
        status = "PASS" if score >= SUCCESS_THRESHOLD[task_id] else "FAIL"
        print(f"  {task_id}: {score:.3f} [{status}]", flush=True)
    overall = sum(all_scores.values()) / len(all_scores)
    print(f"  Overall average: {overall:.3f}", flush=True)


if __name__ == "__main__":
    main()
