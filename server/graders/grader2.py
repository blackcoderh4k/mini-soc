"""
Grader for Task 2: Incident Investigation (Medium)
Scores based on evidence gathering quality, timeline correlation, and verdict accuracy.
Now dynamic: pulls ground truth from the environment state.
"""
from __future__ import annotations
from typing import Dict, Any, List, Set


# Scoring weights
VERDICT_WEIGHT = 0.35
ATTACK_TYPE_WEIGHT = 0.20
EVIDENCE_WEIGHT = 0.30
ATTACKER_ID_WEIGHT = 0.15


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 2 episode.
    state must contain 'ground_truth' for dynamic evaluation.
    """
    gt = state.get("ground_truth", {})
    if not gt:
        return 0.001

    verdict_score = _score_verdict(state, gt)
    attack_type_score = _score_attack_type(state, gt)
    evidence_score = _score_evidence(state, gt)
    attacker_score = _score_attacker_id(state, gt)

    total = (
        verdict_score * VERDICT_WEIGHT
        + attack_type_score * ATTACK_TYPE_WEIGHT
        + evidence_score * EVIDENCE_WEIGHT
        + attacker_score * ATTACKER_ID_WEIGHT
    )

    return round(min(max(total, 0.001), 0.999), 4)


def _score_verdict(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    agent = state.get("agent_verdict", "").lower()
    truth = gt.get("verdict", "true_positive")
    return 1.0 if agent == truth else 0.0


def _score_attack_type(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    agent = state.get("agent_attack_type", "").lower().replace(" ", "_")
    truth = gt.get("attack_type", "")
    return 1.0 if agent == truth else 0.0


def _score_evidence(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Partial credit for each key piece of evidence retrieved."""
    queried_ids: Set[str] = set(state.get("agent_queried_log_ids", []))
    queried_sources: Set[str] = set(state.get("agent_queried_sources", []))

    key_ids = set(gt.get("key_evidence_log_ids", []))
    key_sources = set(gt.get("key_log_sources", []))

    if not key_ids and not key_sources:
        return 1.0

    id_score = len(queried_ids & key_ids) / len(key_ids) if key_ids else 0.0
    source_score = len(queried_sources & key_sources) / len(key_sources) if key_sources else 0.0

    # Penalize for querying too many irrelevant sources (thrashing)
    irrelevant = len(queried_sources - key_sources)
    noise_penalty = min(irrelevant * 0.05, 0.2)

    return max((id_score * 0.6 + source_score * 0.4) - noise_penalty, 0.0)


def _score_attacker_id(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    agent_ip = state.get("agent_attacker_ip", "").strip()
    truth_ip = gt.get("attacker_ip")
    if truth_ip is None: # Insider threat case
        return 1.0 if not agent_ip or agent_ip in gt.get("attacker_ips", []) else 0.0
    return 1.0 if agent_ip == truth_ip else 0.0


def compute_step_reward(action_type: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> float:
    """Dense per-step reward for Task 2."""
    gt = state.get("ground_truth", {})
    if not gt:
        return 0.0

    reward = 0.0

    if action_type == "query_logs":
        source = parameters.get("log_source", "")
        if source in gt.get("key_log_sources", []):
            reward += 0.15
        else:
            reward -= 0.03

        # Bonus if key evidence log ID found in this step
        # Note: state['agent_queried_log_ids'] already includes current step logs
        queried_ids = set(state.get("agent_queried_log_ids", []))
        key_ids = set(gt.get("key_evidence_log_ids", []))
        # This is a bit simplified; ideally we'd track 'newly' found
        reward += len(queried_ids & key_ids) * 0.02 # smaller reward per hit

    elif action_type == "classify_alert":
        classification = parameters.get("classification", "")
        if classification == gt.get("classification", "critical"):
            reward += 0.1

    elif action_type == "close_incident":
        verdict = parameters.get("verdict", "")
        attack_type = parameters.get("attack_type", "").replace(" ", "_")
        if verdict == gt.get("verdict"):
            reward += 0.3
        if attack_type == gt.get("attack_type"):
            reward += 0.2

    elif action_type == "escalate":
        reward += 0.05

    return round(reward, 4)
