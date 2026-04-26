"""
Grader for Task 3: Active Threat Response (Hard)
Scores containment effectiveness, collateral damage, speed, and report quality.
Now dynamic: pulls ground truth from the environment state.
"""
from __future__ import annotations
from typing import Dict, Any, List, Set


CONTAINMENT_WEIGHT = 0.30
COLLATERAL_WEIGHT = 0.20   # negative: penalize unnecessary isolations
EVIDENCE_WEIGHT = 0.20
SPEED_WEIGHT = 0.10
REPORT_WEIGHT = 0.20


def grade(state: Dict[str, Any]) -> float:
    """
    Grade Task 3 episode.
    state must contain 'ground_truth' for dynamic evaluation.
    """
    gt = state.get("ground_truth", {})
    if not gt:
        return 0.001

    containment_score = _score_containment(state, gt)
    collateral_penalty = _score_collateral(state, gt)
    evidence_score = _score_evidence(state, gt)
    speed_score = _score_speed(state)
    report_score = _score_report(state, gt)

    total = (
        containment_score * CONTAINMENT_WEIGHT
        - collateral_penalty * COLLATERAL_WEIGHT
        + evidence_score * EVIDENCE_WEIGHT
        + speed_score * SPEED_WEIGHT
        + report_score * REPORT_WEIGHT
    )

    return round(min(max(total, 0.001), 0.999), 4)


def _score_containment(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    isolated: Set[str] = set(state.get("agent_isolated_assets", []))
    blocked: Set[str] = set(state.get("agent_blocked_ips", []))

    must_isolate = set(gt.get("assets_to_isolate", []))
    must_block = set(gt.get("ips_to_block", []))

    isolation_score = len(isolated & must_isolate) / len(must_isolate) if must_isolate else 1.0
    block_score = len(blocked & must_block) / len(must_block) if must_block else 1.0

    return (isolation_score * 0.6 + block_score * 0.4)


def _score_collateral(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    isolated: Set[str] = set(state.get("agent_isolated_assets", []))
    should_not_isolate = set(gt.get("assets_to_not_isolate", []))

    collateral = isolated & should_not_isolate
    if not collateral:
        return 0.0

    # Criticality map for penalties
    criticality_map = {
        "DC-01": 1.0,          # catastrophic
        "DB-FINANCE-01": 0.8,  # severe
        "WEB-SERVER-01": 0.5,  # significant
    }
    penalty = sum(criticality_map.get(h, 0.3) for h in collateral)
    return min(penalty, 1.0)


def _score_evidence(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    queried: Set[str] = set(state.get("agent_queried_sources", []))
    key_sources = set(gt.get("key_log_sources", []))
    if not key_sources:
        return 1.0
    return len(queried & key_sources) / len(key_sources)


def _score_speed(state: Dict[str, Any]) -> float:
    steps = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 30)
    if max_steps == 0:
        return 0.0
    ratio = steps / max_steps
    speed = max(1.0 - (ratio * 2), 0.0)
    return round(speed, 4)


def _score_report(state: Dict[str, Any], gt: Dict[str, Any]) -> float:
    report: Dict[str, Any] = state.get("agent_report", {})
    if not report:
        return 0.0

    required = set(gt.get("report_required_fields", {"summary"}))
    found = sum(1 for field in required if report.get(field))
    field_score = found / len(required)

    # Bonus: correct attack_type
    if report.get("attack_type", "").lower().replace(" ", "_") == gt.get("attack_type"):
        field_score = min(field_score + 0.15, 1.0)

    # Bonus: correct attacker IP
    if report.get("attacker_ip", "") == gt.get("attacker_ip"):
        field_score = min(field_score + 0.1, 1.0)

    return round(field_score, 4)


def compute_step_reward(action_type: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> float:
    """Dense per-step reward for Task 3."""
    gt = state.get("ground_truth", {})
    if not gt:
        return 0.0
        
    reward = 0.0

    if action_type == "query_logs":
        source = parameters.get("log_source", "")
        if source in gt.get("key_log_sources", []):
            reward += 0.10
        else:
            reward -= 0.02

    elif action_type == "isolate_asset":
        hostname = parameters.get("hostname", "")
        if hostname in gt.get("assets_to_isolate", []):
            reward += 0.25
        elif hostname in gt.get("assets_to_not_isolate", []):
            crit_penalty = {"DC-01": -0.4, "DB-FINANCE-01": -0.3, "WEB-SERVER-01": -0.15}
            reward += crit_penalty.get(hostname, -0.2)

    elif action_type == "block_ip":
        ip = parameters.get("ip_address", "")
        if ip in gt.get("ips_to_block", []):
            reward += 0.20
        else:
            reward -= 0.05

    elif action_type == "write_report":
        report = parameters.get("report", {})
        required = set(gt.get("report_required_fields", {"summary"}))
        filled = sum(1 for f in required if report.get(f))
        reward += (filled / len(required)) * 0.3

    elif action_type == "close_incident":
        isolated = set(state.get("agent_isolated_assets", []))
        if isolated & set(gt.get("assets_to_isolate", [])):
            reward += 0.15

    return round(reward, 4)
