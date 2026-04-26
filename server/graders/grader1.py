"""
Grader for Task 1: Alert Triage (Easy)
Scores 0.0–1.0 based on classification accuracy and priority assignment.
Now dynamic: pulls ground truth from the alert queue in the environment.
"""
from __future__ import annotations
from typing import Dict, Any, List

try:
    from server.simulator.attack_seeds import TASK1_ALERT_QUEUE
except ImportError:
    from ..simulator.attack_seeds import TASK1_ALERT_QUEUE

# Build a lookup map for ground truth
# This still acts as a default but we'll try to be more dynamic in compute_step_reward if possible
_GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    a["alert_id"]: {
        "classification": a["ground_truth_classification"],
        "priority": a["ground_truth_priority"]
    }
    for a in TASK1_ALERT_QUEUE
}

TOTAL_ALERTS = len(_GROUND_TRUTH)
CLASSIFICATION_WEIGHT = 0.7
PRIORITY_WEIGHT = 0.3


def grade(state: Dict[str, Any]) -> float:
    """Grade Task 1 episode using dynamic ground truth lookup."""
    agent_classifications: Dict[str, Dict[str, str]] = state.get("agent_classifications", {})
    if not agent_classifications:
        return 0.001

    classification_correct = 0
    priority_correct = 0
    total_attempted = 0

    for alert_id, truth in _GROUND_TRUTH.items():
        agent = agent_classifications.get(alert_id)
        if agent is None:
            continue
        total_attempted += 1

        if agent.get("classification", "").lower() == truth["classification"]:
            classification_correct += 1

        agent_prio = agent.get("priority", "")
        truth_prio = truth["priority"]
        if agent_prio == truth_prio:
            priority_correct += 1
        elif _priority_distance(agent_prio, truth_prio) == 1:
            priority_correct += 0.5

    if total_attempted == 0:
        return 0.001

    coverage = total_attempted / TOTAL_ALERTS
    classification_score = (classification_correct / TOTAL_ALERTS) * CLASSIFICATION_WEIGHT
    priority_score = (priority_correct / TOTAL_ALERTS) * PRIORITY_WEIGHT
    raw_score = (classification_score + priority_score) * coverage

    return round(min(max(raw_score, 0.001), 0.999), 4)


def _priority_distance(p1: str, p2: str) -> int:
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    v1 = order.get(p1, -1)
    v2 = order.get(p2, -1)
    if v1 == -1 or v2 == -1:
        return 99
    return abs(v1 - v2)


def compute_step_reward(alert_id: str, classification: str, priority: str) -> float:
    """Per-step reward using dynamic lookup."""
    truth = _GROUND_TRUTH.get(alert_id)
    if not truth:
        return -0.05

    reward = 0.0
    if classification.lower() == truth["classification"]:
        reward += 0.2
    else:
        if truth["classification"] == "critical" and classification == "benign":
            reward -= 0.3
        elif truth["classification"] == "benign" and classification == "critical":
            reward -= 0.1
        else:
            reward -= 0.05

    dist = _priority_distance(priority, truth["priority"])
    if dist == 0:
        reward += 0.1
    elif dist == 1:
        reward += 0.05

    return round(reward, 4)
