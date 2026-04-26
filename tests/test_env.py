"""
Smoke tests for Mini SOC environment.
Run: python -m pytest tests/ -v
"""
import pytest
from server.mini_soc_environment import SocEnvironment
from models import Action, ActionType
from server.graders import grader1, grader2, grader3


# ---------------------------------------------------------------------------
# Test reset()
# ---------------------------------------------------------------------------

def test_reset_task1(env):
    result = env.reset("alert_triage")
    obs = result.observation
    assert obs.alert_queue, "Alert queue must not be empty"
    assert obs.current_alert is not None
    assert obs.task_context.task_id == "alert_triage"
    assert obs.task_context.difficulty == "easy"


def test_reset_task2(env):
    result = env.reset("incident_investigation")
    obs = result.observation
    assert obs.current_alert is not None
    assert len(obs.open_incidents) == 1


def test_reset_task3(env):
    result = env.reset("threat_response")
    obs = result.observation
    assert obs.current_alert is not None
    assert len(obs.alert_queue) >= 1


def test_reset_invalid_task(env):
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


def test_reset_clears_state(env):
    env.reset("alert_triage")
    # Do a step
    action = Action(action_type=ActionType.QUERY_LOGS, parameters={"log_source": "auth"})
    env.step(action)
    # Reset and verify clean state
    env.reset("alert_triage")
    state = env.state()
    assert state.step_count == 0
    assert state.total_reward == 0.0


# ---------------------------------------------------------------------------
# Test step()
# ---------------------------------------------------------------------------

def test_step_query_logs(env_task2):
    action = Action(action_type=ActionType.QUERY_LOGS, parameters={"log_source": "auth"})
    result = env_task2.step(action)
    assert result.observation is not None
    assert isinstance(result.reward, float)
    assert result.done is False
    assert len(result.observation.available_logs) > 0


def test_step_classify_alert(env_task1):
    action = Action(
        action_type=ActionType.CLASSIFY_ALERT,
        parameters={"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}
    )
    result = env_task1.step(action)
    assert result.reward > 0  # correct classification should give positive reward


def test_step_classify_alert_wrong(env_task1):
    action = Action(
        action_type=ActionType.CLASSIFY_ALERT,
        parameters={"alert_id": "ALT-001", "classification": "benign", "priority": "P4"}
    )
    result = env_task1.step(action)
    assert result.reward < 0  # missing critical as benign should penalize


def test_step_isolate_correct_asset(env_task3):
    # Query some logs first
    env_task3.step(Action(action_type=ActionType.QUERY_LOGS, parameters={"log_source": "process"}))
    # Isolate correct compromised asset
    action = Action(action_type=ActionType.ISOLATE_ASSET, parameters={"hostname": "WS-HR-03"})
    result = env_task3.step(action)
    assert result.reward > 0


def test_step_isolate_wrong_asset(env_task3):
    action = Action(action_type=ActionType.ISOLATE_ASSET, parameters={"hostname": "DC-01"})
    result = env_task3.step(action)
    assert result.reward < 0  # collateral damage


def test_step_after_done(env):
    env.reset("alert_triage")
    env._done = True
    action = Action(action_type=ActionType.REQUEST_INFO, parameters={})
    result = env.step(action)
    assert result.done is True
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# Test state()
# ---------------------------------------------------------------------------

def test_state_has_ground_truth(env):
    env.reset("incident_investigation")
    state = env.state()
    assert "verdict" in state.ground_truth
    assert state.ground_truth["verdict"] == "true_positive"


# ---------------------------------------------------------------------------
# Test graders directly
# ---------------------------------------------------------------------------

def test_grader1_perfect_score():
    state = {
        "agent_classifications": {
            "ALT-001": {"classification": "critical", "priority": "P1"},
            "ALT-002": {"classification": "critical", "priority": "P1"},
            "ALT-010": {"classification": "suspicious", "priority": "P2"},
            "ALT-011": {"classification": "suspicious", "priority": "P2"},
            "ALT-012": {"classification": "critical", "priority": "P1"},
            "ALT-020": {"classification": "benign", "priority": "P4"},
            "ALT-030": {"classification": "benign", "priority": "P4"},
            "ALT-031": {"classification": "benign", "priority": "P3"},
            "ALT-032": {"classification": "benign", "priority": "P4"},
            "ALT-033": {"classification": "suspicious", "priority": "P2"},
        }
    }
    score = grader1.grade(state)
    assert score == 0.999


def test_grader1_empty_state():
    score = grader1.grade({})
    assert score == 0.001


def test_grader1_partial_score():
    state = {
        "agent_classifications": {
            "ALT-001": {"classification": "critical", "priority": "P1"},
            "ALT-002": {"classification": "critical", "priority": "P1"},
        }
    }
    score = grader1.grade(state)
    assert 0.0 < score < 1.0


def test_grader2_perfect_score():
    state = {
        "agent_verdict": "true_positive",
        "agent_attack_type": "brute_force",
        "agent_attacker_ip": "185.220.101.47",
        "agent_queried_log_ids": ["AUTH-001", "AUTH-002", "FW-001"],
        "agent_queried_sources": ["auth", "firewall"],
    }
    score = grader2.grade(state)
    assert score > 0.9


def test_grader2_wrong_verdict():
    state = {
        "agent_verdict": "false_positive",
        "agent_attack_type": "brute_force",
        "agent_attacker_ip": "185.220.101.47",
        "agent_queried_log_ids": ["AUTH-001", "AUTH-002", "FW-001"],
        "agent_queried_sources": ["auth", "firewall"],
    }
    score = grader2.grade(state)
    assert score < 0.7  # verdict wrong = big penalty


def test_grader3_collateral_damage():
    state = {
        "agent_isolated_assets": ["DC-01"],  # WRONG: healthy critical asset
        "agent_blocked_ips": ["94.102.49.190"],
        "agent_queried_sources": ["process", "network"],
        "agent_report": {},
        "steps_taken": 10,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    assert score < 0.5  # collateral damage tank score


def test_grader3_scores_in_range():
    state = {
        "agent_isolated_assets": [],
        "agent_blocked_ips": [],
        "agent_queried_sources": [],
        "agent_report": {},
        "steps_taken": 30,
        "max_steps": 30,
    }
    score = grader3.grade(state)
    assert 0.0 <= score <= 1.0
