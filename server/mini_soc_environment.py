"""
Mini SOC Environment — Core Episode Logic
Implements the OpenEnv state machine: reset(), step(), state().
Now dynamic: supports multiple scenarios per task and adaptive difficulty.
"""
from __future__ import annotations
import uuid
import copy
from typing import Any, Dict, List, Optional, Tuple

# Dual-import pattern: supports both package-mode and Docker-mode
try:
    from models import (
        Action, ActionType, Alert, Asset, Incident,
        Observation, ResetResult, Reward, StateResult,
        StepResult, TaskContext, LogEntry,
    )
except ImportError:
    from ..models import (
        Action, ActionType, Alert, Asset, Incident,
        Observation, ResetResult, Reward, StateResult,
        StepResult, TaskContext, LogEntry,
    )

try:
    from server.simulator.attack_seeds import (
        ATTACK_SCENARIOS, TASK1_ALERT_QUEUE, ASSET_INVENTORY,
    )
except ImportError:
    from .simulator.attack_seeds import (
        ATTACK_SCENARIOS, TASK1_ALERT_QUEUE, ASSET_INVENTORY,
    )

try:
    from server.simulator.log_gen import (
        get_logs_for_source, build_asset_inventory,
        sanitize_for_agent, get_benign_log_noise,
    )
except ImportError:
    from .simulator.log_gen import (
        get_logs_for_source, build_asset_inventory,
        sanitize_for_agent, get_benign_log_noise,
    )

try:
    from server.graders import grader1, grader2, grader3
except ImportError:
    from .graders import grader1, grader2, grader3


TASK_CONFIG = {
    "alert_triage": {
        "max_steps": 15,
        "scenarios": [None], # Mixed alerts
        "grader": grader1,
        "objective": (
            "Classify all alerts in the queue as benign/suspicious/critical "
            "and assign correct priority (P1–P4) to each."
        ),
    },
    "incident_investigation": {
        "max_steps": 20,
        "scenarios": ["brute_force_ssh_001", "ransomware_001", "insider_threat_001", "bec_fraud_001"],
        "grader": grader2,
        "objective": (
            "Investigate the active incident: query relevant log sources, "
            "identify the attacker IP, and submit a verdict with attack type."
        ),
    },
    "threat_response": {
        "max_steps": 30,
        "scenarios": ["phishing_lateral_001", "supply_chain_001", "multi_stage_apt_001", "sql_injection_001"],
        "grader": grader3,
        "objective": (
            "A multi-stage attack is active. Gather evidence, isolate compromised assets, "
            "block attacker IPs, and write a full incident report."
        ),
    },
}


class SocEnvironment:
    """
    Mini SOC OpenEnv Environment.
    Call reset(task_id) to start an episode, then step(action) repeatedly.
    """

    def __init__(self):
        self._episode_id: str = ""
        self._task_id: str = ""
        self._scenario_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0

        # Episode state
        self._alert_queue: List[Alert] = []
        self._current_alert: Optional[Alert] = None
        self._available_logs: List[LogEntry] = []
        self._asset_inventory: List[Asset] = []
        self._open_incidents: List[Incident] = []
        self._actions_taken: List[str] = []
        self._task_context: Optional[TaskContext] = None

        # Grader tracking state
        self._agent_classifications: Dict[str, Dict[str, str]] = {}
        self._agent_queried_log_ids: List[str] = []
        self._agent_queried_sources: List[str] = []
        self._agent_isolated_assets: List[str] = []
        self._agent_blocked_ips: List[str] = []
        self._agent_verdict: str = ""
        self._agent_attack_type: str = ""
        self._agent_attacker_ip: str = ""
        self._agent_report: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "alert_triage", scenario_id: Optional[str] = None) -> ResetResult:
        """Start a fresh episode for the given task."""
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIG.keys())}")

        config = TASK_CONFIG[task_id]
        
        # Scenario selection
        if scenario_id:
            if scenario_id not in ATTACK_SCENARIOS and scenario_id is not None:
                raise ValueError(f"Unknown scenario_id '{scenario_id}'")
            self._scenario_id = scenario_id
        else:
            # Default or random scenario from task list
            self._scenario_id = config["scenarios"][0]

        self._episode_id = str(uuid.uuid4())[:8]
        self._task_id = task_id
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._actions_taken = []

        # Reset grader state
        self._agent_classifications = {}
        self._agent_queried_log_ids = []
        self._agent_queried_sources = []
        self._agent_isolated_assets = []
        self._agent_blocked_ips = []
        self._agent_verdict = ""
        self._agent_attack_type = ""
        self._agent_attacker_ip = ""
        self._agent_report = {}

        # Build episode state
        self._asset_inventory = build_asset_inventory(self._scenario_id)
        self._available_logs = []
        self._open_incidents = []

        if task_id == "alert_triage":
            self._alert_queue = [Alert(**{k: v for k, v in a.items() if k not in ["ground_truth_classification", "ground_truth_priority"]})
                                  for a in TASK1_ALERT_QUEUE]
            self._current_alert = self._alert_queue[0] if self._alert_queue else None

        elif task_id in ["incident_investigation", "threat_response"]:
            scenario = ATTACK_SCENARIOS[self._scenario_id]
            if task_id == "incident_investigation":
                self._alert_queue = [Alert(**a) for a in scenario["alerts"]]
                self._current_alert = self._alert_queue[0]
            else: # threat_response
                # Start with only first alert visible
                self._alert_queue = [Alert(**scenario["alerts"][0])]
                self._current_alert = self._alert_queue[0]
            
            incident = Incident(
                incident_id=f"INC-{self._episode_id}",
                alert_ids=[a.alert_id for a in self._alert_queue],
                status="open",
            )
            self._open_incidents = [incident]

        self._task_context = TaskContext(
            task_id=task_id,
            task_name=task_id.replace("_", " ").title(),
            difficulty={"alert_triage": "easy", "incident_investigation": "medium", "threat_response": "hard"}[task_id],
            objective=config["objective"],
            max_steps=config["max_steps"],
        )

        obs = self._build_observation(message=f"Episode started. Task: {task_id}. Scenario: {self._scenario_id}. {config['objective']}")
        return ResetResult(observation=obs, info={"episode_id": self._episode_id, "task_id": task_id, "scenario_id": self._scenario_id})

    def step(self, action: Action) -> StepResult:
        """Execute one agent action and return next observation + reward."""
        if self._done:
            obs = self._build_observation(message="Episode already done. Call reset().")
            return StepResult(observation=obs, reward=0.0, done=True, info={"error": "episode_done"})

        self._step_count += 1
        config = TASK_CONFIG[self._task_id]
        self._task_context.current_step = self._step_count

        # Reveal new alerts progressively if scenario supports it
        if self._task_id == "threat_response":
            self._surface_new_alerts()

        # Process action
        reward, message, error = self._process_action(action)
        self._total_reward += reward
        self._actions_taken.append(f"step={self._step_count} {action.action_type.value}")

        # Check terminal conditions
        done = self._check_done(config["max_steps"])
        self._done = done

        obs = self._build_observation(message=message)
        info = {
            "step": self._step_count,
            "total_reward": round(self._total_reward, 4),
            "error": error,
        }
        if done:
            info["final_score"] = self._compute_final_score()

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> StateResult:
        """Full internal state including ground truth. Used by graders."""
        return StateResult(
            observation=self._build_observation(),
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._step_count,
            total_reward=self._total_reward,
            done=self._done,
            ground_truth=self._build_ground_truth(),
        )

    # -----------------------------------------------------------------------
    # Action processing
    # -----------------------------------------------------------------------

    def _process_action(self, action: Action) -> Tuple[float, str, Optional[str]]:
        """Returns (reward, message, error_string)."""
        at = action.action_type
        params = action.parameters
        reward = 0.0
        error = None

        # Penalize repeated identical actions (thrashing)
        type_count = sum(1 for a in self._actions_taken if at.value in a)
        if type_count > 5 and at.value != "classify_alert":
            reward -= 0.05
            return reward, "Warning: repeated action type detected.", "thrashing_penalty"

        if at == ActionType.QUERY_LOGS:
            reward, message = self._handle_query_logs(params)

        elif at == ActionType.CLASSIFY_ALERT:
            reward, message = self._handle_classify_alert(params)

        elif at == ActionType.ISOLATE_ASSET:
            reward, message = self._handle_isolate_asset(params)

        elif at == ActionType.BLOCK_IP:
            reward, message = self._handle_block_ip(params)

        elif at == ActionType.ESCALATE:
            reward, message = self._handle_escalate(params)

        elif at == ActionType.WRITE_REPORT:
            reward, message = self._handle_write_report(params)

        elif at == ActionType.CLOSE_INCIDENT:
            reward, message = self._handle_close_incident(params)

        elif at == ActionType.REQUEST_INFO:
            reward = 0.0
            message = "Info request noted. Consult available logs and asset inventory."

        else:
            reward = -0.05
            message = f"Unknown action type: {at}"
            error = "unknown_action"

        return round(reward, 4), message, error

    def _handle_query_logs(self, params: Dict) -> Tuple[float, str]:
        source = params.get("log_source", "")
        filter_ip = params.get("filter_ip")
        filter_user = params.get("filter_user")
        
        if not self._scenario_id:
            return -0.02, "No scenario logs available for this task."

        logs = get_logs_for_source(self._scenario_id, source, filter_ip, filter_user)
        noise = get_benign_log_noise(source, count=2)
        all_logs = logs + noise

        # Track for grader
        for log in logs:
            if log.log_id not in self._agent_queried_log_ids:
                self._agent_queried_log_ids.append(log.log_id)
        if source not in self._agent_queried_sources:
            self._agent_queried_sources.append(source)

        # Merge into available_logs (deduplicate)
        existing_ids = {l.log_id for l in self._available_logs}
        for log in all_logs:
            if log.log_id not in existing_ids:
                self._available_logs.append(log)
                existing_ids.add(log.log_id)

        # Compute reward based on task
        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward(
            "query_logs", params, self._build_grader_state()
        )
        return reward, f"Retrieved {len(logs)} logs from {source} source."

    def _handle_classify_alert(self, params: Dict) -> Tuple[float, str]:
        alert_id = params.get("alert_id", "")
        classification = params.get("classification", "")
        priority = params.get("priority", "P3")

        if not alert_id or not classification:
            return -0.05, "classify_alert requires alert_id and classification."

        self._agent_classifications[alert_id] = {
            "classification": classification,
            "priority": priority,
        }

        if self._task_id == "alert_triage":
            reward = grader1.compute_step_reward(alert_id, classification, priority)
            # Advance to next unclassified alert
            classified_ids = set(self._agent_classifications.keys())
            for alert in self._alert_queue:
                if alert.alert_id not in classified_ids:
                    self._current_alert = alert
                    break
            return reward, f"Alert {alert_id} classified."

        else:
            grader = TASK_CONFIG[self._task_id]["grader"]
            reward = grader.compute_step_reward("classify_alert", params, self._build_grader_state())
            return reward, f"Alert {alert_id} classified."

    def _handle_isolate_asset(self, params: Dict) -> Tuple[float, str]:
        hostname = params.get("hostname", "")
        if not hostname:
            return -0.05, "isolate_asset requires hostname."

        for asset in self._asset_inventory:
            if asset.hostname == hostname:
                asset.is_isolated = True
                break

        if hostname not in self._agent_isolated_assets:
            self._agent_isolated_assets.append(hostname)

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("isolate_asset", params, self._build_grader_state())
        return reward, f"Asset {hostname} isolated."

    def _handle_block_ip(self, params: Dict) -> Tuple[float, str]:
        ip = params.get("ip_address", "")
        if not ip:
            return -0.05, "block_ip requires ip_address."

        if ip not in self._agent_blocked_ips:
            self._agent_blocked_ips.append(ip)

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("block_ip", params, self._build_grader_state())
        return reward, f"IP {ip} blocked."

    def _handle_escalate(self, params: Dict) -> Tuple[float, str]:
        reason = params.get("reason", "no reason provided")
        reward = 0.05 if self._task_id != "alert_triage" else -0.02
        return reward, f"Alert escalated. Reason: {reason}"

    def _handle_write_report(self, params: Dict) -> Tuple[float, str]:
        report = params.get("report", params)
        self._agent_report = report
        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("write_report", {"report": report}, self._build_grader_state())
        return reward, "Incident report submitted."

    def _handle_close_incident(self, params: Dict) -> Tuple[float, str]:
        verdict = params.get("verdict", "")
        attack_type = params.get("attack_type", "")
        self._agent_verdict = verdict
        self._agent_attack_type = attack_type
        self._agent_attacker_ip = params.get("attacker_ip", "")

        for inc in self._open_incidents:
            inc.status = "resolved"
            inc.verdict = verdict
            inc.attack_type = attack_type

        grader = TASK_CONFIG[self._task_id]["grader"]
        reward = grader.compute_step_reward("close_incident", params, self._build_grader_state())
        self._done = True
        return reward, f"Incident closed. Verdict: {verdict}."

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _surface_new_alerts(self):
        """Reveal additional alerts progressively in threat_response task."""
        if not self._scenario_id: return
        scenario = ATTACK_SCENARIOS[self._scenario_id]
        all_alerts = scenario["alerts"]
        if len(all_alerts) <= 1: return
        
        reveal_at_steps = {3: 1, 6: 2, 10: 3}
        idx = reveal_at_steps.get(self._step_count)
        if idx is not None and idx < len(all_alerts):
            existing_ids = {a.alert_id for a in self._alert_queue}
            alert = Alert(**all_alerts[idx])
            if alert.alert_id not in existing_ids:
                self._alert_queue.append(alert)
                if self._open_incidents:
                    self._open_incidents[0].alert_ids.append(alert.alert_id)

    def _check_done(self, max_steps: int) -> bool:
        if self._done: return True
        if self._step_count >= max_steps: return True
        if self._task_id == "alert_triage":
            classified = set(self._agent_classifications.keys())
            all_ids = {a.alert_id for a in self._alert_queue}
            if all_ids and all_ids.issubset(classified):
                return True
        return False

    def _compute_final_score(self) -> float:
        grader = TASK_CONFIG[self._task_id]["grader"]
        return grader.grade(self._build_grader_state())

    def _build_grader_state(self) -> Dict[str, Any]:
        return {
            "agent_classifications": self._agent_classifications,
            "agent_queried_log_ids": self._agent_queried_log_ids,
            "agent_queried_sources": self._agent_queried_sources,
            "agent_isolated_assets": self._agent_isolated_assets,
            "agent_blocked_ips": self._agent_blocked_ips,
            "agent_verdict": self._agent_verdict,
            "agent_attack_type": self._agent_attack_type,
            "agent_attacker_ip": self._agent_attacker_ip,
            "agent_report": self._agent_report,
            "steps_taken": self._step_count,
            "max_steps": TASK_CONFIG[self._task_id]["max_steps"],
            "ground_truth": self._build_ground_truth(),
        }

    def _build_ground_truth(self) -> Dict[str, Any]:
        if self._scenario_id in ATTACK_SCENARIOS:
            return ATTACK_SCENARIOS[self._scenario_id].get("ground_truth", {})
        return {}

    def _build_observation(self, message: str = "") -> Observation:
        safe_assets = sanitize_for_agent(self._asset_inventory)
        safe_logs = [
            LogEntry(
                log_id=l.log_id, log_source=l.log_source, timestamp=l.timestamp,
                source_ip=l.source_ip, dest_ip=l.dest_ip, user=l.user,
                event_type=l.event_type, details=l.details, is_malicious=False
            )
            for l in self._available_logs
        ]
        ctx = self._task_context
        if ctx:
            ctx = ctx.model_copy(update={"current_step": self._step_count})
        return Observation(
            current_alert=self._current_alert,
            alert_queue=self._alert_queue,
            available_logs=safe_logs,
            asset_inventory=safe_assets,
            open_incidents=self._open_incidents,
            actions_taken=self._actions_taken,
            time_elapsed=self._step_count * 5,
            task_context=ctx,
            message=message,
        )
