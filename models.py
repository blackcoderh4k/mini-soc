"""
Pydantic models for Mini SOC OpenEnv environment.
Defines typed Observation, Action, Reward, and supporting data models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ---------------------------------------------------------------------------
# Supporting data models
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertClassification(str, Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    CRITICAL = "critical"


class AlertPriority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class AttackType(str, Enum):
    BRUTE_FORCE = "brute_force"
    PHISHING = "phishing"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE = "malware"
    RECONNAISSANCE = "reconnaissance"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    FALSE_POSITIVE = "false_positive"


class Alert(BaseModel):
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    timestamp: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    dest_port: Optional[int] = None
    description: str
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    classification: Optional[AlertClassification] = None
    priority: Optional[AlertPriority] = None


class LogEntry(BaseModel):
    log_id: str
    log_source: str  # firewall, auth, dns, process, network
    timestamp: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    user: Optional[str] = None
    event_type: str
    details: Dict[str, Any] = Field(default_factory=dict)
    is_malicious: bool = False  # ground truth, hidden from agent


class Asset(BaseModel):
    hostname: str
    ip_address: str
    asset_type: str  # workstation, server, domain_controller, database
    criticality: int  # 1 (low) to 5 (critical)
    owner: str
    department: str
    is_compromised: bool = False  # ground truth, hidden from agent
    is_isolated: bool = False


class Incident(BaseModel):
    incident_id: str
    alert_ids: List[str]
    status: str  # open, investigating, resolved, escalated
    assigned_to: str = "agent"
    verdict: Optional[str] = None
    attack_type: Optional[AttackType] = None
    notes: List[str] = Field(default_factory=list)


class TaskContext(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    objective: str
    max_steps: int
    current_step: int = 0
    alerts_classified: int = 0
    correct_classifications: int = 0
    logs_queried: List[str] = Field(default_factory=list)
    key_evidence_found: List[str] = Field(default_factory=list)
    assets_isolated: List[str] = Field(default_factory=list)
    ips_blocked: List[str] = Field(default_factory=list)
    report_written: bool = False
    incident_closed: bool = False


# ---------------------------------------------------------------------------
# Core OpenEnv models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    Represents everything the SOC analyst agent can see.
    """
    current_alert: Optional[Alert] = None
    alert_queue: List[Alert] = Field(default_factory=list)
    available_logs: List[LogEntry] = Field(default_factory=list)
    asset_inventory: List[Asset] = Field(default_factory=list)
    open_incidents: List[Incident] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    time_elapsed: int = 0  # simulated minutes
    task_context: Optional[TaskContext] = None
    message: str = ""  # human-readable status message


class ActionType(str, Enum):
    QUERY_LOGS = "query_logs"
    CLASSIFY_ALERT = "classify_alert"
    ESCALATE = "escalate"
    ISOLATE_ASSET = "isolate_asset"
    BLOCK_IP = "block_ip"
    CLOSE_INCIDENT = "close_incident"
    WRITE_REPORT = "write_report"
    REQUEST_INFO = "request_info"


class Action(BaseModel):
    """
    Action submitted by the agent via step().
    action_type determines which parameters are required.

    Examples:
      query_logs:     {"log_source": "firewall", "filter_ip": "10.0.0.5"}
      classify_alert: {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}
      isolate_asset:  {"hostname": "WS-FINANCE-01"}
      block_ip:       {"ip_address": "185.220.101.5"}
      write_report:   {"summary": "...", "attack_type": "lateral_movement", "affected_assets": [...]}
      close_incident: {"incident_id": "INC-001", "verdict": "true_positive"}
      escalate:       {"alert_id": "ALT-001", "reason": "..."}
      request_info:   {"question": "..."}
    """
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """
    Shaped reward with breakdown for interpretability.
    """
    total: float = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class StepResult(BaseModel):
    """
    Return value of env.step(action).
    """
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """
    Return value of env.reset().
    """
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """
    Return value of env.state() — full internal state including ground truth.
    Used for debugging and grader evaluation.
    """
    observation: Observation
    episode_id: str
    task_id: str
    step_count: int
    total_reward: float
    done: bool
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
