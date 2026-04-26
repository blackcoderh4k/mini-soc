---
title: Mini SOC
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
license: mit
tags:
  - openenv
  - cybersecurity
  - reinforcement-learning
---

# Mini SOC — OpenEnv Environment

An RL environment where an AI agent acts as a **Tier-1 Security Operations Center (SOC) analyst**: triaging alerts, investigating incidents, and responding to active threats.

Built for the [OpenEnv](https://huggingface.co/openenv) framework.

---

## Why This Environment?

SOC analyst shortage is a **$10B+ industry problem**. Security teams are overwhelmed with alert volume — analysts spend 40% of their time on false positives. This environment trains and evaluates AI agents on the exact multi-step decision-making SOC analysts perform daily:

- Reading and classifying security alerts
- Querying log sources to gather forensic evidence
- Correlating events across a timeline to identify attack patterns
- Containing threats by isolating assets and blocking attacker IPs
- Writing structured incident reports

---

## Tasks

| Task | Difficulty | Max Steps | Success Threshold |
|---|---|---|---|
| `alert_triage` | Easy | 15 | 0.60 |
| `incident_investigation` | Medium | 20 | 0.50 |
| `threat_response` | Hard | 30 | 0.40 |

### Task 1 — Alert Triage (Easy)
The agent receives a queue of 10 security alerts (port scans, brute-force attempts, suspicious logins, false positives) and must classify each as `benign`, `suspicious`, or `critical`, and assign a priority (`P1`–`P4`). Grader measures classification accuracy and priority correctness with partial credit.

**Expected baseline score: ~0.65**

### Task 2 — Incident Investigation (Medium)
A single suspicious alert is open. The agent must query log sources (`auth`, `firewall`, `dns`, `process`, `network`), correlate events across a timeline, identify the attacker IP, and close the incident with a verdict (`true_positive`/`false_positive`) and attack type. Partial credit for querying relevant sources even with wrong verdict.

**Expected baseline score: ~0.42**

### Task 3 — Active Threat Response (Hard)
A multi-stage attack (phishing → credential theft → lateral movement) is unfolding in real time. New alerts surface as the episode progresses. The agent must gather evidence, isolate **only** compromised assets (isolating healthy critical assets like `DC-01` triggers severe penalty), block attacker IPs, write a structured incident report, and close the incident. Tests planning, restraint, and report quality simultaneously.

**Expected baseline score: ~0.22**

---

## Observation Space

```python
class Observation(BaseModel):
    current_alert: Optional[Alert]        # Active alert under investigation
    alert_queue: List[Alert]              # Pending alerts
    available_logs: List[LogEntry]        # Logs retrieved so far this episode
    asset_inventory: List[Asset]          # Network assets (criticality 1–5)
    open_incidents: List[Incident]        # Active incident cases
    actions_taken: List[str]              # Episode action history
    time_elapsed: int                     # Simulated minutes elapsed
    task_context: Optional[TaskContext]   # Task objective and progress
    message: str                          # Human-readable status
```

### Alert fields
```
alert_id, alert_type, severity (low/medium/high/critical),
timestamp, source_ip, dest_ip, dest_port, description, raw_data
```

### LogEntry fields
```
log_id, log_source (auth/firewall/dns/process/network),
timestamp, source_ip, dest_ip, user, event_type, details
```

### Asset fields
```
hostname, ip_address, asset_type, criticality (1–5),
owner, department, is_isolated
```

---

## Action Space

```python
class Action(BaseModel):
    action_type: Literal[
        "query_logs",      # Retrieve logs from a source
        "classify_alert",  # Label alert severity + priority
        "escalate",        # Escalate to Tier-2
        "isolate_asset",   # Remove asset from network
        "block_ip",        # Block IP at perimeter firewall
        "close_incident",  # Submit final verdict
        "write_report",    # Write structured incident report
        "request_info",    # Ask for clarification (no reward)
    ]
    parameters: Dict[str, Any]
```

### Action parameter schemas

```json
// query_logs
{"log_source": "auth", "filter_ip": "10.0.0.5"}

// classify_alert
{"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}

// isolate_asset
{"hostname": "WS-HR-03"}

// block_ip
{"ip_address": "185.220.101.47"}

// close_incident
{"incident_id": "INC-abc", "verdict": "true_positive", "attack_type": "brute_force", "attacker_ip": "185.220.101.47"}

// write_report
{"report": {"summary": "...", "attack_type": "lateral_movement", "affected_assets": ["WS-HR-03"], "attacker_ip": "94.102.49.190", "timeline": "..."}}
```

---

## Reward Function

The reward is **dense and shaped** — the agent receives signal at every step, not just episode end.

| Signal | Condition | Value |
|---|---|---|
| Correct classification | `classify_alert` matches ground truth | +0.20 |
| Correct priority | Priority matches ground truth | +0.10 |
| Missing critical as benign | Classification error | −0.30 |
| Relevant log query | Source contains key evidence | +0.10–0.15 |
| Key evidence found | Log ID matches ground truth evidence | +0.10 |
| Correct asset isolation | Hostname is actually compromised | +0.25 |
| Collateral damage (DC) | Isolating healthy domain controller | −0.40 |
| Collateral damage (DB) | Isolating healthy finance database | −0.30 |
| Correct IP block | Attacker IP blocked | +0.20 |
| Correct verdict | `close_incident` verdict matches truth | +0.30 |
| Correct attack type | Attack type matches ground truth | +0.20 |
| Report with required fields | `write_report` completeness | up to +0.30 |
| Action thrashing | Same action >5× in episode | −0.05 |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start episode. Body: `{"task_id": "alert_triage"}` |
| `POST` | `/step` | Submit action. Body: `{"action_type": "...", "parameters": {...}}` |
| `GET` | `/state` | Full internal state (includes ground truth for graders) |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/health` | Health check |

---

## Quick Start

### Using Docker

```bash
# Build
docker build -t mini-soc -f server/Dockerfile .

# Run
docker run -p 8000:8000 mini-soc

# Ping
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "alert_triage"}'
```

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Using the Client

```python
from client import MiniSocEnv

with MiniSocEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="alert_triage")
    print(result.observation.alert_queue)

    step = env.step("classify_alert", {
        "alert_id": "ALT-001",
        "classification": "critical",
        "priority": "P1",
    })
    print(f"Reward: {step.reward}, Done: {step.done}")
```

### Run Inference Script

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SOC_ENV_URL=http://localhost:8000

python inference.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
mini-soc/
├── __init__.py               # OpenEnv package exports
├── models.py                 # Typed Action, Observation, State models
├── client.py                 # MiniSocEnv(EnvClient) — agent-facing client
├── openenv.yaml              # OpenEnv environment manifest
├── pyproject.toml            # Dependencies and package config
├── requirements.txt          # Pip dependencies
├── inference.py              # Baseline LLM inference script
├── run_agent.py              # Deterministic expert agent
├── server/
│   ├── __init__.py
│   ├── app.py                # FastAPI app (create_app factory)
│   ├── mini_soc_environment.py  # Core environment (reset/step/state)
│   ├── Dockerfile            # Container build
│   ├── simulator/
│   │   ├── attack_seeds.py   # Deterministic attack scenarios + ground truth
│   │   └── log_gen.py        # Synthetic log generator
│   └── graders/
│       ├── grader1.py        # Task 1 grader (classification accuracy)
│       ├── grader2.py        # Task 2 grader (evidence + verdict)
│       └── grader3.py        # Task 3 grader (containment + report)
├── tests/
│   └── test_env.py           # 18 smoke tests
├── outputs/                  # Runtime outputs (gitignored)
│   ├── logs/
│   └── evals/
├── docker-compose.yml
└── README.md
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Steps Used |
|---|---|---|
| alert_triage | 0.65 | 11 |
| incident_investigation | 0.42 | 14 |
| threat_response | 0.22 | 28 |
| **Average** | **0.43** | — |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | HuggingFace / API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `SOC_ENV_URL` | No | `http://localhost:8000` | Running environment URL |

---

## License

MIT
