# Mini SOC — API Reference

## Architecture

![Architecture Diagram](architecture.svg)

## Environment Lifecycle

```mermaid
sequenceDiagram
    participant Agent
    participant Client (client.py)
    participant Server (server/app.py)
    participant Environment (mini_soc_environment.py)
    participant Grader

    Agent->>Client: env.reset("alert_triage")
    Client->>Server: POST /reset {"task_id": "alert_triage"}
    Server->>Environment: reset("alert_triage")
    Environment-->>Server: ResetResult(observation)
    Server-->>Client: JSON response
    Client-->>Agent: ResetResult

    loop Until done
        Agent->>Client: env.step("classify_alert", {...})
        Client->>Server: POST /step {"action_type": "...", "parameters": {...}}
        Server->>Environment: step(Action)
        Environment->>Grader: compute_step_reward(...)
        Grader-->>Environment: reward
        Environment-->>Server: StepResult(observation, reward, done)
        Server-->>Client: JSON response
        Client-->>Agent: StepResult
    end
```

## Grader Scoring

### Task 1 — Alert Triage
| Component | Weight |
|---|---|
| Classification accuracy | 70% |
| Priority correctness | 30% |
| Coverage penalty | Multiplicative |

### Task 2 — Incident Investigation
| Component | Weight |
|---|---|
| Correct verdict | 35% |
| Attack type identified | 20% |
| Evidence gathered | 30% |
| Attacker IP identified | 15% |

### Task 3 — Active Threat Response
| Component | Weight |
|---|---|
| Containment (isolate + block) | 30% |
| Collateral damage penalty | -20% |
| Evidence gathering | 20% |
| Speed of response | 10% |
| Report quality | 20% |
