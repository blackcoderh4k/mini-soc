"""
Mini SOC — OpenEnv Client
==========================
Implements the EnvClient interface for agents to interact with the
Mini SOC environment over HTTP.

Usage:
    from client import MiniSocEnv

    env = MiniSocEnv(base_url="http://localhost:8000")
    obs = env.reset(task_id="alert_triage")
    result = env.step(action_type="classify_alert", parameters={...})
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from models import (
    Action,
    ActionType,
    Observation,
    ResetResult,
    StepResult,
    StateResult,
)


class MiniSocEnv:
    """
    OpenEnv client for the Mini SOC environment.
    Communicates with the server over HTTP (reset / step / state).
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Core OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "alert_triage") -> ResetResult:
        """Start a new episode for the given task."""
        resp = self._client.post(
            f"{self._base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return ResetResult(**resp.json())

    def step(self, action_type: str, parameters: Optional[Dict[str, Any]] = None) -> StepResult:
        """Submit one agent action. Returns next observation, reward, done."""
        resp = self._client.post(
            f"{self._base_url}/step",
            json={
                "action_type": action_type,
                "parameters": parameters or {},
            },
        )
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> StateResult:
        """Retrieve the full internal environment state (includes ground truth)."""
        resp = self._client.get(f"{self._base_url}/state")
        resp.raise_for_status()
        return StateResult(**resp.json())

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check if the environment server is reachable."""
        resp = self._client.get(f"{self._base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        """List available tasks with metadata."""
        resp = self._client.get(f"{self._base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
