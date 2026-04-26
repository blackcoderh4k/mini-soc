"""
Shared test fixtures for Mini SOC test suite.
"""
import pytest
from server.mini_soc_environment import SocEnvironment


@pytest.fixture
def env():
    """Fresh SocEnvironment instance for each test."""
    return SocEnvironment()


@pytest.fixture
def env_task1(env):
    """Environment pre-reset to alert_triage task."""
    env.reset("alert_triage")
    return env


@pytest.fixture
def env_task2(env):
    """Environment pre-reset to incident_investigation task."""
    env.reset("incident_investigation")
    return env


@pytest.fixture
def env_task3(env):
    """Environment pre-reset to threat_response task."""
    env.reset("threat_response")
    return env
