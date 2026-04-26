"""
Synthetic log generator and asset graph utilities.
Builds log entries and asset inventory from attack scenario seeds.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

# Dual-import pattern for OpenEnv Docker compatibility
try:
    from models import LogEntry, Asset
except ImportError:
    from ...models import LogEntry, Asset

try:
    from server.simulator.attack_seeds import ATTACK_SCENARIOS, ASSET_INVENTORY
except ImportError:
    from .attack_seeds import ATTACK_SCENARIOS, ASSET_INVENTORY


def get_logs_for_source(
    scenario_id: str,
    log_source: str,
    filter_ip: Optional[str] = None,
    filter_user: Optional[str] = None,
) -> List[LogEntry]:
    """
    Returns log entries for a given scenario and log source.
    Optionally filters by IP or user (simulates a log query).
    The is_malicious field is included in the model but not exposed in observations.
    """
    scenario = ATTACK_SCENARIOS.get(scenario_id)
    if not scenario:
        return []

    raw_logs = scenario.get("logs", {}).get(log_source, [])
    entries = []
    for raw in raw_logs:
        entry = LogEntry(**raw)

        if filter_ip and entry.source_ip != filter_ip and entry.dest_ip != filter_ip:
            continue
        if filter_user and entry.user != filter_user:
            continue

        entries.append(entry)

    return entries


def get_all_logs_for_scenario(scenario_id: str) -> Dict[str, List[LogEntry]]:
    """Returns all logs for a scenario, organized by source."""
    scenario = ATTACK_SCENARIOS.get(scenario_id)
    if not scenario:
        return {}
    result = {}
    for source, raw_list in scenario.get("logs", {}).items():
        result[source] = [LogEntry(**r) for r in raw_list]
    return result


def build_asset_inventory(scenario_id: Optional[str] = None) -> List[Asset]:
    """
    Returns asset inventory. If scenario provided, marks compromised assets
    according to ground truth (is_compromised hidden from agent observations).
    """
    assets = [Asset(**a) for a in ASSET_INVENTORY]

    if scenario_id:
        scenario = ATTACK_SCENARIOS.get(scenario_id)
        if scenario:
            affected = scenario.get("ground_truth", {}).get("affected_assets", [])
            for asset in assets:
                if asset.hostname in affected:
                    asset.is_compromised = True

    return assets


def sanitize_for_agent(assets: List[Asset]) -> List[Asset]:
    """
    Returns asset list safe to show agent — hides is_compromised ground truth.
    """
    sanitized = []
    for a in assets:
        copy = a.model_copy()
        copy.is_compromised = False  # never reveal ground truth to agent
        sanitized.append(copy)
    return sanitized


def get_benign_log_noise(log_source: str, count: int = 3) -> List[LogEntry]:
    """
    Returns benign background log noise to make log querying non-trivial.
    Agent must distinguish signal from noise.
    """
    noise_templates = {
        "auth": [
            LogEntry(log_id="NOISE-AUTH-1", log_source="auth", timestamp="2024-01-17T08:00:01Z",
                     source_ip="10.0.1.10", user="alice", event_type="authentication_success",
                     details={"method": "kerberos"}, is_malicious=False),
            LogEntry(log_id="NOISE-AUTH-2", log_source="auth", timestamp="2024-01-17T08:15:33Z",
                     source_ip="10.0.1.11", user="bob", event_type="authentication_success",
                     details={"method": "kerberos"}, is_malicious=False),
            LogEntry(log_id="NOISE-AUTH-3", log_source="auth", timestamp="2024-01-17T08:30:00Z",
                     source_ip="10.0.1.12", user="carol", event_type="authentication_failure",
                     details={"reason": "wrong_password", "attempt_count": 1}, is_malicious=False),
        ],
        "firewall": [
            LogEntry(log_id="NOISE-FW-1", log_source="firewall", timestamp="2024-01-17T07:55:00Z",
                     source_ip="10.0.1.10", dest_ip="8.8.8.8", event_type="connection_allowed",
                     details={"port": 443}, is_malicious=False),
            LogEntry(log_id="NOISE-FW-2", log_source="firewall", timestamp="2024-01-17T08:00:00Z",
                     source_ip="10.0.1.50", dest_ip="10.0.0.5", event_type="connection_allowed",
                     details={"port": 389}, is_malicious=False),
        ],
        "dns": [
            LogEntry(log_id="NOISE-DNS-1", log_source="dns", timestamp="2024-01-17T07:00:00Z",
                     source_ip="10.0.1.10", event_type="dns_query",
                     details={"query": "google.com", "resolved_ip": "142.250.80.46"}, is_malicious=False),
        ],
        "process": [
            LogEntry(log_id="NOISE-PROC-1", log_source="process", timestamp="2024-01-17T08:01:00Z",
                     source_ip="10.0.1.10", user="alice", event_type="process_created",
                     details={"process": "chrome.exe", "parent": "explorer.exe"}, is_malicious=False),
        ],
        "network": [
            LogEntry(log_id="NOISE-NET-1", log_source="network", timestamp="2024-01-17T08:05:00Z",
                     source_ip="10.0.1.10", dest_ip="10.0.0.5", event_type="outbound_connection",
                     details={"port": 443, "bytes": 1200}, is_malicious=False),
        ],
    }
    return noise_templates.get(log_source, [])[:count]
