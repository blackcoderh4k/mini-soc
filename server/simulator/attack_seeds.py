"""
Deterministic attack scenarios with full ground truth.
Each scenario is seeded so results are 100% reproducible.
These define what actually happened — graders compare agent actions against this.
"""
from __future__ import annotations
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# SCENARIO BANK — used across all 3 tasks
# ---------------------------------------------------------------------------

ATTACK_SCENARIOS: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # SCENARIO 1: Brute-force SSH followed by successful login
    # Used in: Task 1 (alert triage), Task 2 (incident investigation)
    # -----------------------------------------------------------------------
    "brute_force_ssh_001": {
        "scenario_id": "brute_force_ssh_001",
        "attack_type": "brute_force",
        "attacker_ip": "185.220.101.47",
        "target_hostname": "WEB-SERVER-01",
        "target_ip": "10.0.1.20",
        "compromised_user": "admin",
        "kill_chain": ["reconnaissance", "brute_force", "initial_access"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "brute_force",
            "attacker_ip": "185.220.101.47",
            "target_hostname": "WEB-SERVER-01",
            "key_evidence_log_ids": {"AUTH-001", "AUTH-002", "FW-001"},
            "key_log_sources": {"auth", "firewall"},
            "assets_to_isolate": {"WEB-SERVER-01"},
            "assets_to_not_isolate": {"DC-01", "DB-FINANCE-01"},
            "ips_to_block": {"185.220.101.47"},
            "report_required_fields": {"summary", "attack_type", "affected_assets", "attacker_ip", "timeline"},
            "affected_assets": ["WEB-SERVER-01"],
            "attacker_ips": ["185.220.101.47"],
        },
        "alerts": [
            {
                "alert_id": "ALT-001",
                "alert_type": "Multiple Failed SSH Logins",
                "severity": "high",
                "timestamp": "2024-01-15T02:14:33Z",
                "source_ip": "185.220.101.47",
                "dest_ip": "10.0.1.20",
                "dest_port": 22,
                "description": "47 failed SSH login attempts in 90 seconds from external IP",
                "raw_data": {"attempts": 47, "window_seconds": 90, "protocol": "SSH"},
            },
            {
                "alert_id": "ALT-002",
                "alert_type": "Successful Login After Brute Force",
                "severity": "critical",
                "timestamp": "2024-01-15T02:16:01Z",
                "source_ip": "185.220.101.47",
                "dest_ip": "10.0.1.20",
                "dest_port": 22,
                "description": "Successful SSH login from same IP responsible for failed attempts",
                "raw_data": {"user": "admin", "auth_method": "password"},
            },
        ],
        "logs": {
            "auth": [
                {
                    "log_id": "AUTH-001",
                    "log_source": "auth",
                    "timestamp": "2024-01-15T02:14:33Z",
                    "source_ip": "185.220.101.47",
                    "user": "admin",
                    "event_type": "authentication_failure",
                    "details": {"reason": "invalid_password", "attempt_count": 47},
                    "is_malicious": True,
                },
                {
                    "log_id": "AUTH-002",
                    "log_source": "auth",
                    "timestamp": "2024-01-15T02:16:01Z",
                    "source_ip": "185.220.101.47",
                    "user": "admin",
                    "event_type": "authentication_success",
                    "details": {"session_id": "ssh-9f3a2b", "shell": "/bin/bash"},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-001",
                    "log_source": "firewall",
                    "timestamp": "2024-01-15T02:14:30Z",
                    "source_ip": "185.220.101.47",
                    "dest_ip": "10.0.1.20",
                    "event_type": "connection_allowed",
                    "details": {"port": 22, "protocol": "TCP", "rule": "ALLOW_SSH_INBOUND"},
                    "is_malicious": True,
                },
            ],
            "process": [],
            "dns": [],
            "network": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 2: Phishing → credential theft → lateral movement
    # Used in: Task 2 (incident investigation), Task 3 (threat response)
    # -----------------------------------------------------------------------
    "phishing_lateral_001": {
        "scenario_id": "phishing_lateral_001",
        "attack_type": "lateral_movement",
        "attacker_ip": "94.102.49.190",
        "initial_victim": "WS-HR-03",
        "lateral_target": "DC-01",
        "compromised_users": ["jsmith", "domain_admin"],
        "kill_chain": [
            "phishing",
            "credential_theft",
            "lateral_movement",
            "privilege_escalation",
        ],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "lateral_movement",
            "attacker_ip": "94.102.49.190",
            "key_evidence_log_ids": {"PROC-001", "NET-001", "AUTH-010", "FW-010", "DNS-001"},
            "key_log_sources": {"process", "network", "auth", "dns"},
            "assets_to_isolate": {"WS-HR-03"},
            "assets_to_not_isolate": {"DC-01", "DB-FINANCE-01", "WEB-SERVER-01"},
            "ips_to_block": {"94.102.49.190"},
            "report_required_fields": {"summary", "attack_type", "affected_assets", "attacker_ip", "timeline"},
            "affected_assets": ["WS-HR-03", "DC-01"],
            "attacker_ips": ["94.102.49.190", "10.0.2.15"],
        },
        "alerts": [
            {
                "alert_id": "ALT-010",
                "alert_type": "Suspicious PowerShell Execution",
                "severity": "high",
                "timestamp": "2024-01-16T09:45:12Z",
                "source_ip": "10.0.2.15",
                "dest_ip": None,
                "dest_port": None,
                "description": "Encoded PowerShell command executed by non-admin user on HR workstation",
                "raw_data": {
                    "hostname": "WS-HR-03",
                    "user": "jsmith",
                    "command_length": 2048,
                    "encoded": True,
                },
            },
            {
                "alert_id": "ALT-011",
                "alert_type": "Unusual Outbound Connection",
                "severity": "medium",
                "timestamp": "2024-01-16T09:47:30Z",
                "source_ip": "10.0.2.15",
                "dest_ip": "94.102.49.190",
                "dest_port": 443,
                "description": "Workstation connecting to known C2 IP over HTTPS",
                "raw_data": {"bytes_sent": 48200, "duration_seconds": 312},
            },
            {
                "alert_id": "ALT-012",
                "alert_type": "Admin Login from Workstation",
                "severity": "critical",
                "timestamp": "2024-01-16T10:02:44Z",
                "source_ip": "10.0.2.15",
                "dest_ip": "10.0.0.5",
                "dest_port": 445,
                "description": "Domain admin credentials used from HR workstation to access Domain Controller",
                "raw_data": {"user": "domain_admin", "target": "DC-01", "protocol": "SMB"},
            },
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-001",
                    "log_source": "process",
                    "timestamp": "2024-01-16T09:45:12Z",
                    "source_ip": "10.0.2.15",
                    "user": "jsmith",
                    "event_type": "process_created",
                    "details": {
                        "process": "powershell.exe",
                        "parent": "outlook.exe",
                        "args": "-EncodedCommand JABzAD0ATgBlAHcA...",
                        "hostname": "WS-HR-03",
                    },
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-001",
                    "log_source": "network",
                    "timestamp": "2024-01-16T09:47:30Z",
                    "source_ip": "10.0.2.15",
                    "dest_ip": "94.102.49.190",
                    "event_type": "outbound_connection",
                    "details": {"port": 443, "bytes": 48200, "threat_intel": "known_c2"},
                    "is_malicious": True,
                },
            ],
            "auth": [
                {
                    "log_id": "AUTH-010",
                    "log_source": "auth",
                    "timestamp": "2024-01-16T10:02:44Z",
                    "source_ip": "10.0.2.15",
                    "user": "domain_admin",
                    "event_type": "authentication_success",
                    "details": {"target_host": "DC-01", "protocol": "Kerberos", "anomaly": "unusual_source"},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-010",
                    "log_source": "firewall",
                    "timestamp": "2024-01-16T09:47:25Z",
                    "source_ip": "10.0.2.15",
                    "dest_ip": "94.102.49.190",
                    "event_type": "connection_allowed",
                    "details": {"port": 443, "direction": "outbound"},
                    "is_malicious": True,
                },
            ],
            "dns": [
                {
                    "log_id": "DNS-001",
                    "log_source": "dns",
                    "timestamp": "2024-01-16T09:47:20Z",
                    "source_ip": "10.0.2.15",
                    "event_type": "dns_query",
                    "details": {"query": "update.microsoft-cdn.net", "resolved_ip": "94.102.49.190", "threat_intel": "domain_spoofing"},
                    "is_malicious": True,
                },
            ],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 3: False positive — port scan from internal IT scanner
    # Used in: Task 1 (alert triage)
    # -----------------------------------------------------------------------
    "false_positive_scan_001": {
        "scenario_id": "false_positive_scan_001",
        "attack_type": "false_positive",
        "ground_truth": {
            "classification": "benign",
            "priority": "P4",
            "verdict": "false_positive",
            "attack_type": "false_positive",
            "attacker_ip": "10.0.0.100",
            "target_hostname": "Internal subnet",
            "key_evidence_log_ids": {"FW-020"},
            "key_log_sources": {"firewall"},
            "assets_to_isolate": set(),
            "assets_to_not_isolate": {"IT-SCANNER-01"},
            "ips_to_block": set(),
            "report_required_fields": {"summary", "attack_type"},
            "affected_assets": [],
            "attacker_ips": [],
        },
        "alerts": [
            {
                "alert_id": "ALT-020",
                "alert_type": "Port Scan Detected",
                "severity": "medium",
                "timestamp": "2024-01-17T03:00:10Z",
                "source_ip": "10.0.0.100",
                "dest_ip": "10.0.0.0/24",
                "dest_port": None,
                "description": "Internal IP scanning full subnet on common service ports",
                "raw_data": {"ports_scanned": [22, 80, 443, 3389, 8080], "scanner": "nmap"},
            },
        ],
        "logs": {
            "firewall": [
                {
                    "log_id": "FW-020",
                    "log_source": "firewall",
                    "timestamp": "2024-01-17T03:00:10Z",
                    "source_ip": "10.0.0.100",
                    "dest_ip": "10.0.0.0/24",
                    "event_type": "port_scan",
                    "details": {
                        "scanner_hostname": "IT-SCANNER-01",
                        "authorized": True,
                        "schedule": "weekly_sunday_0300",
                    },
                    "is_malicious": False,
                },
            ],
            "auth": [], "process": [], "dns": [], "network": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 4: Ransomware — Fast encryption spread
    # New for v2.0
    # -----------------------------------------------------------------------
    "ransomware_001": {
        "scenario_id": "ransomware_001",
        "attack_type": "malware",
        "attacker_ip": "45.33.22.11",
        "target_hostname": "WS-FINANCE-01",
        "kill_chain": ["initial_access", "execution", "impact"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "malware",
            "attacker_ip": "45.33.22.11",
            "target_hostname": "WS-FINANCE-01",
            "key_evidence_log_ids": {"PROC-040", "PROC-041", "NET-040"},
            "key_log_sources": {"process", "network"},
            "assets_to_isolate": {"WS-FINANCE-01"},
            "assets_to_not_isolate": {"DB-FINANCE-01", "DC-01"},
            "ips_to_block": {"45.33.22.11"},
            "report_required_fields": {"summary", "attack_type", "affected_assets", "attacker_ip", "timeline"},
            "affected_assets": ["WS-FINANCE-01"],
            "attacker_ips": ["45.33.22.11"],
        },
        "alerts": [
            {
                "alert_id": "ALT-040",
                "alert_type": "Mass File Renaming Detected",
                "severity": "critical",
                "timestamp": "2024-01-18T11:20:05Z",
                "source_ip": "10.0.2.50",
                "description": "Over 500 files renamed to .encrypted extension on WS-FINANCE-01",
                "raw_data": {"extension": ".encrypted", "count": 542},
            },
            {
                "alert_id": "ALT-041",
                "alert_type": "Shadow Copy Deletion",
                "severity": "high",
                "timestamp": "2024-01-18T11:21:30Z",
                "source_ip": "10.0.2.50",
                "description": "vssadmin.exe used to delete shadow copies (common ransomware behavior)",
                "raw_data": {"process": "vssadmin.exe", "command": "delete shadows /all /quiet"},
            },
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-040",
                    "log_source": "process",
                    "timestamp": "2024-01-18T11:20:00Z",
                    "source_ip": "10.0.2.50",
                    "event_type": "file_operation",
                    "details": {"action": "rename", "new_ext": ".encrypted", "path": "C:\\Users\\bwalker\\Documents\\*"},
                    "is_malicious": True,
                },
                {
                    "log_id": "PROC-041",
                    "log_source": "process",
                    "timestamp": "2024-01-18T11:21:30Z",
                    "source_ip": "10.0.2.50",
                    "event_type": "process_created",
                    "details": {"process": "vssadmin.exe", "args": "delete shadows /all /quiet"},
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-040",
                    "log_source": "network",
                    "timestamp": "2024-01-18T11:15:00Z",
                    "source_ip": "10.0.2.50",
                    "dest_ip": "45.33.22.11",
                    "event_type": "outbound_connection",
                    "details": {"port": 80, "payload": "key_exchange"},
                    "is_malicious": True,
                }
            ],
            "auth": [], "firewall": [], "dns": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 5: Insider Threat — Data Exfiltration
    # New for v2.0
    # -----------------------------------------------------------------------
    "insider_threat_001": {
        "scenario_id": "insider_threat_001",
        "attack_type": "data_exfiltration",
        "attacker_ip": None,  # Internal user
        "target_hostname": "DB-FINANCE-01",
        "kill_chain": ["collection", "exfiltration"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "data_exfiltration",
            "attacker_ip": "10.0.2.15", # jsmith's IP
            "target_hostname": "DB-FINANCE-01",
            "key_evidence_log_ids": {"AUTH-050", "NET-050"},
            "key_log_sources": {"auth", "network"},
            "assets_to_isolate": {"WS-HR-03"}, # user's workstation
            "assets_to_not_isolate": {"DB-FINANCE-01", "DC-01"},
            "ips_to_block": {"203.0.113.5"},
            "report_required_fields": {"summary", "attack_type", "affected_assets", "attacker_ip"},
            "affected_assets": ["DB-FINANCE-01"],
            "attacker_ips": ["10.0.2.15"],
        },
        "alerts": [
            {
                "alert_id": "ALT-050",
                "alert_type": "Anomalous Database Query",
                "severity": "high",
                "timestamp": "2024-01-19T22:45:00Z",
                "source_ip": "10.0.0.30",
                "description": "User 'jsmith' queried 1.5 million records from Payroll table after hours",
                "raw_data": {"user": "jsmith", "rows": 1500000, "table": "Payroll"},
            },
        ],
        "logs": {
            "auth": [
                {
                    "log_id": "AUTH-050",
                    "log_source": "auth",
                    "timestamp": "2024-01-19T22:30:00Z",
                    "source_ip": "10.0.2.15",
                    "user": "jsmith",
                    "event_type": "authentication_success",
                    "details": {"target": "DB-FINANCE-01", "method": "SQL_AUTH"},
                    "is_malicious": True,
                }
            ],
            "network": [
                {
                    "log_id": "NET-050",
                    "log_source": "network",
                    "timestamp": "2024-01-19T23:05:00Z",
                    "source_ip": "10.0.2.15",
                    "dest_ip": "203.0.113.5",  # personal dropbox
                    "event_type": "data_upload",
                    "details": {"bytes": 450000000, "destination": "dropbox.com"},
                    "is_malicious": True,
                }
            ],
            "firewall": [], "process": [], "dns": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 6: Supply Chain — Backdoored Update
    # New for v2.0
    # -----------------------------------------------------------------------
    "supply_chain_001": {
        "scenario_id": "supply_chain_001",
        "attack_type": "malware",
        "attacker_ip": "88.77.66.55",
        "target_hostname": "BACKUP-SRV-01",
        "kill_chain": ["initial_access", "persistence"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P2",
            "verdict": "true_positive",
            "attack_type": "malware",
            "attacker_ip": "88.77.66.55",
            "target_hostname": "BACKUP-SRV-01",
            "key_evidence_log_ids": {"PROC-060", "DNS-060"},
            "key_log_sources": {"process", "dns"},
            "assets_to_isolate": {"BACKUP-SRV-01"},
            "assets_to_not_isolate": {"DC-01", "DB-FINANCE-01"},
            "ips_to_block": {"88.77.66.55"},
            "report_required_fields": {"summary", "attack_type", "affected_assets", "attacker_ip"},
            "affected_assets": ["BACKUP-SRV-01"],
            "attacker_ips": ["88.77.66.55"],
        },
        "alerts": [
            {
                "alert_id": "ALT-060",
                "alert_type": "Unsigned Driver Load",
                "severity": "medium",
                "timestamp": "2024-01-20T04:12:00Z",
                "source_ip": "10.0.0.20",
                "description": "Atypical driver load following software update on BACKUP-SRV-01",
                "raw_data": {"driver": "bkp_helper.sys", "signed": False},
            },
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-060",
                    "log_source": "process",
                    "timestamp": "2024-01-20T04:10:00Z",
                    "source_ip": "10.0.0.20",
                    "event_type": "process_created",
                    "details": {"process": "msiexec.exe", "args": "BackupTool_v4.2.msi"},
                    "is_malicious": True,
                }
            ],
            "dns": [
                {
                    "log_id": "DNS-060",
                    "log_source": "dns",
                    "timestamp": "2024-01-20T04:15:00Z",
                    "source_ip": "10.0.0.20",
                    "event_type": "dns_query",
                    "details": {"query": "cdn.legitbackup.com", "resolved_ip": "88.77.66.55"},
                    "is_malicious": True,
                }
            ],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 7: Multi-stage APT — Stealthy Exfiltration
    # -----------------------------------------------------------------------
    "multi_stage_apt_001": {
        "scenario_id": "multi_stage_apt_001",
        "attack_type": "lateral_movement",
        "attacker_ip": "103.45.67.89",
        "target_hostname": "DC-01",
        "kill_chain": ["reconnaissance", "exploitation", "lateral_movement", "exfiltration"],
        "ground_truth": {
            "classification": "critical", "priority": "P1", "verdict": "true_positive",
            "attack_type": "lateral_movement", "attacker_ip": "103.45.67.89",
            "target_hostname": "DC-01",
            "key_evidence_log_ids": {"PROC-070", "AUTH-070", "DNS-070"},
            "key_log_sources": {"process", "auth", "dns"},
            "assets_to_isolate": {"WEB-SERVER-01", "DC-01"},
            "ips_to_block": {"103.45.67.89"},
        },
        "alerts": [
            {
                "alert_id": "ALT-070", "alert_type": "LSASS Memory Dump", "severity": "critical",
                "timestamp": "2024-01-21T02:00:00Z", "source_ip": "10.0.1.20",
                "description": "Mimikatz-style LSASS memory dump detected on WEB-SERVER-01",
            }
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-070", "log_source": "process", "timestamp": "2024-01-21T02:00:00Z",
                    "source_ip": "10.0.1.20", "event_type": "process_created",
                    "details": {"process": "mimikatz.exe", "args": "sekurlsa::logonpasswords"}, "is_malicious": True,
                }
            ],
            "auth": [
                {
                    "log_id": "AUTH-070", "log_source": "auth", "timestamp": "2024-01-21T02:15:00Z",
                    "source_ip": "10.0.1.20", "user": "domain_admin", "event_type": "authentication_success",
                    "details": {"target": "DC-01", "protocol": "NTLM"}, "is_malicious": True,
                }
            ],
            "dns": [
                {
                    "log_id": "DNS-070", "log_source": "dns", "timestamp": "2024-01-21T03:00:00Z",
                    "source_ip": "10.0.0.5", "event_type": "dns_query",
                    "details": {"query": "a.b.c.tunnel.attacker.com", "record_type": "TXT", "length": 512}, "is_malicious": True,
                }
            ],
            "firewall": [], "network": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 8: BEC — Session Hijacking & Wire Fraud
    # -----------------------------------------------------------------------
    "bec_fraud_001": {
        "scenario_id": "bec_fraud_001",
        "attack_type": "account_takeover",
        "attacker_ip": "192.168.50.100",
        "target_hostname": "MAIL-SRV-01",
        "kill_chain": ["initial_access", "collection", "exfiltration"],
        "ground_truth": {
            "classification": "critical", "priority": "P1", "verdict": "true_positive",
            "attack_type": "account_takeover", "attacker_ip": "192.168.50.100",
            "key_evidence_log_ids": {"AUTH-080", "FW-080"},
            "key_log_sources": {"auth", "firewall"},
            "assets_to_isolate": {"MAIL-SRV-01"},
            "ips_to_block": {"192.168.50.100"},
        },
        "alerts": [
            {
                "alert_id": "ALT-080", "alert_type": "Session Cookie Reuse", "severity": "high",
                "timestamp": "2024-01-22T14:10:00Z", "source_ip": "192.168.50.100",
                "description": "Session cookie for 'ceo@corp.com' used from a new IP address.",
            }
        ],
        "logs": {
            "auth": [
                {
                    "log_id": "AUTH-080", "log_source": "auth", "timestamp": "2024-01-22T14:10:00Z",
                    "source_ip": "192.168.50.100", "user": "ceo@corp.com", "event_type": "session_resumed",
                    "details": {"user_agent": "Python-requests/2.25.1", "cookie_id": "sess_998877"}, "is_malicious": True,
                }
            ],
            "firewall": [
                {
                    "log_id": "FW-080", "log_source": "firewall", "timestamp": "2024-01-22T14:15:00Z",
                    "source_ip": "192.168.50.100", "dest_ip": "10.0.5.5", "event_type": "connection_allowed",
                    "details": {"port": 443, "data_sent": "Wire_Transfer_Auth.pdf"}, "is_malicious": True,
                }
            ],
            "process": [], "dns": [], "network": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 9: Blind SQL Injection — Database Exfiltration
    # -----------------------------------------------------------------------
    "sql_injection_001": {
        "scenario_id": "sql_injection_001",
        "attack_type": "exploitation",
        "attacker_ip": "172.16.0.44",
        "target_hostname": "WEB-SERVER-01",
        "kill_chain": ["exploitation", "collection"],
        "ground_truth": {
            "classification": "critical", "priority": "P1", "verdict": "true_positive",
            "attack_type": "exploitation", "attacker_ip": "172.16.0.44",
            "key_evidence_log_ids": {"NET-090", "DNS-090"},
            "key_log_sources": {"network", "dns"},
            "assets_to_isolate": {"WEB-SERVER-01"},
            "ips_to_block": {"172.16.0.44"},
        },
        "alerts": [
            {
                "alert_id": "ALT-090", "alert_type": "SQLi Pattern Detected", "severity": "high",
                "timestamp": "2024-01-23T18:00:00Z", "source_ip": "172.16.0.44",
                "description": "Repetitive 500 error codes on /api/user?id=.",
            }
        ],
        "logs": {
            "network": [
                {
                    "log_id": "NET-090", "log_source": "network", "timestamp": "2024-01-23T18:00:00Z",
                    "source_ip": "172.16.0.44", "event_type": "http_request",
                    "details": {"uri": "/api/user?id=1' AND (SELECT 1 FROM (SELECT(SLEEP(5)))a)--", "status": 500}, "is_malicious": True,
                }
            ],
            "dns": [
                {
                    "log_id": "DNS-090", "log_source": "dns", "timestamp": "2024-01-23T18:05:00Z",
                    "source_ip": "10.0.1.20", "event_type": "dns_query",
                    "details": {"query": "hex_data_part1.attacker-dns.com"}, "is_malicious": True,
                }
            ],
            "auth": [], "firewall": [], "process": [],
        },
    },
    # -----------------------------------------------------------------------
    # SCENARIO 10: APT - Living off the Land / Lateral WMI
    # -----------------------------------------------------------------------
    "ghost_apt_001": {
        "scenario_id": "ghost_apt_001",
        "attack_type": "lateral_movement",
        "attacker_ip": "10.0.0.5",
        "target_hostname": "WS-01",
        "kill_chain": ["lateral_movement", "persistence"],
        "ground_truth": {
            "classification": "critical", "priority": "P1", "verdict": "true_positive",
            "attack_type": "apt", "attacker_ip": "10.0.0.5",
            "key_evidence_log_ids": {"WMI-100", "TASK-100"},
            "key_log_sources": {"process", "auth"},
            "assets_to_isolate": {"WS-01", "WS-02"},
            "ips_to_block": ["10.0.0.5"],
        },
        "alerts": [
            {
                "alert_id": "ALT-100", "alert_type": "Unusual WMI Process Creation", "severity": "high",
                "timestamp": "2024-01-24T09:00:00Z", "source_ip": "10.0.0.5",
                "description": "WMIC used to spawn powershell.exe on remote node WS-02.",
            }
        ],
        "logs": {
            "process": [
                {
                    "log_id": "WMI-100", "log_source": "process", "timestamp": "2024-01-24T09:00:00Z",
                    "hostname": "WS-01", "event_type": "process_creation",
                    "details": {"command": "wmic /node:WS-02 process call create 'powershell.exe -enc ZmllcmNl'"}, "is_malicious": True,
                },
                {
                    "log_id": "TASK-100", "log_source": "process", "timestamp": "2024-01-24T09:10:00Z",
                    "hostname": "WS-02", "event_type": "process_creation",
                    "details": {"command": "schtasks /create /tn 'SystemUpdate' /tr 'C:\\Windows\\Temp\\ghost.exe'"}, "is_malicious": True,
                }
            ],
            "auth": [
                {
                    "log_id": "AUTH-100", "log_source": "auth", "timestamp": "2024-01-24T09:00:00Z",
                    "source_ip": "10.0.0.5", "user": "IT-ADMIN", "event_type": "authentication_success",
                    "details": {"method": "WMI/Remote"}, "is_malicious": True,
                }
            ],
            "firewall": [], "dns": [], "network": [],
        },
    },
}


# ---------------------------------------------------------------------------
# TASK 1: Alert queue (10 alerts, mix of scenarios)
# ---------------------------------------------------------------------------

TASK1_ALERT_QUEUE = [
    # From brute_force_ssh_001
    {**ATTACK_SCENARIOS["brute_force_ssh_001"]["alerts"][0], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    {**ATTACK_SCENARIOS["brute_force_ssh_001"]["alerts"][1], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    # From phishing_lateral_001
    {**ATTACK_SCENARIOS["phishing_lateral_001"]["alerts"][0], "ground_truth_classification": "suspicious", "ground_truth_priority": "P2"},
    {**ATTACK_SCENARIOS["phishing_lateral_001"]["alerts"][1], "ground_truth_classification": "suspicious", "ground_truth_priority": "P2"},
    {**ATTACK_SCENARIOS["phishing_lateral_001"]["alerts"][2], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    # From Ransomware (ALT-040)
    {**ATTACK_SCENARIOS["ransomware_001"]["alerts"][0], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    # From false positive
    {**ATTACK_SCENARIOS["false_positive_scan_001"]["alerts"][0], "ground_truth_classification": "benign", "ground_truth_priority": "P4"},
    # Additional synthetic benign alerts
    {
        "alert_id": "ALT-030",
        "alert_type": "User Password Changed",
        "severity": "low",
        "timestamp": "2024-01-17T09:00:00Z",
        "source_ip": "10.0.1.50",
        "dest_ip": None, "dest_port": None,
        "description": "Standard password reset via IT helpdesk portal",
        "raw_data": {"user": "bwalker", "method": "helpdesk_ticket"},
        "ground_truth_classification": "benign",
        "ground_truth_priority": "P4",
    },
    {
        "alert_id": "ALT-031",
        "alert_type": "After-hours Login",
        "severity": "medium",
        "timestamp": "2024-01-17T23:15:00Z",
        "source_ip": "10.0.1.75",
        "dest_ip": "10.0.0.10", "dest_port": 443,
        "description": "Employee login outside business hours from internal IP",
        "raw_data": {"user": "mchen", "vpn": True, "hr_approved_overtime": True},
        "ground_truth_classification": "benign",
        "ground_truth_priority": "P3",
    },
    {
        "alert_id": "ALT-033",
        "alert_type": "Tor Exit Node Connection Attempt",
        "severity": "high",
        "timestamp": "2024-01-17T11:45:00Z",
        "source_ip": "198.96.155.3",
        "dest_ip": "10.0.1.20", "dest_port": 80,
        "description": "Inbound connection attempt from known Tor exit node, blocked by firewall",
        "raw_data": {"blocked": True, "threat_intel": "tor_exit_node"},
        "ground_truth_classification": "suspicious",
        "ground_truth_priority": "P2",
    },
]


# ---------------------------------------------------------------------------
# ASSET INVENTORY
# ---------------------------------------------------------------------------

ASSET_INVENTORY = [
    {"hostname": "WEB-SERVER-01", "ip_address": "10.0.1.20", "asset_type": "server", "criticality": 4, "owner": "IT Ops", "department": "Engineering", "is_compromised": False, "is_isolated": False},
    {"hostname": "DC-01", "ip_address": "10.0.0.5", "asset_type": "domain_controller", "criticality": 5, "owner": "IT Security", "department": "IT", "is_compromised": False, "is_isolated": False},
    {"hostname": "WS-HR-03", "ip_address": "10.0.2.15", "asset_type": "workstation", "criticality": 2, "owner": "Jane Smith", "department": "HR", "is_compromised": False, "is_isolated": False},
    {"hostname": "DB-FINANCE-01", "ip_address": "10.0.0.30", "asset_type": "database", "criticality": 5, "owner": "Finance", "department": "Finance", "is_compromised": False, "is_isolated": False},
    {"hostname": "IT-SCANNER-01", "ip_address": "10.0.0.100", "asset_type": "workstation", "criticality": 1, "owner": "IT Ops", "department": "IT", "is_compromised": False, "is_isolated": False},
    {"hostname": "BACKUP-SRV-01", "ip_address": "10.0.0.20", "asset_type": "server", "criticality": 3, "owner": "IT Ops", "department": "IT", "is_compromised": False, "is_isolated": False},
    {"hostname": "WS-FINANCE-01", "ip_address": "10.0.2.50", "asset_type": "workstation", "criticality": 3, "owner": "Bob Walker", "department": "Finance", "is_compromised": False, "is_isolated": False},
]
