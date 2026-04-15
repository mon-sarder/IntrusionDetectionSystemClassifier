"""
threat_intel.py
Static threat intelligence database mapping:
  - NSL-KDD attack patterns  →  known ransomware/malware families
  - Ransomware families       →  CVEs (specific vulnerabilities exploited)
  - Attack types              →  CWEs (weakness classes)
  - MITRE ATT&CK techniques   →  for each attack category

This is a curated, offline database. No external API calls needed.
In a production system you would augment this with live NVD/MITRE API queries.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CWE Definitions — weakness classes associated with each NSL-KDD attack type
# ─────────────────────────────────────────────────────────────────────────────

CWE_MAP = {
    # DoS — volumetric / resource exhaustion
    "dos": [
        {"id": "CWE-400", "name": "Uncontrolled Resource Consumption",
         "url": "https://cwe.mitre.org/data/definitions/400.html",
         "description": "System fails to limit resource usage, enabling exhaustion attacks."},
        {"id": "CWE-770", "name": "Allocation of Resources Without Limits",
         "url": "https://cwe.mitre.org/data/definitions/770.html",
         "description": "Memory/connection allocation without bounds checks."},
        {"id": "CWE-404", "name": "Improper Resource Shutdown / Release",
         "url": "https://cwe.mitre.org/data/definitions/404.html",
         "description": "Half-open connections (SYN flood) exploit incomplete cleanup."},
    ],
    # Probe — scanning, enumeration
    "probe": [
        {"id": "CWE-200", "name": "Exposure of Sensitive Information",
         "url": "https://cwe.mitre.org/data/definitions/200.html",
         "description": "Banner grabbing and service fingerprinting expose version info."},
        {"id": "CWE-693", "name": "Protection Mechanism Failure",
         "url": "https://cwe.mitre.org/data/definitions/693.html",
         "description": "Missing firewall or rate-limiting allows unrestricted enumeration."},
        {"id": "CWE-284", "name": "Improper Access Control",
         "url": "https://cwe.mitre.org/data/definitions/284.html",
         "description": "Services reachable without authentication permit full port scans."},
    ],
    # R2L — remote to local, credential theft
    "r2l": [
        {"id": "CWE-521", "name": "Weak Password Requirements",
         "url": "https://cwe.mitre.org/data/definitions/521.html",
         "description": "Brute-force and dictionary attacks succeed due to weak credentials."},
        {"id": "CWE-307", "name": "Improper Restriction of Auth Attempts",
         "url": "https://cwe.mitre.org/data/definitions/307.html",
         "description": "No account lockout policy enables password spraying."},
        {"id": "CWE-319", "name": "Cleartext Transmission of Sensitive Info",
         "url": "https://cwe.mitre.org/data/definitions/319.html",
         "description": "Credentials transmitted in plaintext (FTP, Telnet) can be sniffed."},
    ],
    # U2R — privilege escalation
    "u2r": [
        {"id": "CWE-269", "name": "Improper Privilege Management",
         "url": "https://cwe.mitre.org/data/definitions/269.html",
         "description": "Local user escalates to root via misconfigured SUID binaries."},
        {"id": "CWE-264", "name": "Permissions, Privileges, and Access Controls",
         "url": "https://cwe.mitre.org/data/definitions/264.html",
         "description": "Inadequate separation between user and kernel privileges."},
        {"id": "CWE-119", "name": "Buffer Overflow / Memory Corruption",
         "url": "https://cwe.mitre.org/data/definitions/119.html",
         "description": "Memory corruption exploits used to gain elevated privileges."},
    ],
    "normal": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# Ransomware / Malware Families  →  Network Behaviour Signature
# Each family lists:
#   - attack_types: which NSL-KDD categories its traffic resembles
#   - network_signature: key feature thresholds that match the family's C2/spread behaviour
#   - cves: known CVEs this family has exploited
#   - mitre: MITRE ATT&CK technique IDs
# ─────────────────────────────────────────────────────────────────────────────

RANSOMWARE_DB = {

    # ── DoS-pattern families ──────────────────────────────────────────────────

    "WannaCry": {
        "attack_types":  ["dos", "probe"],
        "category":      "Ransomware / Worm",
        "first_seen":    "2017-05-12",
        "description":   "EternalBlue-based worm that encrypts files and spreads via SMB (port 445). "
                         "Generates massive scanning traffic before lateral movement.",
        "network_signature": {
            "serror_rate":    (0.7, 1.0),
            "count":          (200, 511),
            "dst_bytes":      (0, 500),
            "protocol_type":  "tcp",
            "flag":           "S0",
        },
        "cves": [
            {"id": "CVE-2017-0144",
             "description": "EternalBlue — SMBv1 remote code execution",
             "cvss": 9.3, "url": "https://nvd.nist.gov/vuln/detail/CVE-2017-0144"},
            {"id": "CVE-2017-0145",
             "description": "EternalRomance — SMB transaction manipulation",
             "cvss": 9.3, "url": "https://nvd.nist.gov/vuln/detail/CVE-2017-0145"},
            {"id": "CVE-2017-0143",
             "description": "MS17-010 SMBv1 buffer overflow",
             "cvss": 9.3, "url": "https://nvd.nist.gov/vuln/detail/CVE-2017-0143"},
        ],
        "cwes":   ["CWE-400", "CWE-770"],
        "mitre":  ["T1190 (Exploit Public-Facing App)", "T1486 (Data Encrypted for Impact)",
                   "T1210 (Exploitation of Remote Services)", "T1071 (App Layer Protocol)"],
        "iocs":   ["Port 445 scan bursts", "S0 TCP flags", "High serror_rate",
                   "count > 400", "Tiny dst_bytes"],
        "severity": "CRITICAL",
    },

    "NotPetya": {
        "attack_types":  ["dos", "probe", "r2l"],
        "category":      "Wiper / Ransomware",
        "first_seen":    "2017-06-27",
        "description":   "Destructive wiper disguised as ransomware. Used EternalBlue + Mimikatz "
                         "for credential harvesting and lateral movement.",
        "network_signature": {
            "serror_rate":     (0.6, 1.0),
            "rerror_rate":     (0.0, 0.3),
            "diff_srv_rate":   (0.4, 1.0),
            "protocol_type":   "tcp",
        },
        "cves": [
            {"id": "CVE-2017-0144",
             "description": "EternalBlue SMBv1 (same as WannaCry)",
             "cvss": 9.3, "url": "https://nvd.nist.gov/vuln/detail/CVE-2017-0144"},
            {"id": "CVE-2017-0145",
             "description": "EternalRomance SMB transaction",
             "cvss": 9.3, "url": "https://nvd.nist.gov/vuln/detail/CVE-2017-0145"},
        ],
        "cwes":   ["CWE-400", "CWE-521", "CWE-269"],
        "mitre":  ["T1003 (Credential Dumping)", "T1486 (Data Encrypted for Impact)",
                   "T1210 (Exploitation of Remote Services)"],
        "iocs":   ["WMIC lateral movement", "PSEXEC remote execution", "MBR overwrite"],
        "severity": "CRITICAL",
    },

    "Mirai": {
        "attack_types":  ["dos"],
        "category":      "Botnet / DDoS",
        "first_seen":    "2016-08-01",
        "description":   "IoT botnet generating volumetric DDoS floods (TCP SYN, UDP, HTTP GET). "
                         "Traffic shows near-maximal count, high serror_rate.",
        "network_signature": {
            "serror_rate":  (0.85, 1.0),
            "count":        (400, 511),
            "src_bytes":    (0, 200),
            "dst_bytes":    (0, 100),
        },
        "cves": [
            {"id": "CVE-2016-10401",
             "description": "ZyXEL hard-coded credential (admin/CentryL1nk)",
             "cvss": 9.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2016-10401"},
        ],
        "cwes":   ["CWE-400", "CWE-770", "CWE-521"],
        "mitre":  ["T1498 (Network Denial of Service)", "T1595 (Active Scanning)"],
        "iocs":   ["Port 23/2323 Telnet bruteforce", "High UDP/TCP flood volume"],
        "severity": "HIGH",
    },

    "LockBit": {
        "attack_types":  ["dos", "r2l"],
        "category":      "Ransomware-as-a-Service",
        "first_seen":    "2019-09-01",
        "description":   "Fast-encrypting RaaS targeting enterprise. Uses brute-forced RDP "
                         "for initial access then generates internal scanning before encryption.",
        "network_signature": {
            "num_failed_logins": (2, 10),
            "logged_in":         0,
            "serror_rate":       (0.3, 0.8),
            "protocol_type":     "tcp",
        },
        "cves": [
            {"id": "CVE-2021-44228",
             "description": "Log4Shell — remote code execution via JNDI",
             "cvss": 10.0, "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"},
            {"id": "CVE-2021-34527",
             "description": "PrintNightmare — Windows Print Spooler RCE",
             "cvss": 8.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-34527"},
            {"id": "CVE-2018-13379",
             "description": "Fortinet FortiOS SSL-VPN path traversal",
             "cvss": 9.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2018-13379"},
        ],
        "cwes":   ["CWE-521", "CWE-307", "CWE-400"],
        "mitre":  ["T1110 (Brute Force)", "T1486 (Encrypt for Impact)",
                   "T1490 (Inhibit System Recovery)", "T1071.001 (Web Protocols)"],
        "iocs":   ["RDP brute force bursts", "Failed logins > 3", "SMB lateral scan"],
        "severity": "CRITICAL",
    },

    "BlackCat_ALPHV": {
        "attack_types":  ["probe", "r2l"],
        "category":      "Ransomware-as-a-Service",
        "first_seen":    "2021-11-01",
        "description":   "Rust-based RaaS with advanced evasion. Conducts extensive network "
                         "reconnaissance before deploying encryption.",
        "network_signature": {
            "diff_srv_rate":   (0.5, 1.0),
            "rerror_rate":     (0.4, 1.0),
            "same_srv_rate":   (0.0, 0.3),
            "count":           (20, 80),
        },
        "cves": [
            {"id": "CVE-2021-31207",
             "description": "Microsoft Exchange Server SSRF (ProxyShell)",
             "cvss": 7.2, "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-31207"},
            {"id": "CVE-2022-26134",
             "description": "Atlassian Confluence Server OGNL injection",
             "cvss": 9.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2022-26134"},
        ],
        "cwes":   ["CWE-200", "CWE-693", "CWE-284"],
        "mitre":  ["T1046 (Network Service Scanning)", "T1595 (Active Scanning)",
                   "T1078 (Valid Accounts)", "T1486 (Data Encrypted for Impact)"],
        "iocs":   ["High diff_srv_rate during recon", "REJ flag patterns", "Low same_srv_rate"],
        "severity": "CRITICAL",
    },

    "Emotet": {
        "attack_types":  ["r2l"],
        "category":      "Banking Trojan / Dropper",
        "first_seen":    "2014-06-01",
        "description":   "Polymorphic banking trojan and malware delivery platform. "
                         "Spreads via phishing, uses encrypted C2 over HTTP/HTTPS.",
        "network_signature": {
            "dst_bytes":     (5000, 150000),
            "logged_in":     1,
            "service":       "http",
            "duration":      (30, 5000),
        },
        "cves": [
            {"id": "CVE-2017-11882",
             "description": "Microsoft Office memory corruption RCE",
             "cvss": 7.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2017-11882"},
            {"id": "CVE-2018-0802",
             "description": "Microsoft Office Equation Editor buffer overflow",
             "cvss": 7.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2018-0802"},
        ],
        "cwes":   ["CWE-319", "CWE-521", "CWE-307"],
        "mitre":  ["T1566.001 (Spear Phishing Attachment)", "T1071.001 (Web Protocols)",
                   "T1027 (Obfuscated Files)", "T1547 (Boot/Logon Autostart)"],
        "iocs":   ["HTTP C2 beaconing", "Large dst_bytes", "Long session duration"],
        "severity": "HIGH",
    },

    "Conti": {
        "attack_types":  ["r2l", "u2r", "probe"],
        "category":      "Ransomware",
        "first_seen":    "2020-02-01",
        "description":   "Prolific double-extortion ransomware. Uses Cobalt Strike, "
                         "credential dumping, and domain privilege escalation.",
        "network_signature": {
            "root_shell":    1,
            "su_attempted":  1,
            "logged_in":     1,
            "duration":      (100, 10000),
        },
        "cves": [
            {"id": "CVE-2021-34527",
             "description": "PrintNightmare Windows Print Spooler RCE",
             "cvss": 8.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-34527"},
            {"id": "CVE-2020-1472",
             "description": "Zerologon — Netlogon privilege escalation",
             "cvss": 10.0, "url": "https://nvd.nist.gov/vuln/detail/CVE-2020-1472"},
            {"id": "CVE-2021-44228",
             "description": "Log4Shell",
             "cvss": 10.0, "url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"},
        ],
        "cwes":   ["CWE-269", "CWE-264", "CWE-119"],
        "mitre":  ["T1003 (Credential Dumping)", "T1068 (Privilege Escalation)",
                   "T1486 (Data Encrypted for Impact)", "T1083 (File and Directory Discovery)"],
        "iocs":   ["root_shell=1", "su_attempted=1", "Cobalt Strike beacon traffic"],
        "severity": "CRITICAL",
    },

    "Ryuk": {
        "attack_types":  ["u2r", "r2l"],
        "category":      "Ransomware",
        "first_seen":    "2018-08-01",
        "description":   "Enterprise-targeting ransomware typically deployed post-TrickBot infection. "
                         "Terminates backups, escalates privileges, encrypts network shares.",
        "network_signature": {
            "root_shell":    1,
            "num_root":      (1, 5),
            "duration":      (500, 12000),
            "dst_bytes":     (1000, 25000),
        },
        "cves": [
            {"id": "CVE-2018-8174",
             "description": "Windows VBScript Engine RCE",
             "cvss": 7.5, "url": "https://nvd.nist.gov/vuln/detail/CVE-2018-8174"},
            {"id": "CVE-2019-0604",
             "description": "Microsoft SharePoint RCE",
             "cvss": 9.8, "url": "https://nvd.nist.gov/vuln/detail/CVE-2019-0604"},
        ],
        "cwes":   ["CWE-269", "CWE-264", "CWE-119"],
        "mitre":  ["T1490 (Inhibit System Recovery)", "T1489 (Service Stop)",
                   "T1486 (Data Encrypted for Impact)", "T1134 (Access Token Manipulation)"],
        "iocs":   ["root_shell=1", "Backup process termination", "SMB share encryption"],
        "severity": "CRITICAL",
    },

    "Slammer": {
        "attack_types":  ["dos"],
        "category":      "Worm",
        "first_seen":    "2003-01-25",
        "description":   "SQL Server worm causing massive UDP flood. Historic but pattern "
                         "still appears in traffic datasets and IDS benchmarks.",
        "network_signature": {
            "protocol_type": "udp",
            "serror_rate":   (0.9, 1.0),
            "count":         (450, 511),
            "src_bytes":     (0, 500),
        },
        "cves": [
            {"id": "CVE-2002-0649",
             "description": "MS SQL Server Resolution Service stack overflow",
             "cvss": 10.0, "url": "https://nvd.nist.gov/vuln/detail/CVE-2002-0649"},
        ],
        "cwes":   ["CWE-119", "CWE-400"],
        "mitre":  ["T1498 (Network Denial of Service)"],
        "iocs":   ["376-byte UDP packets to port 1434", "Exponential traffic growth"],
        "severity": "HIGH",
    },

    "Neptune_DoS": {
        "attack_types":  ["dos"],
        "category":      "Classic DoS",
        "first_seen":    "1998-01-01",
        "description":   "SYN-flood tool from NSL-KDD era. Sends half-open TCP connections "
                         "exhausting server connection tables.",
        "network_signature": {
            "flag":         "S0",
            "serror_rate":  (0.95, 1.0),
            "count":        (490, 511),
            "dst_bytes":    (0, 10),
            "src_bytes":    (0, 50),
        },
        "cves": [
            {"id": "CVE-1999-0116",
             "description": "TCP SYN flood (generic — predates CVE system)",
             "cvss": 7.5, "url": "https://nvd.nist.gov/vuln/detail/CVE-1999-0116"},
        ],
        "cwes":   ["CWE-400", "CWE-404"],
        "mitre":  ["T1498.001 (Direct Network Flood)"],
        "iocs":   ["S0 flag", "serror_rate ≈ 1.0", "count > 490"],
        "severity": "MEDIUM",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MITRE ATT&CK technique → attack_type mapping
# ─────────────────────────────────────────────────────────────────────────────

MITRE_BY_ATTACK_TYPE = {
    "dos":    ["T1498 (Network DoS)", "T1499 (Endpoint DoS)"],
    "probe":  ["T1595 (Active Scanning)", "T1046 (Network Service Scanning)",
               "T1590 (Gather Victim Network Info)"],
    "r2l":    ["T1110 (Brute Force)", "T1078 (Valid Accounts)",
               "T1021.004 (SSH)", "T1071.001 (Web Protocols)"],
    "u2r":    ["T1068 (Exploitation for Privilege Escalation)",
               "T1055 (Process Injection)", "T1134 (Access Token Manipulation)"],
    "normal": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core lookup function
# ─────────────────────────────────────────────────────────────────────────────

def lookup_threat(attack_type: str, features: dict | None = None) -> dict:
    """
    Given a predicted attack_type (dos/probe/r2l/u2r/normal) and optionally
    the raw feature dict, return:
      - cwes:              list of CWE dicts
      - ransomware_matches: list of matching ransomware family dicts (scored)
      - mitre_techniques:  list of MITRE ATT&CK technique strings
      - risk_score:        0–100 composite risk
    """
    attack_type = attack_type.lower().strip()

    if attack_type == "normal":
        return {
            "attack_type":        "normal",
            "cwes":               [],
            "ransomware_matches": [],
            "mitre_techniques":   [],
            "risk_score":         0,
            "severity":           "NONE",
        }

    cwes             = CWE_MAP.get(attack_type, [])
    mitre_techniques = MITRE_BY_ATTACK_TYPE.get(attack_type, [])

    # Match ransomware families by attack_type
    candidates = [
        (name, fam) for name, fam in RANSOMWARE_DB.items()
        if attack_type in fam["attack_types"]
    ]

    # If we have features, score each candidate by signature match
    scored = []
    for name, fam in candidates:
        score = _signature_score(fam, features or {})
        scored.append({
            "family":      name,
            "category":    fam["category"],
            "first_seen":  fam["first_seen"],
            "description": fam["description"],
            "match_score": score,            # 0.0–1.0
            "cves":        fam["cves"],
            "cwes":        [CWE_MAP.get(attack_type, [{}])[0].get("id", "N/A")]
                           if CWE_MAP.get(attack_type) else [],
            "mitre":       fam["mitre"],
            "iocs":        fam["iocs"],
            "severity":    fam["severity"],
        })

    # Sort by match_score desc, then severity
    SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    scored.sort(key=lambda x: (-x["match_score"],
                                SEV_ORDER.get(x["severity"], 9)))

    # Risk score: 0–100 based on attack type + top match severity
    base_risk = {"dos": 65, "probe": 45, "r2l": 70, "u2r": 85}.get(attack_type, 30)
    sev_bonus = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 0}
    top_sev   = scored[0]["severity"] if scored else "LOW"
    risk      = min(100, base_risk + sev_bonus.get(top_sev, 0))

    return {
        "attack_type":        attack_type.upper(),
        "cwes":               cwes,
        "ransomware_matches": scored[:5],     # top 5
        "mitre_techniques":   mitre_techniques,
        "risk_score":         risk,
        "severity":           top_sev,
    }


def _signature_score(family: dict, features: dict) -> float:
    """
    Compare observed features against a family's network_signature.
    Returns float 0.0–1.0 representing how well the traffic matches.
    """
    sig    = family.get("network_signature", {})
    if not sig or not features:
        return 0.5   # no data to compare → neutral

    hits   = 0
    checks = 0

    for key, expected in sig.items():
        obs = features.get(key)
        if obs is None:
            continue
        checks += 1
        if isinstance(expected, tuple):
            lo, hi = expected
            if lo <= float(obs) <= hi:
                hits += 1
        elif isinstance(expected, (int, float)):
            if float(obs) == float(expected):
                hits += 1
        elif isinstance(expected, str):
            if str(obs).lower() == expected.lower():
                hits += 1

    if checks == 0:
        return 0.5
    return round(hits / checks, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_features = {
        "serror_rate": 0.98, "count": 511, "dst_bytes": 0,
        "src_bytes": 10, "protocol_type": "tcp", "flag": "S0",
    }

    for atype in ["dos", "probe", "r2l", "u2r", "normal"]:
        result = lookup_threat(atype, test_features if atype == "dos" else {})
        print(f"\n{'='*55}")
        print(f"  Attack: {atype.upper()}  |  Risk: {result['risk_score']}/100  |  Severity: {result['severity']}")
        print(f"  CWEs: {[c['id'] for c in result['cwes']]}")
        print(f"  Top ransomware matches:")
        for m in result["ransomware_matches"][:2]:
            print(f"    • {m['family']} (score={m['match_score']})  CVEs: {[c['id'] for c in m['cves']]}")