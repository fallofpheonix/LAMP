from __future__ import annotations

import re
import subprocess
from pathlib import Path

from lamp.core.models import SecurityRisk
from lamp.utils.filesystem import read_text_safely

SUSPICIOUS_PATTERNS = {
    r"open\(\s*arg": "direct open() call from untrusted arg",
    r"os\.path\.join\(\s*arg": "join() with untrusted arg",
    r"pd\.read_csv\(\s*arg": "CSV read from untrusted arg",
}

IGNORED_PATH_PARTS = {
    ".git",
    ".venv",
    ".python_env",
    "site-packages",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "outputs",
    "outputs_production",
    "tests",
}


def check_security_tool_availability(tool_name: str = "safety") -> bool:
    result = subprocess.run(
        ["python3", "-m", "pip", "show", tool_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def find_path_traversal_risks(root_dir: Path) -> list[SecurityRisk]:
    findings: list[SecurityRisk] = []
    for python_file in root_dir.rglob("*.py"):
        if any(part in IGNORED_PATH_PARTS for part in python_file.parts):
            continue
        text = read_text_safely(python_file)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in SUSPICIOUS_PATTERNS:
                if re.search(pattern, line):
                    findings.append(SecurityRisk(path=python_file, pattern=pattern, line_number=line_number))
    return findings


def render_security_report(risks: list[SecurityRisk], has_security_tool: bool) -> str:
    lines = ["# Security Audit", ""]
    lines.append(f"- Dependency scanner available: {'yes' if has_security_tool else 'no'}")
    if not risks:
        lines.append("- Path traversal heuristics: no obvious issues found")
        return "\n".join(lines) + "\n"

    lines.append("- Path traversal heuristics: potential risks detected")
    lines.append("")
    for risk in risks:
        lines.append(f"- {risk.path}:{risk.line_number} matched `{risk.pattern}`")
    return "\n".join(lines) + "\n"
