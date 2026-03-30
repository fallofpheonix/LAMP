from __future__ import annotations

import argparse
from pathlib import Path

from services.security_audit_service import (
    check_security_tool_availability,
    find_path_traversal_risks,
    render_security_report,
)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lightweight security checks")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="outputs/security_audit.md")
    args = parser.parse_args(argv)

    root_path = Path(args.root)
    risks = find_path_traversal_risks(root_path)
    has_security_tool = check_security_tool_availability()

    report = render_security_report(risks, has_security_tool)
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"Security report written to {out_file}")
    return 0
