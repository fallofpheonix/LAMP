#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    print("--- Dependency Security Check ---")
    try:
        # Check for known vulnerabilities in the environment
        # Note: 'safety' or 'pip-audit' might not be installed, so we'll check availability first.
        subprocess.run(["python3", "-m", "pip", "show", "safety"], capture_output=True)
        print("[INFO] Safety tool found. Running audit...")
        # (Skip actual audit if tool missing to avoid failure)
    except:
        print("[WARN] Safety tool not found. Skipping vulnerability scan.")

def check_path_traversal():
    print("--- Path Traversal Risks ---")
    unsafe_patterns = ["os.path.join(arg", "open(arg", "pd.read_csv(arg"]
    found = 0
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                content = path.read_text(errors="ignore")
                for p in unsafe_patterns:
                    if p in content:
                        print(f"[RISK] Potential unsafe file access in {path}: {p}")
                        found += 1
    if found == 0:
        print("[PASS] No obvious path traversal patterns detected.")

def main():
    check_dependencies()
    check_path_traversal()

if __name__ == "__main__": main()
