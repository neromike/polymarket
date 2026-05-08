from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    print(f"\n=== Running {script_name} ===")
    completed = subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent)
    if completed.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {completed.returncode}")
    print(f"=== Finished {script_name} ===")


def main() -> int:
    steps = [
        "download_data.py",
        "luck_skill_analysis.py",
        "dashboard.py",
    ]

    try:
        for step in steps:
            run_step(step)
    except Exception as exc:
        print(f"\nPipeline stopped: {exc}", file=sys.stderr)
        return 1

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
