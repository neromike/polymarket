from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent


def run_step(args: List[str]) -> None:
    command = [sys.executable, str(ROOT / "app.py"), *args]
    print(f"\n=== python app.py {' '.join(args)} ===")
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {completed.returncode}: {' '.join(args)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper around app.py. Prefer app.py for targeted update/analyze/dashboard runs."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip all API update steps and run analysis/dashboard generation from cached data only.",
    )
    parser.add_argument(
        "--fast-update",
        action="store_true",
        help="When updating users, skip price-history refresh.",
    )
    parser.add_argument(
        "--skip-market-trades",
        action="store_true",
        help="Skip the market-trade cache refresh step.",
    )
    parser.add_argument(
        "--scanner-allow-api",
        action="store_true",
        help="Allow scanner analysis to refresh missing market-trade caches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps: List[List[str]] = []

    if not args.offline:
        steps.append(["update", "markets"])
        update_users = ["update", "users"]
        if args.fast_update:
            update_users.append("--skip-price-history")
        steps.append(update_users)
        if not args.skip_market_trades:
            steps.append(["update", "market-trades"])

    steps.append(["analyze", "luck"])
    scanner = ["analyze", "scanner"]
    if args.scanner_allow_api:
        scanner.append("--allow-api")
    steps.append(scanner)
    steps.append(["dashboard", "build"])

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
