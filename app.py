from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from runtime_state import (
    build_inventory,
    command_text,
    finish_run,
    list_jobs,
    list_latest_runs,
    load_watermarks,
    start_run,
)


ROOT = Path(__file__).resolve().parent


def script(name: str) -> str:
    return str(ROOT / name)


def run_steps(
    *,
    name: str,
    kind: str,
    commands: Sequence[Sequence[str]],
    metadata: Dict[str, Any] | None = None,
) -> int:
    pseudo_command = ["step"] + [" | ".join(command_text(cmd) for cmd in commands)]
    run = start_run(ROOT, name=name, kind=kind, command=pseudo_command, metadata=metadata)
    for command in commands:
        print(f"\n=== {command_text(command)} ===", flush=True)
        try:
            completed = subprocess.run(list(command), cwd=ROOT)
        except KeyboardInterrupt:
            finish_run(ROOT, run, returncode=130, error="Cancelled by KeyboardInterrupt")
            raise
        if completed.returncode != 0:
            finish_run(ROOT, run, returncode=completed.returncode, error=f"Step failed: {command_text(command)}")
            return completed.returncode
    finish_run(ROOT, run, returncode=0)
    return 0


def add_option(command: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    command.extend([flag, str(value)])


def add_repeated(command: List[str], flag: str, values: Iterable[str] | None) -> None:
    for value in values or []:
        if str(value).strip():
            command.extend([flag, str(value)])


def add_flag(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def update_users_command(args: argparse.Namespace) -> List[str]:
    command = [sys.executable, script("download_data.py")]
    add_option(command, "--data-dir", args.data_dir)
    add_repeated(command, "--user", args.user)
    add_repeated(command, "--users-file", args.users_file)
    add_option(command, "--sleep", args.sleep)
    add_option(command, "--timeout-seconds", args.timeout_seconds)
    add_option(command, "--max-retries", args.max_retries)
    add_option(command, "--trades-page-limit", args.trades_page_limit)
    add_option(command, "--trades-max-offset", args.trades_max_offset)
    add_option(command, "--price-fidelity-minutes", args.price_fidelity_minutes)
    add_option(command, "--max-price-window-days", args.max_price_window_days)
    add_option(command, "--candidate-confidences", args.candidate_confidences)
    add_flag(command, "--skip-price-history", args.skip_price_history)
    add_flag(command, "--force-trades-refresh", args.force_trades_refresh)
    add_flag(command, "--force-market-refresh", args.force_market_refresh)
    add_flag(command, "--force-price-refresh", args.force_price_refresh)
    add_flag(command, "--no-seed-from-candidates", args.no_seed_from_candidates)
    return command


def update_markets_command(args: argparse.Namespace) -> List[str]:
    command = [sys.executable, script("discover_markets.py")]
    add_option(command, "--data-dir", args.data_dir)
    add_repeated(command, "--market", args.market)
    add_option(command, "--markets-file", args.markets_file)
    add_option(command, "--limit-pages", args.limit_pages)
    add_option(command, "--page-size", args.page_size)
    add_option(command, "--min-volume", args.min_volume)
    add_option(command, "--min-liquidity", args.min_liquidity)
    add_option(command, "--max-markets", args.max_markets)
    add_option(command, "--price-fidelity-minutes", args.price_fidelity_minutes)
    add_option(command, "--price-window-days", args.price_window_days)
    add_option(command, "--sleep", args.sleep)
    add_flag(command, "--closed", args.closed)
    add_flag(command, "--skip-price-history", args.skip_price_history)
    add_flag(command, "--force-market-refresh", args.force_market_refresh)
    add_flag(command, "--force-price-refresh", args.force_price_refresh)
    return command


def update_market_trades_command(args: argparse.Namespace) -> List[str]:
    command = [sys.executable, script("market_scanner.py"), "--cache-trades-only", "--allow-api"]
    add_option(command, "--data-dir", args.data_dir)
    add_option(command, "--max-markets", args.max_markets)
    add_repeated(command, "--market", args.market)
    add_option(command, "--markets-file", args.markets_file)
    add_option(command, "--sleep", args.sleep)
    add_option(command, "--trades-page-limit", args.trades_page_limit)
    add_option(command, "--trades-max-offset", args.trades_max_offset)
    add_flag(command, "--refresh-market-trades", args.refresh_market_trades)
    return command


def analyze_luck_command(args: argparse.Namespace) -> List[str]:
    command = [sys.executable, script("luck_skill_analysis.py")]
    add_option(command, "--data-dir", args.data_dir)
    add_option(command, "--reports-dir", args.reports_dir)
    add_repeated(command, "--user", args.user)
    add_repeated(command, "--users-file", args.users_file)
    add_option(command, "--sleep", args.sleep)
    add_flag(command, "--hydrate-missing-users", args.hydrate_missing_users)
    add_flag(command, "--use-price-history", args.use_price_history)
    return command


def analyze_scanner_command(args: argparse.Namespace) -> List[str]:
    command = [sys.executable, script("market_scanner.py")]
    add_option(command, "--data-dir", args.data_dir)
    add_option(command, "--out-dir", args.out_dir)
    add_option(command, "--lookback-windows", args.lookback_windows)
    add_option(command, "--price-window", args.price_window)
    add_option(command, "--tau", args.tau)
    add_option(command, "--min-abs-move", args.min_abs_move)
    add_option(command, "--jump-sigma-mult", args.jump_sigma_mult)
    add_option(command, "--volume-spike-mult", args.volume_spike_mult)
    add_option(command, "--min-history-points", args.min_history_points)
    add_option(command, "--max-markets", args.max_markets)
    add_repeated(command, "--market", args.market)
    add_option(command, "--markets-file", args.markets_file)
    add_option(command, "--event-cluster-window", args.event_cluster_window)
    add_option(command, "--min-independent-events", args.min_independent_events)
    add_option(command, "--max-single-event-share", args.max_single_event_share)
    add_option(command, "--min-directional-ratio", args.min_directional_ratio)
    add_option(command, "--sleep", args.sleep)
    add_option(command, "--trades-page-limit", args.trades_page_limit)
    add_option(command, "--trades-max-offset", args.trades_max_offset)
    add_flag(command, "--allow-api", args.allow_api)
    add_flag(command, "--refresh-market-trades", args.refresh_market_trades)
    return command


def scanner_analysis_command(
    *,
    data_dir: str,
    market: Iterable[str] | None = None,
    markets_file: str | None = None,
    out_dir: str = "reports/market_scanner",
) -> List[str]:
    command = [sys.executable, script("market_scanner.py")]
    add_option(command, "--data-dir", data_dir)
    add_option(command, "--out-dir", out_dir)
    add_repeated(command, "--market", market)
    add_option(command, "--markets-file", markets_file)
    return command


def cmd_status(args: argparse.Namespace) -> int:
    inventory = build_inventory(args.data_dir, args.reports_dir, root=ROOT, cache_seconds=0 if args.refresh else 60)
    runs = list_latest_runs(ROOT)
    jobs = list_jobs(ROOT, limit=8)
    watermarks = load_watermarks(args.data_dir)

    print("Inventory")
    print(f"  users: {inventory.get('user_count', 0)}")
    print(f"  market metadata rows: {inventory.get('market_metadata_rows', 0)}")
    print(f"  price history assets: {inventory.get('price_history_assets', 0)}")
    print(f"  market trade sets: {inventory.get('market_trade_sets', 0)}")
    print(f"  generated: {inventory.get('generated_at', 'unknown')}")

    print("\nLatest runs")
    if not runs:
        print("  none recorded")
    for run in runs[:10]:
        duration = run.get("duration_seconds")
        duration_text = f"{duration:.1f}s" if isinstance(duration, (int, float)) else (
            "running" if run.get("status") == "running" else "n/a"
        )
        print(f"  {run.get('name')}: {run.get('status')} at {run.get('started_at')} ({duration_text})")

    print("\nRecent jobs")
    if not jobs:
        print("  none recorded")
    for job in jobs:
        print(f"  {job.get('label')}: {job.get('status')} at {job.get('created_at')}")

    if watermarks:
        print("\nWatermarks")
        for namespace, values in sorted(watermarks.items()):
            count = len(values) if isinstance(values, dict) else 0
            print(f"  {namespace}: {count} record(s)")
    return 0


def cmd_db_status(args: argparse.Namespace) -> int:
    from sqlite_store import database_summary, default_db_path, ensure_database

    db_path = args.db_path or str(default_db_path(args.data_dir))
    ensure_database(args.data_dir, db_path)
    summary = database_summary(db_path)
    print(f"SQLite database: {summary['db_path']}")
    if not summary.get("exists"):
        print("  not built")
        return 1
    meta = summary.get("meta", {})
    if meta.get("built_at"):
        print(f"  built_at: {meta['built_at']}")
    for name, count in sorted(summary.get("counts", {}).items()):
        print(f"  {name}: {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Polymarket local data app: update local SQL data, run offline analysis, and serve dashboards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show database/report freshness and recent runs")
    status.add_argument("--data-dir", default="data")
    status.add_argument("--reports-dir", default="reports")
    status.add_argument("--refresh", action="store_true", help="Recompute inventory instead of using the saved snapshot")

    db = subparsers.add_parser("db", help="Inspect the local SQLite database")
    db_sub = db.add_subparsers(dest="target", required=True)

    db_status = db_sub.add_parser("status", help="Show SQLite table counts")
    db_status.add_argument("--data-dir", default="data")
    db_status.add_argument("--db-path")

    update = subparsers.add_parser("update", help="Run API-backed database updates")
    update_sub = update.add_subparsers(dest="target", required=True)

    users = update_sub.add_parser("users", help="Refresh user trades, market metadata, and price history")
    users.add_argument("--data-dir", default="data")
    users.add_argument("--user", action="append")
    users.add_argument("--users-file", action="append", help="Text file of user keys or wallets, one per line")
    users.add_argument("--sleep", type=float, default=0.05)
    users.add_argument("--timeout-seconds", type=int, default=30)
    users.add_argument("--max-retries", type=int, default=5)
    users.add_argument("--trades-page-limit", type=int, default=500)
    users.add_argument("--trades-max-offset", type=int, default=3000)
    users.add_argument("--price-fidelity-minutes", type=int, default=60)
    users.add_argument("--max-price-window-days", type=int, default=7)
    users.add_argument("--skip-price-history", action="store_true")
    users.add_argument("--force-trades-refresh", action="store_true")
    users.add_argument("--force-market-refresh", action="store_true")
    users.add_argument("--force-price-refresh", action="store_true")
    users.add_argument("--candidate-confidences", default="high,very_high")
    users.add_argument("--no-seed-from-candidates", action="store_true")
    users.add_argument("--analyze-after", action="store_true")
    users.add_argument("--analysis-reports-dir", default="reports/luck_skill")
    users.add_argument("--analysis-use-price-history", action="store_true")

    markets = update_sub.add_parser("markets", help="Discover/store high-signal markets and price history")
    markets.add_argument("--data-dir", default="data")
    markets.add_argument("--limit-pages", type=int, default=10)
    markets.add_argument("--page-size", type=int, default=500)
    markets.add_argument("--min-volume", type=float, default=25_000)
    markets.add_argument("--min-liquidity", type=float, default=1_000)
    markets.add_argument("--max-markets", type=int, default=250)
    markets.add_argument("--market", action="append")
    markets.add_argument("--markets-file")
    markets.add_argument("--closed", action="store_true")
    markets.add_argument("--price-fidelity-minutes", type=int, default=15)
    markets.add_argument("--price-window-days", type=int, default=90)
    markets.add_argument("--sleep", type=float, default=0.05)
    markets.add_argument("--skip-price-history", action="store_true")
    markets.add_argument("--force-market-refresh", action="store_true")
    markets.add_argument("--force-price-refresh", action="store_true")
    markets.add_argument("--analyze-after", action="store_true")
    markets.add_argument("--scanner-out-dir", default="reports/market_scanner")

    market_trades = update_sub.add_parser("market-trades", help="Refresh scanner market-trade rows")
    market_trades.add_argument("--data-dir", default="data")
    market_trades.add_argument("--max-markets", type=int, default=0)
    market_trades.add_argument("--market", action="append")
    market_trades.add_argument("--markets-file")
    market_trades.add_argument("--sleep", type=float, default=0.05)
    market_trades.add_argument("--trades-page-limit", type=int, default=1000)
    market_trades.add_argument("--trades-max-offset", type=int, default=3000)
    market_trades.add_argument("--refresh-market-trades", action="store_true")
    market_trades.add_argument("--analyze-after", action="store_true")
    market_trades.add_argument("--scanner-out-dir", default="reports/market_scanner")

    analyze = subparsers.add_parser("analyze", help="Run database-backed analysis")
    analyze_sub = analyze.add_subparsers(dest="target", required=True)

    luck = analyze_sub.add_parser("luck", help="Run luck/skill analysis from local SQL data")
    luck.add_argument("--data-dir", default="data")
    luck.add_argument("--reports-dir", default="reports/luck_skill")
    luck.add_argument("--user", action="append")
    luck.add_argument("--users-file", action="append", help="Text file of user keys, one per line")
    luck.add_argument("--hydrate-missing-users", action="store_true")
    luck.add_argument("--use-price-history", action="store_true")
    luck.add_argument("--sleep", type=float, default=0.05)

    scanner = analyze_sub.add_parser("scanner", help="Run market scanner from local SQL data by default")
    scanner.add_argument("--data-dir", default="data")
    scanner.add_argument("--out-dir", default="reports/market_scanner")
    scanner.add_argument("--lookback-windows", default="15m,1h,6h,24h,7d")
    scanner.add_argument("--price-window", default="30m")
    scanner.add_argument("--tau", default="1h")
    scanner.add_argument("--min-abs-move", type=float, default=0.10)
    scanner.add_argument("--jump-sigma-mult", type=float, default=2.5)
    scanner.add_argument("--volume-spike-mult", type=float, default=1.5)
    scanner.add_argument("--min-history-points", type=int, default=50)
    scanner.add_argument("--max-markets", type=int, default=0)
    scanner.add_argument("--market", action="append")
    scanner.add_argument("--markets-file")
    scanner.add_argument("--event-cluster-window", default="24h")
    scanner.add_argument("--min-independent-events", type=int, default=2)
    scanner.add_argument("--max-single-event-share", type=float, default=0.65)
    scanner.add_argument("--min-directional-ratio", type=float, default=0.10)
    scanner.add_argument("--sleep", type=float, default=0.05)
    scanner.add_argument("--trades-page-limit", type=int, default=1000)
    scanner.add_argument("--trades-max-offset", type=int, default=3000)
    scanner.add_argument("--allow-api", action="store_true")
    scanner.add_argument("--refresh-market-trades", action="store_true")

    dashboard = subparsers.add_parser("dashboard", help="Build or serve dashboards")
    dashboard_sub = dashboard.add_subparsers(dest="target", required=True)

    serve = dashboard_sub.add_parser("serve", help="Serve a local dashboard/job control panel")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--data-dir", default="data")
    serve.add_argument("--reports-dir", default="reports")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        return cmd_status(args)

    if args.command == "db":
        if args.target == "status":
            return cmd_db_status(args)

    if args.command == "update":
        if args.target == "users":
            command = update_users_command(args)
            commands = [command]
            if args.analyze_after:
                analyze_command = [
                    sys.executable,
                    script("luck_skill_analysis.py"),
                    "--data-dir",
                    args.data_dir,
                    "--reports-dir",
                    args.analysis_reports_dir,
                ]
                add_repeated(analyze_command, "--user", args.user)
                add_repeated(analyze_command, "--users-file", args.users_file)
                add_option(analyze_command, "--sleep", args.sleep)
                add_flag(analyze_command, "--use-price-history", args.analysis_use_price_history)
                commands.append(analyze_command)
            return run_steps(
                name="update_users",
                kind="update",
                commands=commands,
                metadata={"data_dir": args.data_dir, "analyze_after": bool(args.analyze_after)},
            )
        if args.target == "markets":
            command = update_markets_command(args)
            commands = [command]
            if args.analyze_after:
                commands.append(
                    scanner_analysis_command(
                        data_dir=args.data_dir,
                        market=args.market,
                        markets_file=args.markets_file,
                        out_dir=args.scanner_out_dir,
                    )
                )
            return run_steps(
                name="update_markets",
                kind="update",
                commands=commands,
                metadata={"data_dir": args.data_dir, "analyze_after": bool(args.analyze_after)},
            )
        if args.target == "market-trades":
            command = update_market_trades_command(args)
            commands = [command]
            if args.analyze_after:
                commands.append(
                    scanner_analysis_command(
                        data_dir=args.data_dir,
                        market=args.market,
                        markets_file=args.markets_file,
                        out_dir=args.scanner_out_dir,
                    )
                )
            return run_steps(
                name="update_market_trades",
                kind="update",
                commands=commands,
                metadata={"data_dir": args.data_dir, "analyze_after": bool(args.analyze_after)},
            )

    if args.command == "analyze":
        if args.target == "luck":
            command = analyze_luck_command(args)
            return run_steps(name="analyze_luck", kind="analysis", commands=[command], metadata={"data_dir": args.data_dir})
        if args.target == "scanner":
            command = analyze_scanner_command(args)
            return run_steps(
                name="analyze_scanner",
                kind="analysis",
                commands=[command],
                metadata={"data_dir": args.data_dir, "allow_api": args.allow_api},
            )

    if args.command == "dashboard":
        if args.target == "serve":
            from dashboard_server import serve_dashboard

            serve_dashboard(host=args.host, port=args.port, data_dir=args.data_dir, reports_dir=args.reports_dir)
            return 0

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
