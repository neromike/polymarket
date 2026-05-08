from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from config import AnalyzerConfig
from utils import parse_duration_to_seconds, safe_float


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_users_from_file(path: str) -> List[str]:
    users: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            users.append(s)
    return users


def parse_args() -> Tuple[AnalyzerConfig, List[str]]:
    parser = argparse.ArgumentParser(description="Analyze Polymarket users for luck, skill, and copy candidates.")
    parser.add_argument("users", nargs="*", help="Polymarket handles or proxy wallets, for example magamyman")
    parser.add_argument("--users-file", help="Optional file with one handle or wallet per line")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds to sleep between API calls")
    parser.add_argument("--with-clv", action="store_true", help="Fetch price history for CLV style metrics")
    parser.add_argument(
        "--clv-horizons",
        default="1h,24h",
        help="Comma-separated CLV horizons, for example 1h,24h,7d",
    )
    parser.add_argument("--clv-fidelity-minutes", type=int, default=60)
    parser.add_argument("--max-clv-assets", type=int, default=250)
    parser.add_argument("--min-copy-shares", type=float, default=10.0)
    parser.add_argument("--min-copy-value", type=float, default=10.0)
    parser.add_argument("--min-copy-liquidity", type=float, default=100.0)
    parser.add_argument("--max-copy-price-worse-than-avg", type=float, default=0.05)
    parser.add_argument("--max-copy-spread", type=float, default=0.05)
    parser.add_argument("--recent-buy-days", type=float, default=30.0)
    args = parser.parse_args()

    users = [u.strip().lstrip("@") for u in args.users if u and str(u).strip()]
    if args.users_file:
        users.extend([u.strip().lstrip("@") for u in read_users_from_file(args.users_file)])

    horizons = tuple(parse_duration_to_seconds(x) for x in args.clv_horizons.split(",") if x.strip())
    cfg = AnalyzerConfig(
        out_dir="out",
        sleep_between_requests=args.sleep,
        with_clv=args.with_clv,
        clv_horizons_seconds=horizons or (3600, 86400),
        clv_fidelity_minutes=args.clv_fidelity_minutes,
        max_clv_assets_per_user=args.max_clv_assets,
        min_copy_shares=args.min_copy_shares,
        min_copy_value_usdc=args.min_copy_value,
        min_copy_liquidity_usdc=args.min_copy_liquidity,
        max_copy_price_worse_than_avg=args.max_copy_price_worse_than_avg,
        max_copy_spread=args.max_copy_spread,
        recent_buy_days=args.recent_buy_days,
    )
    return cfg, users


def print_summary(user_metrics: Sequence[Dict[str, Any]], copy_rows: Sequence[Dict[str, Any]]) -> None:
    print("\nUser metrics:")
    for row in sorted(user_metrics, key=lambda r: safe_float(r.get("analysis_score")) or -999, reverse=True):
        score = safe_float(row.get("analysis_score"))
        score_text = f"{score:.3f}" if score is not None else "NA"
        if row.get("error"):
            print(f"- {row.get('input_user')}: ERROR {row.get('error')}")
            continue
        print(
            f"- {row.get('input_user')}: "
            f"score={score_text} "
            f"luck_z={row.get('buy_probability_adjusted_luck_z')} "
            f"roi={row.get('closed_positions_roi')} "
            f"open_value={row.get('current_positions_value_usdc')}"
        )

    flagged = [r for r in copy_rows if str(r.get("copy_candidate")).lower() == "true"]
    print(f"\nCopy candidates flagged: {len(flagged)}")
    for row in flagged[:20]:
        copy_score = safe_float(row.get("copy_score"))
        copy_score_text = f"{copy_score:.2f}" if copy_score is not None else "NA"
        print(
            f"- {row.get('input_user')} | score={copy_score_text} | "
            f"{row.get('outcome')} | ask={row.get('best_ask_to_buy_now')} | "
            f"avg={row.get('user_avg_entry_price')} | {row.get('market_title')}"
        )
