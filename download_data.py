from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

from api import ApiError, MetadataCache, PolymarketClient, fetch_paginated, resolve_user_to_wallet
from cli import write_csv
from config import CLOB_BASE, DATA_BASE, AnalyzerConfig
from utils import safe_float


USER_KEY_RE = re.compile(r"[^a-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Polymarket user trade data and related market data into monthly CSV partitions."
        )
    )
    parser.add_argument(
        "--user",
        action="append",
        help=(
            "Polymarket handle or proxy wallet. Repeatable. "
            "If omitted, users are discovered from subfolders under <data-dir>/user."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory for downloaded CSV files. Defaults to ./data",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--price-fidelity-minutes",
        type=int,
        default=60,
        help="Price-history fidelity in minutes for CLOB /prices-history calls.",
    )
    parser.add_argument(
        "--skip-price-history",
        action="store_true",
        help="Skip downloading CLOB price history rows.",
    )
    parser.add_argument(
        "--max-price-window-days",
        type=int,
        default=7,
        help="Maximum number of days per CLOB /prices-history request window.",
    )
    parser.add_argument(
        "--force-trades-refresh",
        action="store_true",
        help="Always fetch user trades from API even if cached trade files exist.",
    )
    parser.add_argument(
        "--force-market-refresh",
        action="store_true",
        help="Always fetch and rewrite market metadata files.",
    )
    parser.add_argument(
        "--force-price-refresh",
        action="store_true",
        help="Always refetch and rewrite price-history files.",
    )
    parser.add_argument(
        "--candidates-csv",
        default="reports/market_scanner/candidate_users.csv",
        help=(
            "Scanner output CSV used to auto-seed users when --user is omitted "
            "(default: reports/market_scanner/candidate_users.csv)"
        ),
    )
    parser.add_argument(
        "--candidate-confidences",
        default="high,very_high",
        help=(
            "Comma-separated confidence levels to auto-seed from candidates CSV "
            "(default: high,very_high)"
        ),
    )
    parser.add_argument(
        "--no-seed-from-candidates",
        action="store_true",
        help="Disable automatic seeding of data/user folders from scanner candidates CSV.",
    )
    return parser.parse_args()


def user_key(value: str) -> str:
    s = value.strip().lstrip("@").lower()
    s = USER_KEY_RE.sub("_", s)
    return s or "unknown_user"


def to_unix_seconds(ts: Any) -> float | None:
    value = safe_float(ts)
    if value is None:
        return None
    if value > 10_000_000_000:
        value /= 1000.0
    return value


def month_key_from_ts(ts: Any) -> str | None:
    unix_ts = to_unix_seconds(ts)
    if unix_ts is None:
        return None
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return dt.strftime("%Y-%m")


def csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=False)
    return value


def csv_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: csv_safe(value) for key, value in row.items()}


def sanitize_file_component(value: Any, fallback: str = "row") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = USER_KEY_RE.sub("_", text.lower())
    text = text.strip("._")
    if not text:
        text = fallback
    return text[:140]


def parse_confidence_levels(value: str) -> Set[str]:
    levels = {x.strip().lower() for x in str(value or "").split(",") if x.strip()}
    return levels or {"high", "very_high"}


def load_candidate_wallets(candidates_csv: Path, accepted_confidences: Set[str]) -> List[str]:
    if not candidates_csv.exists():
        return []

    wallets: List[str] = []
    seen: Set[str] = set()
    try:
        with candidates_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                confidence = str(row.get("confidence_level") or "").strip().lower()
                if confidence not in accepted_confidences:
                    continue
                wallet = str(row.get("wallet") or "").strip()
                if not wallet:
                    continue
                wallet_l = wallet.lower()
                if wallet_l in seen:
                    continue
                seen.add(wallet_l)
                wallets.append(wallet)
    except OSError as exc:
        print(f"Failed reading candidates CSV {candidates_csv}: {exc}", file=sys.stderr)
        return []

    return wallets


def seed_user_dirs_from_candidates(
    data_dir: Path,
    candidates_csv: Path,
    accepted_confidences: Set[str],
) -> Tuple[int, int]:
    wallets = load_candidate_wallets(candidates_csv, accepted_confidences)
    if not wallets:
        return 0, 0

    users_dir = data_dir / "user"
    users_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    existing = 0
    for wallet in wallets:
        folder = users_dir / sanitize_file_component(wallet, fallback="wallet")
        if folder.exists():
            existing += 1
            continue
        folder.mkdir(parents=True, exist_ok=True)
        created += 1

    return created, existing


def build_record_key(row: Dict[str, Any], key_fields: Sequence[str], idx: int) -> str:
    for field in key_fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return sanitize_file_component(value, fallback=f"{idx:08d}")
    return f"{idx:08d}"


def record_file_path(base_dir: Path, month: str, prefix: str, record_key: str) -> Path:
    return base_dir / month / f"{prefix}_{record_key}.csv"


def write_monthly_record_files(
    base_dir: Path,
    rows_by_month: Dict[str, List[Dict[str, Any]]],
    *,
    prefix: str,
    key_fields: Sequence[str],
    overwrite: bool,
) -> int:
    written = 0
    for month, rows in sorted(rows_by_month.items()):
        for idx, row in enumerate(rows, start=1):
            record_key = build_record_key(row, key_fields, idx)
            out_path = record_file_path(base_dir, month, prefix, record_key)
            if out_path.exists() and not overwrite:
                continue
            write_csv(out_path, [csv_safe_row(row)])
            written += 1
    return written


def trade_uid(trade: Dict[str, Any]) -> str:
    """Extract stable unique identifier for a trade."""
    tid = str(trade.get("id") or "").strip()
    if tid:
        return tid
    txhash = str(trade.get("transactionHash") or "").strip()
    if txhash:
        return txhash
    return ""


def load_existing_trade_uids(data_dir: Path, key: str) -> Set[str]:
    """Load trade UIDs from existing cached trade files."""
    base_dir = data_dir / "user" / key / "trades"
    if not base_dir.exists():
        return set()

    uids: Set[str] = set()
    for month_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        for csv_file in sorted(month_dir.glob("trade_*.csv")):
            try:
                with csv_file.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        uid = trade_uid(row)
                        if uid:
                            uids.add(uid)
            except OSError:
                continue
    return uids


def month_bounds_utc(month_key: str) -> Tuple[int, int]:
    dt = datetime.strptime(month_key, "%Y-%m").replace(tzinfo=timezone.utc)
    year = dt.year
    month = dt.month
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    start = int(dt.timestamp())
    end = int(next_month.timestamp()) - 1
    return start, end


def iter_time_windows(start_ts: float, end_ts: float, window_days: int) -> Iterable[Tuple[int, int]]:
    window_seconds = max(1, window_days) * 86400
    start = int(start_ts)
    end = int(end_ts)
    cur = start
    while cur <= end:
        window_end = min(end, cur + window_seconds - 1)
        yield cur, window_end
        cur = window_end + 1


def is_interval_too_long_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "startts" in text and "endts" in text and "too long" in text


def fetch_price_history_adaptive(
    client: PolymarketClient,
    *,
    asset: str,
    start_ts: int,
    end_ts: int,
    fidelity_minutes: int,
    min_window_seconds: int = 6 * 3600,
) -> List[Dict[str, Any]]:
    """Fetch price history while shrinking request windows when CLOB rejects long intervals."""
    windows: List[Tuple[int, int]] = [(start_ts, end_ts)]
    rows: List[Dict[str, Any]] = []

    while windows:
        window_start, window_end = windows.pop(0)
        try:
            data = client.get_json(
                CLOB_BASE,
                "/prices-history",
                {
                    "market": asset,
                    "startTs": window_start,
                    "endTs": window_end,
                    "interval": "all",
                    "fidelity": fidelity_minutes,
                },
            )
        except ApiError as exc:
            span = window_end - window_start + 1
            if is_interval_too_long_error(exc) and span > min_window_seconds:
                mid = window_start + (span // 2)
                left = (window_start, max(window_start, mid - 1))
                right = (mid, window_end)
                windows = [left, right, *windows]
                continue
            raise

        history = data.get("history") if isinstance(data, dict) else []
        for point in history or []:
            if isinstance(point, dict):
                rows.append(point)

    return rows


def flatten_market_row(market: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "conditionId": market.get("conditionId"),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "eventSlug": market.get("eventSlug"),
        "closed": market.get("closed"),
        "active": market.get("active"),
        "closedTime": market.get("closedTime"),
        "endDate": market.get("endDate"),
        "winner": market.get("winner"),
        "winningOutcomeIndex": market.get("winningOutcomeIndex"),
        "outcomes": market.get("outcomes"),
        "outcomePrices": market.get("outcomePrices"),
        "clobTokenIds": market.get("clobTokenIds"),
        "liquidityNum": market.get("liquidityNum"),
        "volumeNum": market.get("volumeNum"),
        "volume24hr": market.get("volume24hr"),
        "umaResolutionStatus": market.get("umaResolutionStatus"),
    }


def iter_trade_months(trades: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for trade in trades:
        month = month_key_from_ts(trade.get("timestamp"))
        if month is None:
            continue
        yield month, trade


def download_for_user(
    user: str,
    cfg: AnalyzerConfig,
    data_dir: Path,
    *,
    skip_price_history: bool,
    price_fidelity_minutes: int,
    max_price_window_days: int,
    force_trades_refresh: bool,
    force_market_refresh: bool,
    force_price_refresh: bool,
) -> None:
    client = PolymarketClient(cfg)
    cache = MetadataCache()

    wallet, profile = resolve_user_to_wallet(client, user)
    key = user_key(user)

    print(f"Resolved {user} to wallet {wallet}", file=sys.stderr)

    trades = fetch_paginated(
        client,
        DATA_BASE,
        "/trades",
        {
            "user": wallet,
            "takerOnly": False,
        },
        limit=cfg.trades_page_limit,
        max_offset=cfg.trades_max_offset,
        progress_label=f"trades for {wallet}",
    )
    print(f"Fetched {len(trades)} trades from API", file=sys.stderr)

    existing_trade_uids = load_existing_trade_uids(data_dir, key)
    if existing_trade_uids:
        print(f"Found {len(existing_trade_uids)} existing cached trade UIDs", file=sys.stderr)

    new_trades = [t for t in trades if trade_uid(t) not in existing_trade_uids]
    if new_trades:
        print(f"Identified {len(new_trades)} new trades to write", file=sys.stderr)
    else:
        print("No new trades found; all trades already cached", file=sys.stderr)

    user_trade_rows_by_month: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    condition_months: DefaultDict[str, set[str]] = defaultdict(set)
    asset_time_ranges: Dict[str, Tuple[float, float]] = {}
    asset_trade_months: DefaultDict[str, Set[str]] = defaultdict(set)
    new_trade_uids_set = {trade_uid(t) for t in new_trades if trade_uid(t)}

    for trade_index, (month, trade) in enumerate(iter_trade_months(trades), start=1):
        is_new = trade_uid(trade) in new_trade_uids_set
        if is_new:
            trade_row = dict(trade)
            trade_row["input_user"] = user
            trade_row["user_key"] = key
            trade_row["wallet"] = wallet
            trade_row["profile_name"] = profile.get("name") or ""
            trade_row["profile_pseudonym"] = profile.get("pseudonym") or ""
            trade_row["trade_record_key"] = (
                str(trade.get("id") or "")
                or str(trade.get("transactionHash") or "")
                or f"trade_{trade_index:08d}"
            )
            user_trade_rows_by_month[month].append(trade_row)

        condition_id = str(trade.get("conditionId") or "")
        if condition_id:
            condition_months[condition_id].add(month)

        asset = str(trade.get("asset") or "")
        ts = to_unix_seconds(trade.get("timestamp"))
        if asset and ts is not None:
            asset_trade_months[asset].add(month)
            if asset not in asset_time_ranges:
                asset_time_ranges[asset] = (ts, ts)
            else:
                start_ts, end_ts = asset_time_ranges[asset]
                asset_time_ranges[asset] = (min(start_ts, ts), max(end_ts, ts))

    user_profile_path = data_dir / "user" / key / "profile.csv"
    write_csv(
        user_profile_path,
        [
            {
                "input_user": user,
                "user_key": key,
                "wallet": wallet,
                "profile_name": profile.get("name") or "",
                "profile_pseudonym": profile.get("pseudonym") or "",
            }
        ],
    )

    trade_files_written = write_monthly_record_files(
        data_dir / "user" / key / "trades",
        dict(user_trade_rows_by_month),
        prefix="trade",
        key_fields=("trade_record_key", "id", "transactionHash"),
        overwrite=force_trades_refresh,
    )
    print(f"Wrote {trade_files_written} new trade files", file=sys.stderr)

    condition_ids = sorted(condition_months.keys())
    markets_dir = data_dir / "market" / "markets"
    if force_market_refresh:
        condition_ids_to_fetch = condition_ids
    else:
        condition_ids_to_fetch = []
        for condition_id in condition_ids:
            months = condition_months.get(condition_id) or set()
            record_key = sanitize_file_component(condition_id)
            needs_fetch = False
            for month in months:
                out_path = record_file_path(markets_dir, month, "market", record_key)
                if not out_path.exists():
                    needs_fetch = True
                    break
            if needs_fetch:
                condition_ids_to_fetch.append(condition_id)

    markets: Dict[str, Dict[str, Any]] = {}
    if condition_ids_to_fetch:
        markets = cache.fetch_markets_for_conditions(client, condition_ids_to_fetch)
        print(
            f"Fetched metadata for {len(markets)} markets ({len(condition_ids_to_fetch)} requested)",
            file=sys.stderr,
        )
    else:
        print("Skipped market metadata API calls (all market files already present)", file=sys.stderr)

    market_rows_by_month: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for condition_id in condition_ids_to_fetch:
        market = markets.get(condition_id)
        if not market:
            continue

        market_row = flatten_market_row(market)
        market_row["conditionId"] = condition_id
        for month in sorted(condition_months.get(condition_id) or []):
            market_rows_by_month[month].append(market_row)

    market_files_written = write_monthly_record_files(
        markets_dir,
        dict(market_rows_by_month),
        prefix="market",
        key_fields=("conditionId", "slug", "eventSlug"),
        overwrite=force_market_refresh,
    )
    if market_files_written:
        print(f"Wrote {market_files_written} market files", file=sys.stderr)

    if skip_price_history:
        print("Skipped price history download (--skip-price-history).", file=sys.stderr)
        return

    asset_to_condition: Dict[str, str] = {}
    for trade in trades:
        asset = str(trade.get("asset") or "")
        condition_id = str(trade.get("conditionId") or "")
        if asset and condition_id and asset not in asset_to_condition:
            asset_to_condition[asset] = condition_id

    price_rows_by_month_asset: DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    price_history_root = data_dir / "market" / "prices-history"
    assets = sorted(asset_time_ranges.keys())
    fetched_assets = 0
    skipped_assets = 0

    for i, asset in enumerate(assets, start=1):
        traded_months = sorted(asset_trade_months.get(asset) or [])
        if not traded_months:
            continue

        missing_months: List[str] = []
        for month in traded_months:
            file_name = f"price_history_asset_{sanitize_file_component(asset, 'asset')}.csv"
            out_path = price_history_root / month / file_name
            if force_price_refresh or not out_path.exists():
                missing_months.append(month)

        if not missing_months:
            skipped_assets += 1
            if i % 25 == 0 or i == len(assets):
                print(
                    f"Price history progress {i}/{len(assets)} assets (skipped {skipped_assets}, fetched {fetched_assets})",
                    file=sys.stderr,
                )
            continue

        window_starts_ends = [month_bounds_utc(month) for month in missing_months]
        query_start = min(start for start, _ in window_starts_ends)
        query_end = max(end for _, end in window_starts_ends)

        by_timestamp: Dict[int, float] = {}
        failed_asset = False
        for window_start, window_end in iter_time_windows(query_start, query_end, max_price_window_days):
            try:
                history = fetch_price_history_adaptive(
                    client,
                    asset=asset,
                    start_ts=window_start,
                    end_ts=window_end,
                    fidelity_minutes=price_fidelity_minutes,
                )
            except Exception as exc:
                print(
                    (
                        f"Price history fetch failed for asset {asset} "
                        f"window [{window_start}, {window_end}]: {exc}"
                    ),
                    file=sys.stderr,
                )
                failed_asset = True
                break

            for point in history or []:
                if not isinstance(point, dict):
                    continue
                ts = to_unix_seconds(point.get("t"))
                price = safe_float(point.get("p"))
                if ts is None or price is None:
                    continue
                by_timestamp[int(ts)] = price

        if failed_asset:
            continue

        fetched_assets += 1
        condition_id = asset_to_condition.get(asset, "")
        missing_month_set = set(missing_months)

        for ts in sorted(by_timestamp.keys()):
            month = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
            if month not in missing_month_set and not force_price_refresh:
                continue
            price_rows_by_month_asset[month][asset].append(
                {
                    "asset": asset,
                    "conditionId": condition_id,
                    "timestamp": ts,
                    "timestamp_utc": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "price": by_timestamp[ts],
                    "source": "clob_prices_history",
                }
            )

        if i % 25 == 0 or i == len(assets):
            print(
                f"Price history progress {i}/{len(assets)} assets (skipped {skipped_assets}, fetched {fetched_assets})",
                file=sys.stderr,
            )

    price_files_written = 0
    for month, rows_by_asset in sorted(price_rows_by_month_asset.items()):
        month_dir = data_dir / "market" / "prices-history" / month
        for asset, rows in sorted(rows_by_asset.items()):
            out_path = month_dir / f"price_history_asset_{sanitize_file_component(asset, 'asset')}.csv"
            if out_path.exists() and not force_price_refresh:
                continue
            write_csv(out_path, [csv_safe_row(r) for r in rows])
            price_files_written += 1

    print(
        f"Wrote {price_files_written} price-history files; skipped {skipped_assets} cached assets",
        file=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    cfg = AnalyzerConfig(sleep_between_requests=args.sleep)
    data_dir = Path(args.data_dir)

    if not args.user and not args.no_seed_from_candidates:
        accepted_confidences = parse_confidence_levels(args.candidate_confidences)
        created, existing = seed_user_dirs_from_candidates(
            data_dir,
            Path(args.candidates_csv),
            accepted_confidences,
        )
        if created or existing:
            print(
                (
                    f"Auto-seeded users from {args.candidates_csv}: "
                    f"created={created}, already_present={existing}, "
                    f"confidences={','.join(sorted(accepted_confidences))}"
                ),
                file=sys.stderr,
            )
        else:
            print(
                (
                    f"No users auto-seeded from {args.candidates_csv}. "
                    "Run market_scanner.py first or adjust --candidate-confidences."
                ),
                file=sys.stderr,
            )

    if args.user:
        users = [u.strip() for u in args.user if str(u).strip()]
    else:
        users_dir = data_dir / "user"
        if not users_dir.exists():
            print(
                (
                    "ERROR: No --user provided and no users discovered under "
                    f"{users_dir}. Create user folders there or pass --user."
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        users = sorted([p.name for p in users_dir.iterdir() if p.is_dir()])

    if not users:
        print("ERROR: No users to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(users)} user(s): {', '.join(users)}", file=sys.stderr)
    failures = 0
    for user in users:
        try:
            download_for_user(
                user,
                cfg,
                data_dir,
                skip_price_history=args.skip_price_history,
                price_fidelity_minutes=args.price_fidelity_minutes,
                max_price_window_days=args.max_price_window_days,
                force_trades_refresh=args.force_trades_refresh,
                force_market_refresh=args.force_market_refresh,
                force_price_refresh=args.force_price_refresh,
            )
        except Exception as exc:
            failures += 1
            print(f"ERROR processing user {user}: {exc}", file=sys.stderr)

    if failures:
        print(f"Done with {failures} failure(s). Wrote data under: {data_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote data under: {data_dir.resolve()}")


if __name__ == "__main__":
    main()
