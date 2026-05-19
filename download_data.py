from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

from api import ApiError, MetadataCache, PolymarketClient, resolve_user_to_wallet
from config import CLOB_BASE, DATA_BASE, AnalyzerConfig
from runtime_state import update_watermark
from sqlite_store import (
    candidate_wallets as sqlite_candidate_wallets,
    ensure_database,
    list_user_keys as sqlite_list_user_keys,
    load_all_user_trades as sqlite_load_all_user_trades,
    load_user_trade_uids as sqlite_load_user_trade_uids,
    load_user_profile_row as sqlite_load_user_profile_row,
    market_condition_ids as sqlite_market_condition_ids,
    price_history_cached as sqlite_price_history_cached,
    upsert_user_alias as sqlite_upsert_user_alias,
    upsert_markets as sqlite_upsert_markets,
    upsert_price_history as sqlite_upsert_price_history,
    upsert_user_profile as sqlite_upsert_user_profile,
    upsert_user_trades as sqlite_upsert_user_trades,
)
from utils import safe_float


USER_KEY_RE = re.compile(r"[^a-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Polymarket user trade data and related market data into SQLite."
        )
    )
    parser.add_argument(
        "--user",
        action="append",
        help=(
            "Polymarket handle or proxy wallet. Repeatable. "
            "If omitted, users are selected from the local database."
        ),
    )
    parser.add_argument(
        "--users-file",
        action="append",
        help="Text file containing Polymarket handles or proxy wallets, one per line.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory for the SQLite database. Defaults to ./data",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="HTTP timeout per API request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum HTTP retry attempts before a request is treated as failed.",
    )
    parser.add_argument(
        "--trades-page-limit",
        type=int,
        default=500,
        help="Number of user trades requested per /trades page.",
    )
    parser.add_argument(
        "--trades-max-offset",
        type=int,
        default=3_000,
        help="Maximum /trades pagination offset to fetch per user.",
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
        help="Always fetch user trades from API even if rows already exist.",
    )
    parser.add_argument(
        "--force-market-refresh",
        action="store_true",
        help="Always fetch and rewrite market metadata rows.",
    )
    parser.add_argument(
        "--force-price-refresh",
        action="store_true",
        help="Always refetch and rewrite price-history rows.",
    )
    parser.add_argument(
        "--candidate-confidences",
        default="high,very_high",
        help="Comma-separated confidence levels to auto-select from scanner candidates.",
    )
    parser.add_argument(
        "--no-seed-from-candidates",
        action="store_true",
        help="Disable automatic selection from scanner candidates.",
    )
    return parser.parse_args()


def user_key(value: str) -> str:
    s = value.strip().lstrip("@").lower()
    s = USER_KEY_RE.sub("_", s)
    return s or "unknown_user"


def wallet_like(value: Any) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", str(value or "").strip()))


def non_wallet_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if wallet_like(text) else text


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


def parse_confidence_levels(value: str) -> Set[str]:
    levels = {x.strip().lower() for x in str(value or "").split(",") if x.strip()}
    return levels or {"high", "very_high"}


def load_users_file(path: Path) -> List[str]:
    users: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value and not value.startswith("#"):
                    users.append(value)
    except OSError as exc:
        print(f"Failed reading users file {path}: {exc}", file=sys.stderr)
    return users


def collect_requested_users(users: Sequence[str] | None, users_files: Sequence[str] | None) -> List[str]:
    requested: List[str] = []
    seen: Set[str] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        requested.append(text)

    for user in users or []:
        add(user)
    for users_file in users_files or []:
        for user in load_users_file(Path(users_file)):
            add(user)
    return requested


def trade_uid(trade: Dict[str, Any]) -> str:
    """Extract stable unique identifier for a trade."""
    tid = str(trade.get("id") or "").strip()
    if tid:
        return tid
    txhash = str(trade.get("transactionHash") or "").strip()
    if txhash:
        return txhash
    pieces = [
        str(trade.get("conditionId") or ""),
        str(trade.get("asset") or ""),
        str(trade.get("side") or ""),
        str(trade.get("timestamp") or ""),
        str(trade.get("price") or ""),
        str(trade.get("size") or ""),
    ]
    fallback = "|".join(pieces).strip("|")
    return fallback


def load_existing_trade_uids(data_dir: Path, key: str) -> Set[str]:
    """Load trade UIDs from SQLite."""
    return sqlite_load_user_trade_uids(data_dir, key)


def load_stored_trade_rows(data_dir: Path, key: str) -> List[Dict[str, Any]]:
    return sqlite_load_all_user_trades(data_dir, key)


def transient_trade_error(exc: ApiError) -> bool:
    message = str(exc).lower()
    return (
        "request failed for" in message
        or "connection" in message
        or "timeout" in message
        or "timed out" in message
        or "failed after retries: http 429" in message
        or bool(re.search(r"failed after retries: http 5\d\d", message))
    )


def connection_reset_hint(exc: Exception) -> str:
    message = str(exc).lower()
    if (
        "connectionreseterror" in message
        or "connection was reset" in message
        or "forcibly closed by the remote host" in message
        or "recv failure: connection was reset" in message
    ):
        return (
            "Polymarket reset the HTTPS connection before returning an HTTP response. "
            "This is usually an upstream/network/IP restriction or temporary edge failure; "
            "retry later or from a different network if it persists."
        )
    return ""


def fetch_trade_page_resilient(
    client: PolymarketClient,
    wallet: str,
    cfg: AnalyzerConfig,
    *,
    page_limit: int,
    offset: int,
) -> Tuple[List[Dict[str, Any]], int]:
    min_page_limit = max(25, min(100, page_limit))
    while True:
        retry_count = cfg.max_retries if page_limit <= min_page_limit else min(2, cfg.max_retries)
        try:
            batch = client.get_json(
                DATA_BASE,
                "/trades",
                {
                    "user": wallet,
                    "takerOnly": False,
                    "limit": page_limit,
                    "offset": offset,
                },
                max_retries=retry_count,
            )
        except ApiError as exc:
            if page_limit > min_page_limit and transient_trade_error(exc):
                next_limit = max(min_page_limit, page_limit // 2)
                print(
                    (
                        f"    trades for {wallet}: request failed at offset {offset}; "
                        f"retrying with page size {next_limit} instead of {page_limit}"
                    ),
                    file=sys.stderr,
                )
                page_limit = next_limit
                continue
            raise
        if not isinstance(batch, list):
            raise ApiError(f"Expected list from /trades, got {type(batch).__name__}")
        return batch, page_limit


def fetch_user_trades_full(
    client: PolymarketClient,
    wallet: str,
    cfg: AnalyzerConfig,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    page_limit = max(1, cfg.trades_page_limit)

    while offset <= cfg.trades_max_offset:
        try:
            batch, page_limit = fetch_trade_page_resilient(
                client,
                wallet,
                cfg,
                page_limit=page_limit,
                offset=offset,
            )
        except ApiError as exc:
            message = str(exc)
            if "max historical activity offset" in message:
                print(
                    f"Pagination cap reached for /trades at offset {offset}. Retrieved {len(rows)} rows. "
                    "The Polymarket API does not expose older rows beyond this limit.",
                    file=sys.stderr,
                )
                return rows
            raise

        rows.extend(batch)
        page_number = (offset // page_limit) + 1 if page_limit > 0 else 1
        print(
            f"    trades for {wallet}: page {page_number} returned {len(batch)} rows "
            f"({len(rows)} total so far, page_size={page_limit})",
            file=sys.stderr,
        )
        if len(batch) < page_limit:
            return rows

        offset += page_limit

    print(
        f"Pagination cap reached for /trades. Retrieved {len(rows)} rows. "
        "A very active user may need archival/on-chain data for completeness.",
        file=sys.stderr,
    )
    return rows


def fetch_user_trades_incremental(
    client: PolymarketClient,
    wallet: str,
    cfg: AnalyzerConfig,
    *,
    seen_uids: Set[str],
) -> List[Dict[str, Any]]:
    """Fetch newest trade pages until a page contains no unseen trades."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    page_limit = max(1, cfg.trades_page_limit)

    while offset <= cfg.trades_max_offset:
        batch, page_limit = fetch_trade_page_resilient(
            client,
            wallet,
            cfg,
            page_limit=page_limit,
            offset=offset,
        )
        if not batch:
            break

        unseen_in_batch = 0
        for row in batch:
            if not isinstance(row, dict):
                continue
            uid = trade_uid(row)
            if uid and uid in seen_uids:
                continue
            rows.append(row)
            if uid:
                seen_uids.add(uid)
            unseen_in_batch += 1

        page_number = (offset // page_limit) + 1 if page_limit > 0 else 1
        print(
            f"    trades for {wallet}: page {page_number} unseen={unseen_in_batch} page_size={page_limit}",
            file=sys.stderr,
        )

        if unseen_in_batch == 0 or len(batch) < page_limit:
            break
        offset += page_limit

    rows.sort(key=lambda r: to_unix_seconds(r.get("timestamp")) or -1.0)
    return rows


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
    event_slug = market.get("eventSlug")
    if not event_slug:
        for event in market.get("events") or []:
            if isinstance(event, dict) and event.get("slug"):
                event_slug = event.get("slug")
                break
    return {
        "conditionId": market.get("conditionId"),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "eventSlug": event_slug,
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
    key = user_key(wallet or user)
    existing_profile = sqlite_load_user_profile_row(data_dir, key)
    profile_name = non_wallet_text(profile.get("name")) or non_wallet_text(existing_profile.get("name"))
    profile_pseudonym = non_wallet_text(profile.get("pseudonym")) or non_wallet_text(existing_profile.get("pseudonym"))
    profile_display = profile_name or profile_pseudonym

    print(f"Resolved {user} to wallet {wallet}", file=sys.stderr)
    sqlite_upsert_user_alias(
        data_dir,
        user,
        key,
        display_name=profile_display or ("" if wallet_like(user) else user),
        source="user_update",
    )
    sqlite_upsert_user_profile(
        data_dir,
        key,
        {
            "input_user": user,
            "user_key": key,
            "wallet": wallet,
            "profile_name": profile_name,
            "profile_pseudonym": profile_pseudonym,
            "name": profile_name,
            "pseudonym": profile_pseudonym,
            "referral": non_wallet_text(profile.get("referral")) or non_wallet_text(existing_profile.get("referral")),
        },
    )

    existing_trade_uids = load_existing_trade_uids(data_dir, key)
    if existing_trade_uids:
        print(f"Found {len(existing_trade_uids)} existing stored trade UIDs", file=sys.stderr)

    if force_trades_refresh or not existing_trade_uids:
        trades = fetch_user_trades_full(
            client,
            wallet,
            cfg,
        )
        print(f"Fetched {len(trades)} trades from API", file=sys.stderr)
    else:
        trades = fetch_user_trades_incremental(
            client,
            wallet,
            cfg,
            seen_uids=set(existing_trade_uids),
        )
        print(f"Fetched {len(trades)} new/changed trades from API", file=sys.stderr)

    new_trades = [t for t in trades if force_trades_refresh or trade_uid(t) not in existing_trade_uids]
    if new_trades:
        print(f"Identified {len(new_trades)} new trades to write", file=sys.stderr)
    else:
        print("No new trades found; all trades already stored", file=sys.stderr)

    trades_to_process = list(trades)
    if existing_trade_uids and (force_market_refresh or force_price_refresh) and not force_trades_refresh:
        stored_trades = load_stored_trade_rows(data_dir, key)
        stored_uids = {trade_uid(t) for t in stored_trades if trade_uid(t)}
        appended_new = [t for t in trades if trade_uid(t) not in stored_uids]
        trades_to_process = stored_trades + appended_new
        print(
            (
                f"Loaded {len(stored_trades)} stored trades for forced metadata/price refresh; "
                f"processing {len(trades_to_process)} trade rows"
            ),
            file=sys.stderr,
        )

    user_trade_rows_by_month: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    condition_months: DefaultDict[str, set[str]] = defaultdict(set)
    asset_time_ranges: Dict[str, Tuple[float, float]] = {}
    asset_trade_months: DefaultDict[str, Set[str]] = defaultdict(set)
    new_trade_uids_set = {trade_uid(t) for t in new_trades if trade_uid(t)}

    for trade_index, (month, trade) in enumerate(iter_trade_months(trades_to_process), start=1):
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

    sqlite_upsert_user_profile(
        data_dir,
        key,
        {
            "input_user": user,
            "user_key": key,
            "wallet": wallet,
            "profile_name": profile.get("name") or "",
            "profile_pseudonym": profile.get("pseudonym") or "",
        },
    )

    trade_rows_to_write = [row for rows in user_trade_rows_by_month.values() for row in rows]
    trade_rows_written = sqlite_upsert_user_trades(
        data_dir,
        key,
        trade_rows_to_write if not force_trades_refresh else trades_to_process,
        replace_user=force_trades_refresh,
    )
    print(f"Wrote {trade_rows_written} user trade rows to SQLite", file=sys.stderr)

    condition_ids = sorted(condition_months.keys())
    if force_market_refresh:
        condition_ids_to_fetch = condition_ids
    else:
        stored_market_ids = sqlite_market_condition_ids(data_dir)
        condition_ids_to_fetch = [condition_id for condition_id in condition_ids if condition_id not in stored_market_ids]

    markets: Dict[str, Dict[str, Any]] = {}
    if condition_ids_to_fetch:
        markets = cache.fetch_markets_for_conditions(client, condition_ids_to_fetch)
        print(
            f"Fetched metadata for {len(markets)} markets ({len(condition_ids_to_fetch)} requested)",
            file=sys.stderr,
        )
    else:
        print("Skipped market metadata API calls (all market rows already present)", file=sys.stderr)

    market_rows: List[Dict[str, Any]] = []
    for condition_id in condition_ids_to_fetch:
        market = markets.get(condition_id)
        if not market:
            continue

        market_row = flatten_market_row(market)
        market_row["conditionId"] = condition_id
        market_rows.append(market_row)

    market_rows_written = sqlite_upsert_markets(data_dir, market_rows)
    if market_rows_written:
        print(f"Wrote {market_rows_written} market rows to SQLite", file=sys.stderr)

    latest_trade_ts = max(
        (ts for ts in (to_unix_seconds(t.get("timestamp")) for t in trades_to_process) if ts is not None),
        default=None,
    )
    user_update_state: Dict[str, Any] = {
        "wallet": wallet,
        "profile_name": profile.get("name") or "",
        "profile_pseudonym": profile.get("pseudonym") or "",
        "trades_seen_in_last_update": len(trades),
        "trades_processed_for_related_data": len(trades_to_process),
        "new_trades_written": len(new_trades),
        "trade_rows_written": trade_rows_written,
        "market_rows_written": market_rows_written,
        "price_history_skipped": bool(skip_price_history),
        "force_trades_refresh": bool(force_trades_refresh),
        "force_market_refresh": bool(force_market_refresh),
        "force_price_refresh": bool(force_price_refresh),
    }
    if latest_trade_ts is not None:
        user_update_state["last_seen_trade_ts"] = int(latest_trade_ts)
        user_update_state["last_seen_trade_utc"] = datetime.fromtimestamp(
            latest_trade_ts,
            tz=timezone.utc,
        ).isoformat()
    update_watermark(data_dir, "users", key, user_update_state)

    if skip_price_history:
        print("Skipped price history download (--skip-price-history).", file=sys.stderr)
        return

    asset_to_condition: Dict[str, str] = {}
    for trade in trades_to_process:
        asset = str(trade.get("asset") or "")
        condition_id = str(trade.get("conditionId") or "")
        if asset and condition_id and asset not in asset_to_condition:
            asset_to_condition[asset] = condition_id

    price_rows_by_month_asset: DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    assets = sorted(asset_time_ranges.keys())
    fetched_assets = 0
    skipped_assets = 0

    for i, asset in enumerate(assets, start=1):
        traded_months = sorted(asset_trade_months.get(asset) or [])
        if not traded_months:
            continue

        missing_months: List[str] = []
        for month in traded_months:
            start_ts, end_ts = month_bounds_utc(month)
            if force_price_refresh or not sqlite_price_history_cached(data_dir, asset, start_ts, end_ts):
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

    price_rows_written = 0
    for _month, rows_by_asset in sorted(price_rows_by_month_asset.items()):
        for _asset, rows in sorted(rows_by_asset.items()):
            price_rows_written += sqlite_upsert_price_history(data_dir, rows)

    print(
        f"Wrote {price_rows_written} price-history rows; skipped {skipped_assets} stored assets",
        file=sys.stderr,
    )
    update_watermark(
        data_dir,
        "users",
        key,
        {
            "price_history_skipped": False,
            "price_rows_written": price_rows_written,
            "price_assets_fetched": fetched_assets,
            "price_assets_skipped_stored": skipped_assets,
        },
    )


def main() -> None:
    args = parse_args()
    cfg = AnalyzerConfig(
        sleep_between_requests=args.sleep,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        trades_page_limit=args.trades_page_limit,
        trades_max_offset=args.trades_max_offset,
    )
    data_dir = Path(args.data_dir)
    ensure_database(data_dir)
    requested_users = collect_requested_users(args.user, args.users_file)

    if not requested_users and not args.no_seed_from_candidates:
        accepted_confidences = parse_confidence_levels(args.candidate_confidences)
        requested_users = sqlite_candidate_wallets(data_dir, accepted_confidences)
        if requested_users:
            print(
                (
                    f"Auto-selected {len(requested_users)} candidate users from SQLite "
                    f"with confidences={','.join(sorted(accepted_confidences))}"
                ),
                file=sys.stderr,
            )
        else:
            print(
                (
                    "No users auto-selected from SQLite scanner candidates. "
                    "Run analyze scanner first or adjust --candidate-confidences."
                ),
                file=sys.stderr,
            )

    if requested_users:
        users = requested_users
    else:
        users = sqlite_list_user_keys(data_dir, include_candidates=True)

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
            hint = connection_reset_hint(exc)
            if hint:
                print(f"  Hint: {hint}", file=sys.stderr)

    if failures:
        print(f"Done with {failures} failure(s). Wrote data under: {data_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote data under: {data_dir.resolve()}")


if __name__ == "__main__":
    main()
