from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from api import ApiError, MetadataCache, PolymarketClient
from cli import write_csv
from config import GAMMA_BASE, AnalyzerConfig
from download_data import fetch_price_history_adaptive, flatten_market_row, sanitize_file_component
from runtime_state import update_watermark
from utils import parse_jsonish_list, safe_float


HIGH_SIGNAL_TERMS = [
    "strike",
    "military",
    "war",
    "iran",
    "israel",
    "ceasefire",
    "peace deal",
    "sanction",
    "executive order",
    "pardon",
    "indict",
    "court",
    "supreme court",
    "ruling",
    "lawsuit",
    "investigation",
    "cabinet",
    "nomination",
    "resign",
    "fed",
    "fomc",
    "rate decision",
    "central bank",
    "regulation",
    "approval",
    "ban",
    "bill",
    "senate",
    "house vote",
    "listing",
    "merger",
    "acquisition",
]

LOW_SIGNAL_TERMS = [
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "ufc",
    "soccer",
    "tennis",
    "game",
    "match",
    "playoffs",
    "championship",
    "score",
    "spread",
]

CONDITION_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and cache high-value Polymarket markets for jump-first user scanning."
    )
    parser.add_argument("--data-dir", default="data", help="Workspace data directory (default: data)")
    parser.add_argument("--limit-pages", type=int, default=10, help="Max pages from /markets/keyset")
    parser.add_argument("--page-size", type=int, default=500, help="Rows per /markets/keyset page")
    parser.add_argument(
        "--min-volume",
        type=float,
        default=25_000,
        help="Minimum market volumeNum filter for discovery (default: 25000)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=1_000,
        help="Minimum market liquidityNum filter for discovery (default: 1000)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=250,
        help="Maximum selected markets to cache (default: 250)",
    )
    parser.add_argument(
        "--market",
        action="append",
        help="Specific conditionId to cache/update. Repeat to refresh more than one market.",
    )
    parser.add_argument(
        "--markets-file",
        help="Text file containing conditionIds to cache/update, one per line.",
    )
    parser.add_argument(
        "--closed",
        action="store_true",
        help="Discover closed markets (useful for retrospective runs)",
    )
    parser.add_argument(
        "--price-fidelity-minutes",
        type=int,
        default=15,
        help="CLOB /prices-history fidelity in minutes (default: 15)",
    )
    parser.add_argument(
        "--price-window-days",
        type=int,
        default=90,
        help="Lookback window for price history pull (default: 90)",
    )
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds to sleep between API calls")
    parser.add_argument(
        "--skip-price-history",
        action="store_true",
        help="Skip CLOB price-history caching",
    )
    parser.add_argument(
        "--force-market-refresh",
        action="store_true",
        help="Rewrite cached market metadata even when market cache file exists.",
    )
    parser.add_argument(
        "--force-price-refresh",
        action="store_true",
        help="Refetch and rewrite price-history files even when monthly cache files exist.",
    )
    return parser.parse_args()


def load_market_ids_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    market_path = Path(path)
    if not market_path.exists():
        raise SystemExit(f"Markets file not found: {market_path}")

    ids: List[str] = []
    for line in market_path.read_text(encoding="utf-8-sig").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        ids.append(token)
    return ids


def text_blob(market: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(market.get("question") or ""),
        str(market.get("description") or ""),
        str(market.get("resolutionSource") or ""),
        str(market.get("category") or ""),
        str(market.get("subcategory") or ""),
        str(market.get("slug") or ""),
    ]

    for tag in market.get("tags") or []:
        if isinstance(tag, dict):
            parts.append(str(tag.get("label") or ""))
            parts.append(str(tag.get("slug") or ""))

    for event in market.get("events") or []:
        if isinstance(event, dict):
            parts.append(str(event.get("title") or ""))
            parts.append(str(event.get("description") or ""))
            parts.append(str(event.get("category") or ""))

    return " ".join(parts).lower()


def has_valid_tokens(market: Dict[str, Any]) -> bool:
    condition_id = str(market.get("conditionId") or "")
    if not CONDITION_RE.match(condition_id):
        return False

    outcomes = parse_jsonish_list(market.get("outcomes"))
    token_ids = parse_jsonish_list(market.get("clobTokenIds"))
    return len(outcomes) >= 2 and len(token_ids) >= 2


def term_score(blob: str, terms: Iterable[str]) -> float:
    hits = sum(1 for term in terms if term in blob)
    return min(1.0, hits / 4.0)


def tradability_score(market: Dict[str, Any], min_volume: float, min_liquidity: float) -> float:
    volume = safe_float(market.get("volumeNum")) or safe_float(market.get("volume")) or 0.0
    liquidity = safe_float(market.get("liquidityNum")) or safe_float(market.get("liquidity")) or 0.0

    if volume < min_volume or liquidity < min_liquidity:
        return 0.0

    volume_part = min(1.0, math.log1p(volume) / math.log1p(max(1.0, min_volume * 20.0)))
    liquidity_part = min(1.0, math.log1p(liquidity) / math.log1p(max(1.0, min_liquidity * 20.0)))
    return 0.6 * volume_part + 0.4 * liquidity_part


def resolution_clarity_score(blob: str) -> float:
    good = ["official", "resolve", "resolution", "source", "announced", "reported", "certified"]
    bad = ["ambiguous", "subjective", "spirit", "intent", "unclear"]

    score = 0.5 + 0.1 * sum(1 for x in good if x in blob) - 0.15 * sum(1 for x in bad if x in blob)
    return max(0.0, min(1.0, score))


def discrete_jump_score(market: Dict[str, Any]) -> float:
    one_day_change = abs(safe_float(market.get("oneDayPriceChange")) or 0.0)
    one_hour_change = abs(safe_float(market.get("oneHourPriceChange")) or 0.0)
    return min(1.0, max(one_day_change, one_hour_change) / 0.20)


def data_quality_score(market: Dict[str, Any]) -> float:
    if not has_valid_tokens(market):
        return 0.0
    archived = bool(market.get("archived"))
    malformed = bool(market.get("isMalformed"))
    if archived or malformed:
        return 0.2
    return 1.0


def score_market_base(
    market: Dict[str, Any],
    min_volume: float,
    min_liquidity: float,
) -> Dict[str, float]:
    blob = text_blob(market)
    return {
        "private_decision_score": term_score(blob, HIGH_SIGNAL_TERMS),
        "discrete_jump_score": discrete_jump_score(market),
        "tradability_score": tradability_score(market, min_volume, min_liquidity),
        "data_quality_score": data_quality_score(market),
        "resolution_clarity_score": resolution_clarity_score(blob),
        "sports_penalty": term_score(blob, LOW_SIGNAL_TERMS),
    }


def diversity_scores(markets: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    items = list(markets)
    category_counts = Counter(str(m.get("category") or "unknown").strip().lower() for m in items)
    event_counts = Counter(str(m.get("eventSlug") or m.get("slug") or "unknown").strip().lower() for m in items)

    out: Dict[str, float] = {}
    for m in items:
        cid = str(m.get("conditionId") or "")
        if not cid:
            continue
        cat_key = str(m.get("category") or "unknown").strip().lower()
        evt_key = str(m.get("eventSlug") or m.get("slug") or "unknown").strip().lower()
        cat_freq = max(1, category_counts[cat_key])
        evt_freq = max(1, event_counts[evt_key])

        # Prefer markets from less-concentrated categories/events.
        score = 0.5 * (1.0 / math.sqrt(cat_freq)) + 0.5 * (1.0 / math.sqrt(evt_freq))
        out[cid] = max(0.0, min(1.0, score))
    return out


def final_market_score(base: Dict[str, float], diversity: float) -> float:
    return (
        0.35 * base["private_decision_score"]
        + 0.20 * base["discrete_jump_score"]
        + 0.15 * base["tradability_score"]
        + 0.15 * base["data_quality_score"]
        + 0.10 * base["resolution_clarity_score"]
        + 0.05 * diversity
        - 0.25 * base["sports_penalty"]
    )


def fetch_markets_keyset(
    client: PolymarketClient,
    *,
    page_size: int,
    limit_pages: int,
    closed: bool,
    min_volume: float,
    min_liquidity: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    for page in range(limit_pages):
        params: Dict[str, Any] = {
            "limit": page_size,
            "closed": closed,
            "include_tag": True,
            "order": "volume_num,liquidity_num",
            "ascending": False,
            "volume_num_min": min_volume,
            "liquidity_num_min": min_liquidity,
        }
        if cursor:
            params["after_cursor"] = cursor

        data = client.get_json(GAMMA_BASE, "/markets/keyset", params)
        if isinstance(data, dict):
            markets = data.get("markets") or []
            next_cursor = data.get("next_cursor")
        elif isinstance(data, list):
            markets = data
            next_cursor = None
        else:
            markets = []
            next_cursor = None

        batch = [m for m in markets if isinstance(m, dict)]
        if not batch:
            break

        rows.extend(batch)
        print(f"Fetched discovery page {page + 1}: {len(batch)} markets", file=sys.stderr)

        cursor = str(next_cursor or "").strip() or None
        if cursor is None:
            break

    # Keep the freshest entry for each condition id.
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("conditionId") or "")
        if cid:
            dedup[cid] = row
    return list(dedup.values())


def unix_now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def month_key_for_market(market: Dict[str, Any]) -> str:
    end_date = str(market.get("endDate") or market.get("endDateIso") or "")
    if end_date:
        try:
            return datetime.fromisoformat(end_date.replace("Z", "+00:00")).strftime("%Y-%m")
        except ValueError:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m")


def load_cached_condition_ids(data_dir: Path) -> Set[str]:
    markets_root = data_dir / "market" / "markets"
    if not markets_root.exists():
        return set()

    cached: Set[str] = set()
    for csv_file in markets_root.rglob("market_*.csv"):
        stem = csv_file.stem
        if not stem.startswith("market_"):
            continue
        condition_id = stem[len("market_") :].strip().lower()
        if CONDITION_RE.match(condition_id):
            cached.add(condition_id)
    return cached


def market_cache_path(data_dir: Path, market: Dict[str, Any]) -> Optional[Path]:
    condition_id = str(market.get("conditionId") or "")
    if not condition_id:
        return None
    month = month_key_for_market(market)
    out_dir = data_dir / "market" / "markets" / month
    return out_dir / f"market_{sanitize_file_component(condition_id)}.csv"


def write_market_cache(data_dir: Path, market: Dict[str, Any]) -> None:
    out_path = market_cache_path(data_dir, market)
    if out_path is None:
        return

    row = flatten_market_row(market)
    row["conditionId"] = str(market.get("conditionId") or "")
    write_csv(out_path, [row])


def iter_token_ids(market: Dict[str, Any]) -> Iterable[str]:
    for token_id in parse_jsonish_list(market.get("clobTokenIds")):
        token = str(token_id or "").strip()
        if token:
            yield token


def months_in_window(start_ts: int, end_ts: int) -> Set[str]:
    if end_ts < start_ts:
        return set()
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    out: Set[str] = set()
    while cur <= end_dt:
        out.add(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
    return out


def token_price_history_cached(
    data_dir: Path,
    token_id: str,
    start_ts: int,
    end_ts: int,
) -> bool:
    needed_months = months_in_window(start_ts, end_ts)
    if not needed_months:
        return False

    file_name = f"price_history_asset_{sanitize_file_component(token_id, 'asset')}.csv"
    for month in needed_months:
        out_path = data_dir / "market" / "prices-history" / month / file_name
        if not out_path.exists():
            return False
    return True


def write_price_history_cache(
    client: PolymarketClient,
    data_dir: Path,
    market: Dict[str, Any],
    *,
    fidelity_minutes: int,
    window_days: int,
) -> Tuple[int, int]:
    condition_id = str(market.get("conditionId") or "")
    now_ts = unix_now()
    start_ts = now_ts - max(1, window_days) * 86400

    written_files = 0
    tokens_with_data = 0

    for token_id in iter_token_ids(market):
        try:
            history = fetch_price_history_adaptive(
                client,
                asset=token_id,
                start_ts=start_ts,
                end_ts=now_ts,
                fidelity_minutes=fidelity_minutes,
            )
        except ApiError as exc:
            print(f"Price history failed for {token_id}: {exc}", file=sys.stderr)
            continue

        by_month: Dict[str, List[Dict[str, Any]]] = {}
        for point in history:
            ts = safe_float(point.get("t"))
            px = safe_float(point.get("p"))
            if ts is None or px is None:
                continue

            ts_i = int(ts)
            month = datetime.fromtimestamp(ts_i, tz=timezone.utc).strftime("%Y-%m")
            by_month.setdefault(month, []).append(
                {
                    "asset": token_id,
                    "conditionId": condition_id,
                    "timestamp": ts_i,
                    "timestamp_utc": datetime.fromtimestamp(ts_i, tz=timezone.utc).isoformat(),
                    "price": px,
                    "source": "clob_prices_history",
                }
            )

        for month, rows in by_month.items():
            out_dir = data_dir / "market" / "prices-history" / month
            out_path = out_dir / f"price_history_asset_{sanitize_file_component(token_id, 'asset')}.csv"
            write_csv(out_path, rows)
            written_files += 1

        if by_month:
            tokens_with_data += 1

    return written_files, tokens_with_data


def main() -> None:
    args = parse_args()
    cfg = AnalyzerConfig(sleep_between_requests=args.sleep)
    client = PolymarketClient(cfg)

    data_dir = Path(args.data_dir)
    requested_markets = []
    seen_requested = set()
    for value in [*(args.market or []), *load_market_ids_file(args.markets_file)]:
        condition_id = str(value or "").strip().lower()
        if not condition_id or condition_id in seen_requested:
            continue
        seen_requested.add(condition_id)
        requested_markets.append(condition_id)
    invalid_markets = [value for value in requested_markets if not CONDITION_RE.match(value)]
    if invalid_markets:
        raise SystemExit(f"Invalid conditionId: {invalid_markets[0]}")

    if requested_markets:
        metadata = MetadataCache()
        markets_by_id = metadata.fetch_markets_for_conditions(client, requested_markets)
        missing = [condition_id for condition_id in requested_markets if condition_id not in markets_by_id]
        if missing:
            print(
                f"Could not find metadata for {len(missing)} requested market(s): {', '.join(missing[:5])}",
                file=sys.stderr,
            )
        markets = [markets_by_id[condition_id] for condition_id in requested_markets if condition_id in markets_by_id]
        print(f"Fetched {len(markets)} requested market(s) by conditionId.", file=sys.stderr)
    else:
        markets = fetch_markets_keyset(
            client,
            page_size=args.page_size,
            limit_pages=args.limit_pages,
            closed=args.closed,
            min_volume=args.min_volume,
            min_liquidity=args.min_liquidity,
        )

    cached_condition_ids = set()
    if not args.force_market_refresh:
        cached_condition_ids = load_cached_condition_ids(data_dir)
        if cached_condition_ids:
            before = len(markets)
            markets = [
                m
                for m in markets
                if str(m.get("conditionId") or "").strip().lower() not in cached_condition_ids
            ]
            skipped = before - len(markets)
            print(
                (
                    f"Filtered cached markets from discovery set: skipped={skipped}, "
                    f"remaining_new_candidates={len(markets)}"
                ),
                file=sys.stderr,
            )

    if not markets:
        print("No markets fetched from Gamma.", file=sys.stderr)
        update_watermark(
            data_dir,
            "updates",
            "markets",
            {
                "markets_fetched": 0,
                "markets_written": 0,
                "price_files_written": 0,
                "closed": bool(args.closed),
                "requested_markets": len(requested_markets),
            },
        )
        return

    if requested_markets:
        selected = [(market, {}) for market in markets]
        print(f"Selected {len(selected)} requested markets for caching.", file=sys.stderr)
    else:
        diversity = diversity_scores(markets)

        scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        for market in markets:
            base = score_market_base(market, args.min_volume, args.min_liquidity)
            if base["data_quality_score"] <= 0:
                continue
            div = diversity.get(str(market.get("conditionId") or ""), 0.0)
            score = final_market_score(base, div)
            if score <= 0:
                continue

            features = {**base, "diversity_score": div, "market_score": score}
            scored.append((market, features))

        scored.sort(key=lambda x: x[1]["market_score"], reverse=True)
        selected = scored[: max(0, args.max_markets)] if args.max_markets > 0 else scored
        print(f"Selected {len(selected)} new markets for caching.", file=sys.stderr)

    market_written = 0
    market_skipped_cached = 0
    price_files_written = 0
    price_tokens_skipped_cached = 0

    for idx, (market, _) in enumerate(selected, start=1):
        m_path = market_cache_path(data_dir, market)
        if m_path is not None and m_path.exists() and not args.force_market_refresh:
            market_skipped_cached += 1
        else:
            write_market_cache(data_dir, market)
            market_written += 1

        if not args.skip_price_history:
            now_ts = unix_now()
            start_ts = now_ts - max(1, args.price_window_days) * 86400

            tokens_to_fetch: List[str] = []
            for token_id in iter_token_ids(market):
                if args.force_price_refresh:
                    tokens_to_fetch.append(token_id)
                elif token_price_history_cached(data_dir, token_id, start_ts, now_ts):
                    price_tokens_skipped_cached += 1
                else:
                    tokens_to_fetch.append(token_id)

            if tokens_to_fetch:
                market_for_prices = dict(market)
                market_for_prices["clobTokenIds"] = tokens_to_fetch
                written, _ = write_price_history_cache(
                    client,
                    data_dir,
                    market_for_prices,
                    fidelity_minutes=args.price_fidelity_minutes,
                    window_days=args.price_window_days,
                )
                price_files_written += written

        if idx % 25 == 0 or idx == len(selected):
            print(f"Cached {idx}/{len(selected)} selected markets", file=sys.stderr)

    print(
        (
            "Discovery cache summary: "
            f"markets_written={market_written}, "
            f"markets_skipped_cached={market_skipped_cached}, "
            f"price_files_written={price_files_written}, "
            f"price_tokens_skipped_cached={price_tokens_skipped_cached}"
        ),
        file=sys.stderr,
    )
    update_watermark(
        data_dir,
        "updates",
        "markets",
        {
            "markets_fetched": len(markets),
            "markets_selected": len(selected),
            "markets_written": market_written,
            "markets_skipped_cached": market_skipped_cached,
            "price_files_written": price_files_written,
            "price_tokens_skipped_cached": price_tokens_skipped_cached,
            "closed": bool(args.closed),
            "force_market_refresh": bool(args.force_market_refresh),
            "force_price_refresh": bool(args.force_price_refresh),
            "requested_markets": len(requested_markets),
        },
    )


if __name__ == "__main__":
    main()
