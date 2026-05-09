from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from api import PolymarketClient, fetch_paginated
from cli import write_csv
from config import DATA_BASE, AnalyzerConfig
from utils import parse_jsonish_list, safe_float


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    slug: str
    event_slug: str
    yes_asset: str
    asset_outcome: Dict[str, str]


@dataclass
class JumpEvent:
    event_id: str
    condition_id: str
    question: str
    slug: str
    event_slug: str
    jump_time: float
    p_before: float
    p_after: float
    delta_p: float
    jump_size_logit: float
    vol_sigma: float
    volume_during_jump: float
    volume_before_jump: float
    price_window_seconds: int
    cluster_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Market-first scanner: detect large market moves, then identify users who "
            "built correct exposure before those jumps."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Workspace data directory (default: data)")
    parser.add_argument(
        "--out-dir",
        default="reports/market_scanner",
        help="Output directory for scanner artifacts (default: reports/market_scanner)",
    )
    parser.add_argument(
        "--lookback-windows",
        default="15m,1h,6h,24h,7d",
        help="Comma-separated lookback windows before each jump",
    )
    parser.add_argument(
        "--price-window",
        default="30m",
        help="Window used to measure jump size (default: 30m)",
    )
    parser.add_argument(
        "--tau",
        default="1h",
        help="Exponential decay constant for information proximity weighting (default: 1h)",
    )
    parser.add_argument(
        "--min-abs-move",
        type=float,
        default=0.10,
        help="Minimum absolute price move to consider a jump (default: 0.10)",
    )
    parser.add_argument(
        "--jump-sigma-mult",
        type=float,
        default=2.5,
        help="Minimum logit jump multiple of market volatility sigma (default: 2.5)",
    )
    parser.add_argument(
        "--volume-spike-mult",
        type=float,
        default=1.5,
        help="Require jump window volume >= multiplier * prior window volume (default: 1.5)",
    )
    parser.add_argument(
        "--min-history-points",
        type=int,
        default=50,
        help="Minimum YES price points required to scan a market (default: 50)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=0,
        help="Optional cap on markets scanned (0 means no cap)",
    )
    parser.add_argument(
        "--market",
        action="append",
        help=(
            "Optional specific market target(s): conditionId or slug. "
            "Repeatable. If omitted, scans all cached markets."
        ),
    )
    parser.add_argument(
        "--market-trades-cache-dir",
        default="market/trades",
        help=(
            "Relative path under data-dir for cached market trades "
            "(default: market/trades)"
        ),
    )
    parser.add_argument(
        "--refresh-market-trades",
        action="store_true",
        help="Force full market-trade refresh from API for scanned markets.",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Do not call APIs. Use only cached market trades in data/market/trades.",
    )
    parser.add_argument(
        "--event-cluster-window",
        default="24h",
        help=(
            "Cluster width used to de-duplicate related markets in same event into "
            "independent information events (default: 24h)"
        ),
    )
    parser.add_argument(
        "--min-independent-events",
        type=int,
        default=2,
        help="Minimum independent event clusters to classify beyond low confidence (default: 2)",
    )
    parser.add_argument(
        "--max-single-event-share",
        type=float,
        default=0.65,
        help=(
            "If one event contributes over this share of total capture, lower confidence "
            "(default: 0.65)"
        ),
    )
    parser.add_argument(
        "--min-directional-ratio",
        type=float,
        default=0.10,
        help=(
            "Minimum directional exposure ratio to count an event as informational "
            "(downweights market-maker-like symmetric flow, default: 0.10)"
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between API calls",
    )
    parser.add_argument(
        "--trades-page-limit",
        type=int,
        default=1000,
        help="Page size for market trade pulls",
    )
    parser.add_argument(
        "--trades-max-offset",
        type=int,
        default=3000,
        help="Max historical offset for /trades endpoint",
    )
    return parser.parse_args()


def parse_duration_seconds(text: str) -> int:
    s = text.strip().lower()
    if not s:
        raise ValueError("Empty duration")
    mult = 1
    if s.endswith("d"):
        mult = 86400
        s = s[:-1]
    elif s.endswith("h"):
        mult = 3600
        s = s[:-1]
    elif s.endswith("m"):
        mult = 60
        s = s[:-1]
    elif s.endswith("s"):
        mult = 1
        s = s[:-1]
    return int(float(s) * mult)


def clamp_prob(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, p))


def logit(p: float) -> float:
    q = clamp_prob(p)
    return math.log(q / (1.0 - q))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def slugify_text(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    text = "_".join(part for part in text.split("_") if part)
    return text or "unknown"


def trade_uid(trade: Dict[str, Any], idx: int = 0) -> str:
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
        str(idx),
    ]
    return "|".join(pieces)


def market_trade_cache_path(data_dir: Path, cache_dir: str, condition_id: str) -> Path:
    return data_dir / cache_dir / condition_id / "trades.csv"


def load_cached_market_trades(data_dir: Path, cache_dir: str, condition_id: str) -> List[Dict[str, Any]]:
    path = market_trade_cache_path(data_dir, cache_dir, condition_id)
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
    except OSError as exc:
        print(f"Failed reading cached trades for {condition_id}: {exc}", file=sys.stderr)
        return []

    rows.sort(key=lambda r: safe_float(r.get("timestamp")) or -1.0)
    return rows


def write_market_trade_cache(data_dir: Path, cache_dir: str, condition_id: str, rows: Sequence[Dict[str, Any]]) -> None:
    path = market_trade_cache_path(data_dir, cache_dir, condition_id)
    write_csv(path, rows)


def fetch_market_trades_incremental(
    client: PolymarketClient,
    condition_id: str,
    *,
    page_limit: int,
    max_offset: int,
    seen_uids: Set[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0

    while offset <= max_offset:
        batch = client.get_json(
            DATA_BASE,
            "/trades",
            {
                "market": condition_id,
                "takerOnly": False,
                "limit": page_limit,
                "offset": offset,
            },
        )
        if not isinstance(batch, list):
            break
        if not batch:
            break

        unseen_in_batch = 0
        for idx, row in enumerate(batch):
            if not isinstance(row, dict):
                continue
            uid = trade_uid(row, idx)
            if uid in seen_uids:
                continue
            rows.append(row)
            seen_uids.add(uid)
            unseen_in_batch += 1

        page_number = (offset // page_limit) + 1 if page_limit > 0 else 1
        print(
            f"    market trades {condition_id[:10]}... page {page_number} unseen={unseen_in_batch}",
            file=sys.stderr,
        )

        if unseen_in_batch == 0:
            break
        if len(batch) < page_limit:
            break
        offset += page_limit

    rows.sort(key=lambda r: safe_float(r.get("timestamp")) or -1.0)
    return rows


def select_market_ids(markets: Dict[str, MarketInfo], requested: Optional[Sequence[str]]) -> List[str]:
    market_ids = sorted(markets.keys())
    if not requested:
        return market_ids

    by_slug: Dict[str, str] = {}
    for cid, info in markets.items():
        slug = info.slug.strip().lower()
        if slug and slug not in by_slug:
            by_slug[slug] = cid

    selected: List[str] = []
    seen: Set[str] = set()
    for token in requested:
        key = str(token or "").strip()
        if not key:
            continue
        cid = key if key in markets else by_slug.get(key.lower())
        if not cid:
            print(f"Warning: requested market not found in cache: {key}", file=sys.stderr)
            continue
        if cid in seen:
            continue
        seen.add(cid)
        selected.append(cid)
    return selected


def build_event_cluster_id(event: JumpEvent, cluster_window_seconds: int) -> str:
    base = event.event_slug or event.slug or event.condition_id
    bucket = int(event.jump_time) // max(1, cluster_window_seconds)
    return f"{slugify_text(base)}:{bucket}"


def load_markets(data_dir: Path) -> Dict[str, MarketInfo]:
    markets_root = data_dir / "market" / "markets"
    if not markets_root.exists():
        return {}

    result: Dict[str, MarketInfo] = {}
    for csv_file in sorted(markets_root.rglob("market_*.csv")):
        try:
            with csv_file.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    condition_id = str(row.get("conditionId") or "")
                    if not condition_id:
                        continue
                    outcomes = [str(x) for x in parse_jsonish_list(row.get("outcomes"))]
                    token_ids = [str(x) for x in parse_jsonish_list(row.get("clobTokenIds"))]
                    if len(token_ids) < 2 or len(outcomes) < 2:
                        continue

                    yes_idx = None
                    for i, label in enumerate(outcomes):
                        if label.strip().lower() == "yes":
                            yes_idx = i
                            break
                    if yes_idx is None:
                        yes_idx = 0

                    if yes_idx >= len(token_ids):
                        continue

                    asset_outcome = {}
                    for i, asset in enumerate(token_ids):
                        label = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
                        asset_outcome[asset] = str(label)

                    result[condition_id] = MarketInfo(
                        condition_id=condition_id,
                        question=str(row.get("question") or ""),
                        slug=str(row.get("slug") or ""),
                        event_slug=str(row.get("eventSlug") or ""),
                        yes_asset=token_ids[yes_idx],
                        asset_outcome=asset_outcome,
                    )
        except OSError as exc:
            print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)
    return result


def load_price_history_by_asset(data_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    prices_root = data_dir / "market" / "prices-history"
    if not prices_root.exists():
        return {}

    by_asset: Dict[str, Dict[int, float]] = defaultdict(dict)
    for csv_file in sorted(prices_root.rglob("price_history_asset_*.csv")):
        try:
            with csv_file.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    asset = str(row.get("asset") or "")
                    ts = safe_float(row.get("timestamp"))
                    px = safe_float(row.get("price"))
                    if not asset or ts is None or px is None:
                        continue
                    by_asset[asset][int(ts)] = px
        except OSError as exc:
            print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)

    result: Dict[str, List[Tuple[float, float]]] = {}
    for asset, values in by_asset.items():
        points = [(float(ts), px) for ts, px in sorted(values.items())]
        if points:
            result[asset] = points
    return result


def estimate_logit_volatility(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    diffs: List[float] = []
    prev = logit(points[0][1])
    for _, p in points[1:]:
        cur = logit(p)
        diffs.append(cur - prev)
        prev = cur
    if len(diffs) < 2:
        return 0.0
    mean = sum(diffs) / len(diffs)
    var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
    return math.sqrt(max(0.0, var))


def detect_jump_candidates(
    market: MarketInfo,
    yes_points: Sequence[Tuple[float, float]],
    *,
    price_window_seconds: int,
    min_abs_move: float,
    jump_sigma_mult: float,
) -> List[JumpEvent]:
    if len(yes_points) < 3:
        return []

    sigma = estimate_logit_volatility(yes_points)
    threshold = jump_sigma_mult * sigma if sigma > 0 else 0.0
    timestamps = [x[0] for x in yes_points]

    jumps: List[JumpEvent] = []
    for i, (ts_before, p_before) in enumerate(yes_points):
        target = ts_before + price_window_seconds
        j = bisect_left(timestamps, target)
        if j >= len(yes_points):
            break
        ts_after, p_after = yes_points[j]
        if ts_after <= ts_before:
            continue

        dp = p_after - p_before
        if abs(dp) < min_abs_move:
            continue

        ljump = logit(p_after) - logit(p_before)
        if threshold > 0 and abs(ljump) < threshold:
            continue

        event_id = f"{market.condition_id}:{int(ts_after)}"
        jumps.append(
            JumpEvent(
                event_id=event_id,
                condition_id=market.condition_id,
                question=market.question,
                slug=market.slug,
                event_slug=market.event_slug,
                jump_time=ts_after,
                p_before=p_before,
                p_after=p_after,
                delta_p=dp,
                jump_size_logit=ljump,
                vol_sigma=sigma,
                volume_during_jump=0.0,
                volume_before_jump=0.0,
                price_window_seconds=price_window_seconds,
            )
        )

    # De-duplicate overlapping detections by keeping only the largest jump in each nearby cluster.
    jumps.sort(key=lambda e: e.jump_time)
    deduped: List[JumpEvent] = []
    cluster: List[JumpEvent] = []
    for evt in jumps:
        if not cluster:
            cluster = [evt]
            continue
        if evt.jump_time - cluster[-1].jump_time <= price_window_seconds:
            cluster.append(evt)
            continue
        deduped.append(max(cluster, key=lambda x: abs(x.jump_size_logit)))
        cluster = [evt]
    if cluster:
        deduped.append(max(cluster, key=lambda x: abs(x.jump_size_logit)))
    return deduped


def signed_yes_exposure(trade: Dict[str, Any], market: MarketInfo) -> Optional[float]:
    side = str(trade.get("side") or "").upper()
    size = safe_float(trade.get("size"))
    if side not in {"BUY", "SELL"} or size is None or size <= 0:
        return None

    outcome = str(trade.get("outcome") or "").strip().lower()
    asset = str(trade.get("asset") or "")

    is_yes: Optional[bool] = None
    if outcome:
        if outcome == "yes":
            is_yes = True
        elif outcome == "no":
            is_yes = False

    if is_yes is None and asset:
        label = market.asset_outcome.get(asset, "").strip().lower()
        if label == "yes":
            is_yes = True
        elif label == "no":
            is_yes = False

    if is_yes is None and asset:
        # Fallback for non-YES/NO labels: treat primary YES asset as positive side.
        is_yes = asset == market.yes_asset

    if is_yes is None:
        return None

    if side == "BUY" and is_yes:
        return size
    if side == "SELL" and is_yes:
        return -size
    if side == "BUY" and not is_yes:
        return -size
    if side == "SELL" and not is_yes:
        return size
    return None


def market_trade_rows(
    client: PolymarketClient,
    data_dir: Path,
    cache_dir: str,
    condition_id: str,
    *,
    page_limit: int,
    max_offset: int,
    allow_api: bool,
    force_refresh: bool,
) -> List[Dict[str, Any]]:
    cached_rows = load_cached_market_trades(data_dir, cache_dir, condition_id)
    if not allow_api:
        return cached_rows

    if force_refresh:
        fetched = fetch_paginated(
            client,
            DATA_BASE,
            "/trades",
            {
                "market": condition_id,
                "takerOnly": False,
            },
            limit=page_limit,
            max_offset=max_offset,
            progress_label=f"market trades {condition_id[:10]}...",
        )
        fetched.sort(key=lambda r: safe_float(r.get("timestamp")) or -1.0)
        write_market_trade_cache(data_dir, cache_dir, condition_id, fetched)
        return fetched

    seen_uids = {trade_uid(row, i) for i, row in enumerate(cached_rows)}
    new_rows = fetch_market_trades_incremental(
        client,
        condition_id,
        page_limit=page_limit,
        max_offset=max_offset,
        seen_uids=seen_uids,
    )

    if not new_rows:
        return cached_rows

    combined = list(cached_rows) + new_rows
    dedup: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(combined):
        dedup[trade_uid(row, i)] = row
    out = list(dedup.values())
    out.sort(key=lambda r: safe_float(r.get("timestamp")) or -1.0)
    write_market_trade_cache(data_dir, cache_dir, condition_id, out)
    return out


def sum_notional_in_window(trades: Sequence[Dict[str, Any]], start_ts: float, end_ts: float) -> float:
    total = 0.0
    for t in trades:
        ts = safe_float(t.get("timestamp"))
        px = safe_float(t.get("price"))
        sz = safe_float(t.get("size"))
        if ts is None or px is None or sz is None or sz <= 0:
            continue
        if start_ts <= ts <= end_ts:
            total += px * sz
    return total


def event_user_captures(
    event: JumpEvent,
    market: MarketInfo,
    trades: Sequence[Dict[str, Any]],
    lookbacks: Sequence[int],
    tau_seconds: float,
) -> Dict[str, Dict[str, Any]]:
    by_user: Dict[str, Dict[str, Any]] = {}
    for lookback in lookbacks:
        start_ts = event.jump_time - lookback
        end_ts = event.jump_time
        per_user_q: Dict[str, float] = defaultdict(float)
        per_user_notional: Dict[str, float] = defaultdict(float)

        for trade in trades:
            ts = safe_float(trade.get("timestamp"))
            if ts is None or ts < start_ts or ts > end_ts:
                continue
            q = signed_yes_exposure(trade, market)
            px = safe_float(trade.get("price"))
            if q is None or px is None:
                continue

            lead = max(0.0, event.jump_time - ts)
            weight = math.exp(-lead / tau_seconds) if tau_seconds > 0 else 1.0
            weighted_q = q * weight

            wallet = str(trade.get("proxyWallet") or "").strip().lower()
            if not wallet:
                continue

            per_user_q[wallet] += weighted_q
            per_user_notional[wallet] += abs(q) * px

        for wallet, q_sum in per_user_q.items():
            capture = q_sum * event.delta_p
            gross_notional = per_user_notional.get(wallet, 0.0)
            directional_ratio = abs(q_sum) / gross_notional if gross_notional > 0 else 0.0
            row = by_user.get(wallet)
            if row is None or abs(capture) > abs(row["jump_capture"]):
                by_user[wallet] = {
                    "wallet": wallet,
                    "jump_capture": capture,
                    "signed_exposure": q_sum,
                    "gross_notional": gross_notional,
                    "directional_ratio": directional_ratio,
                    "chosen_lookback_seconds": lookback,
                }
    return by_user


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    lookbacks = [parse_duration_seconds(x) for x in args.lookback_windows.split(",") if x.strip()]
    if not lookbacks:
        print("ERROR: No lookback windows parsed.", file=sys.stderr)
        sys.exit(1)

    price_window_seconds = parse_duration_seconds(args.price_window)
    tau_seconds = float(parse_duration_seconds(args.tau))
    cluster_window_seconds = parse_duration_seconds(args.event_cluster_window)

    markets = load_markets(data_dir)
    if not markets:
        print(f"ERROR: No market metadata found under {data_dir / 'market' / 'markets'}", file=sys.stderr)
        sys.exit(1)

    by_asset = load_price_history_by_asset(data_dir)
    if not by_asset:
        print(f"ERROR: No price history found under {data_dir / 'market' / 'prices-history'}", file=sys.stderr)
        sys.exit(1)

    cfg = AnalyzerConfig(
        sleep_between_requests=args.sleep,
        trades_page_limit=args.trades_page_limit,
        trades_max_offset=args.trades_max_offset,
    )
    client = PolymarketClient(cfg)

    events: List[JumpEvent] = []
    market_ids = select_market_ids(markets, args.market)

    if args.max_markets > 0:
        market_ids = market_ids[: args.max_markets]

    print(f"Scanning {len(market_ids)} markets for jump events...", file=sys.stderr)
    scanned = 0
    for condition_id in market_ids:
        scanned += 1
        market = markets[condition_id]
        yes_points = by_asset.get(market.yes_asset) or []
        if len(yes_points) < args.min_history_points:
            continue

        candidates = detect_jump_candidates(
            market,
            yes_points,
            price_window_seconds=price_window_seconds,
            min_abs_move=args.min_abs_move,
            jump_sigma_mult=args.jump_sigma_mult,
        )
        for evt in candidates:
            evt.cluster_id = build_event_cluster_id(evt, cluster_window_seconds)
        events.extend(candidates)

        if scanned % 100 == 0 or scanned == len(market_ids):
            print(f"  market scan progress {scanned}/{len(market_ids)} | events={len(events)}", file=sys.stderr)

    if not events:
        print("No jump events detected with current thresholds.", file=sys.stderr)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "jump_events.csv", [])
        write_csv(out_dir / "candidate_users.csv", [])
        return

    events.sort(key=lambda e: e.jump_time)
    print(f"Detected {len(events)} jump candidates. Fetching market trades...", file=sys.stderr)

    # Aggregate user-event captures.
    user_event_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    captures_by_event: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    trades_cache: Dict[str, List[Dict[str, Any]]] = {}
    kept_events = 0

    for idx, event in enumerate(events, start=1):
        market = markets.get(event.condition_id)
        if market is None:
            continue

        if event.condition_id not in trades_cache:
            trades_cache[event.condition_id] = market_trade_rows(
                client,
                data_dir,
                args.market_trades_cache_dir,
                event.condition_id,
                page_limit=cfg.trades_page_limit,
                max_offset=cfg.trades_max_offset,
                allow_api=not args.no_api,
                force_refresh=args.refresh_market_trades,
            )
        market_trades = trades_cache[event.condition_id]

        jump_start = event.jump_time - event.price_window_seconds
        vol_during = sum_notional_in_window(market_trades, jump_start, event.jump_time)
        vol_before = sum_notional_in_window(
            market_trades,
            jump_start - event.price_window_seconds,
            jump_start,
        )
        event.volume_during_jump = vol_during
        event.volume_before_jump = vol_before

        if vol_before > 0 and vol_during < args.volume_spike_mult * vol_before:
            continue

        per_user = event_user_captures(event, market, market_trades, lookbacks, tau_seconds)
        if not per_user:
            continue

        kept_events += 1
        event_rows.append(
            {
                "event_id": event.event_id,
                "cluster_id": event.cluster_id,
                "condition_id": event.condition_id,
                "question": event.question,
                "slug": event.slug,
                "event_slug": event.event_slug,
                "jump_time": int(event.jump_time),
                "jump_time_utc": datetime.fromtimestamp(event.jump_time, tz=timezone.utc).isoformat(),
                "p_before": event.p_before,
                "p_after": event.p_after,
                "delta_p": event.delta_p,
                "jump_size_logit": event.jump_size_logit,
                "vol_sigma": event.vol_sigma,
                "volume_during_jump": event.volume_during_jump,
                "volume_before_jump": event.volume_before_jump,
                "price_window_seconds": event.price_window_seconds,
                "participants": len(per_user),
            }
        )

        for wallet, rec in per_user.items():
            directional_ratio = safe_float(rec.get("directional_ratio")) or 0.0
            directional_weight = min(1.0, directional_ratio / max(args.min_directional_ratio, 1e-6))

            base_capture = float(rec["jump_capture"])
            capture = base_capture * directional_weight

            captures_by_event[event.event_id].append((wallet, capture))
            user_event_rows.append(
                {
                    "event_id": event.event_id,
                    "cluster_id": event.cluster_id,
                    "condition_id": event.condition_id,
                    "wallet": wallet,
                    "jump_capture": capture,
                    "raw_jump_capture": base_capture,
                    "signed_exposure": rec["signed_exposure"],
                    "gross_notional": rec["gross_notional"],
                    "directional_ratio": directional_ratio,
                    "directional_weight": directional_weight,
                    "chosen_lookback_seconds": rec["chosen_lookback_seconds"],
                    "jump_time": int(event.jump_time),
                    "delta_p": event.delta_p,
                }
            )

        if idx % 25 == 0 or idx == len(events):
            print(f"  event scoring progress {idx}/{len(events)} | kept={kept_events}", file=sys.stderr)

    if not user_event_rows:
        print("No user-event captures after filters.", file=sys.stderr)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "jump_events.csv", event_rows)
        write_csv(out_dir / "candidate_users.csv", [])
        write_csv(out_dir / "user_event_scores.csv", [])
        return

    # Standardize per event to reduce size bias and event heterogeneity.
    event_stats: Dict[str, Tuple[float, float]] = {}
    for event_id, values in captures_by_event.items():
        captures = [c for _, c in values]
        if len(captures) < 2:
            event_stats[event_id] = (captures[0] if captures else 0.0, 0.0)
            continue
        mu = sum(captures) / len(captures)
        var = sum((x - mu) ** 2 for x in captures) / (len(captures) - 1)
        event_stats[event_id] = (mu, math.sqrt(max(0.0, var)))

    user_acc: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "wallet": "",
            "total_jump_capture": 0.0,
            "gross_notional": 0.0,
            "event_ids": set(),
            "cluster_ids": set(),
            "cluster_best": {},
            "z_parts": [],
            "jumps_captured": 0,
            "max_single_event_contribution": 0.0,
            "avg_directional_ratio": 0.0,
            "_directional_count": 0,
        }
    )

    for row in user_event_rows:
        wallet = str(row["wallet"])
        capture = safe_float(row.get("jump_capture")) or 0.0
        event_id = str(row["event_id"])
        cluster_id = str(row.get("cluster_id") or event_id)
        mu, sd = event_stats.get(event_id, (0.0, 0.0))
        z = (capture - mu) / sd if sd > 0 else 0.0

        rec = user_acc[wallet]
        rec["wallet"] = wallet
        rec["total_jump_capture"] += capture
        rec["gross_notional"] += safe_float(row.get("gross_notional")) or 0.0
        rec["event_ids"].add(event_id)
        rec["cluster_ids"].add(cluster_id)
        rec["z_parts"].append(z)
        if capture > 0 and (safe_float(row.get("directional_ratio")) or 0.0) >= args.min_directional_ratio:
            rec["jumps_captured"] += 1
        rec["max_single_event_contribution"] = max(
            rec["max_single_event_contribution"],
            abs(capture),
        )

        rec["avg_directional_ratio"] += safe_float(row.get("directional_ratio")) or 0.0
        rec["_directional_count"] += 1

        # Keep only the strongest event per cluster to avoid over-counting related markets.
        best = rec["cluster_best"].get(cluster_id)
        if best is None or abs(capture) > abs(best["capture"]):
            rec["cluster_best"][cluster_id] = {
                "capture": capture,
                "z": z,
                "event_id": event_id,
            }

    candidates: List[Dict[str, Any]] = []
    for wallet, rec in user_acc.items():
        cluster_best = rec["cluster_best"]
        z_parts = [float(v["z"]) for v in cluster_best.values()]
        n = len(z_parts)
        if n == 0:
            continue
        mean_z = sum(z_parts) / n
        timing_z = mean_z * math.sqrt(n)
        timing_index = 100.0 * normal_cdf(timing_z)

        event_count = len(cluster_best)

        capped_total_capture = sum(float(v["capture"]) for v in cluster_best.values())
        max_cluster_capture = max((abs(float(v["capture"])) for v in cluster_best.values()), default=0.0)
        max_single_event_share = (
            max_cluster_capture / abs(capped_total_capture)
            if abs(capped_total_capture) > 1e-9
            else 1.0
        )

        avg_directional_ratio = (
            rec["avg_directional_ratio"] / rec["_directional_count"]
            if rec["_directional_count"] > 0
            else 0.0
        )

        confidence = "low"
        if event_count < args.min_independent_events:
            confidence = "low"
        elif event_count >= 10 and timing_z >= 5.0:
            confidence = "very_high"
        elif event_count >= 5 and timing_z >= 4.0:
            confidence = "high"
        elif event_count >= 3 and timing_z >= 3.0:
            confidence = "candidate"
        elif event_count >= 2 and timing_z >= 2.0:
            confidence = "monitor"

        if max_single_event_share > args.max_single_event_share:
            if confidence == "very_high":
                confidence = "high"
            elif confidence == "high":
                confidence = "candidate"
            elif confidence == "candidate":
                confidence = "monitor"
            else:
                confidence = "low"

        candidates.append(
            {
                "wallet": wallet,
                "total_jump_capture": rec["total_jump_capture"],
                "cluster_capped_total_jump_capture": capped_total_capture,
                "timing_z": timing_z,
                "timing_index": timing_index,
                "number_of_jumps_captured": rec["jumps_captured"],
                "number_of_independent_events": event_count,
                "max_single_event_contribution": rec["max_single_event_contribution"],
                "max_single_event_share": max_single_event_share,
                "gross_notional": rec["gross_notional"],
                "avg_directional_ratio": avg_directional_ratio,
                "confidence_level": confidence,
            }
        )

    candidates.sort(
        key=lambda r: (safe_float(r.get("timing_z")) or -999.0, safe_float(r.get("total_jump_capture")) or -999.0),
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "jump_events.csv", event_rows)
    write_csv(out_dir / "user_event_scores.csv", user_event_rows)
    write_csv(out_dir / "candidate_users.csv", candidates)

    print("", file=sys.stderr)
    print(f"Jump events written: {out_dir / 'jump_events.csv'}", file=sys.stderr)
    print(f"User-event scores written: {out_dir / 'user_event_scores.csv'}", file=sys.stderr)
    print(f"Candidate users written: {out_dir / 'candidate_users.csv'}", file=sys.stderr)

    top_n = min(15, len(candidates))
    if top_n:
        print("", file=sys.stderr)
        print("Top candidate users by TimingZ:", file=sys.stderr)
        for row in candidates[:top_n]:
            print(
                f"  {row['wallet']} | timing_z={row['timing_z']:.3f} | "
                f"timing_index={row['timing_index']:.2f} | "
                f"events={row['number_of_independent_events']} | "
                f"capture={row['total_jump_capture']:.2f}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
