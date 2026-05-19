from __future__ import annotations

import bisect
import json
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from config import CLOB_BASE, CONDITION_RE, DATA_BASE, GAMMA_BASE, WALLET_RE, AnalyzerConfig
from utils import (
    chunks,
    duration_label,
    normalize_user,
    safe_float,
    stable_trade_uid,
    to_iso_utc,
)


class ApiError(RuntimeError):
    pass


class PolymarketClient:
    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg
        self.session = self._new_session()

    def _new_session(self) -> requests.Session:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "polymarket-luck-skill-analyzer/1.0"})
        return self.session

    def _reset_session(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        self.session = self._new_session()

    def get_json(
        self,
        base: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        allow_404: bool = False,
        max_retries: Optional[int] = None,
    ) -> Any:
        url = f"{base}{path}"
        encoded_params = self._encode_params(params or {})
        retry_count = max(1, int(max_retries if max_retries is not None else self.cfg.max_retries))

        last_response: Optional[requests.Response] = None
        for attempt in range(retry_count):
            if self.cfg.sleep_between_requests:
                time.sleep(self.cfg.sleep_between_requests)

            try:
                response = self.session.get(
                    url,
                    params=encoded_params,
                    timeout=self.cfg.timeout_seconds,
                )
            except requests.RequestException as exc:
                self._reset_session()
                if attempt == retry_count - 1:
                    raise ApiError(f"Request failed for {url}: {exc}") from exc
                time.sleep(min(2**attempt, 30))
                continue

            last_response = response

            if allow_404 and response.status_code == 404:
                return None

            if response.status_code == 429 or 500 <= response.status_code < 600:
                wait = min(2**attempt, 30)
                print(
                    f"HTTP {response.status_code} for {url}. Retrying in {wait}s.",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                raise ApiError(
                    f"HTTP {response.status_code} for {url}. Body: {response.text[:500]}"
                )

            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise ApiError(f"Invalid JSON from {url}: {response.text[:500]}") from exc

        assert last_response is not None
        raise ApiError(
            f"Failed after retries: HTTP {last_response.status_code} for {url}. "
            f"Body: {last_response.text[:500]}"
        )

    @staticmethod
    def _encode_params(params: Dict[str, Any]) -> List[Tuple[str, Any]]:
        encoded: List[Tuple[str, Any]] = []
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                encoded.append((key, "true" if value else "false"))
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, bool):
                        encoded.append((key, "true" if item else "false"))
                    else:
                        encoded.append((key, item))
            else:
                encoded.append((key, value))
        return encoded


def fetch_paginated(
    client: PolymarketClient,
    base: str,
    path: str,
    params: Dict[str, Any],
    *,
    limit: int,
    max_offset: int,
    progress_label: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0

    while offset <= max_offset:
        page_params = dict(params)
        page_params["limit"] = limit
        page_params["offset"] = offset
        try:
            batch = client.get_json(base, path, page_params)
        except ApiError as exc:
            message = str(exc)
            if path == "/trades" and "max historical activity offset" in message:
                print(
                    f"Pagination cap reached for {path} at offset {offset}. Retrieved {len(rows)} rows. "
                    "The Polymarket API does not expose older rows beyond this limit.",
                    file=sys.stderr,
                )
                return rows
            raise

        if not isinstance(batch, list):
            raise ApiError(f"Expected list from {path}, got {type(batch).__name__}")

        rows.extend(batch)
        if progress_label:
            page_number = (offset // limit) + 1 if limit > 0 else 1
            print(
                f"    {progress_label}: page {page_number} returned {len(batch)} rows "
                f"({len(rows)} total so far)",
                file=sys.stderr,
            )
        if len(batch) < limit:
            return rows

        offset += limit

    print(
        f"Pagination cap reached for {path}. Retrieved {len(rows)} rows. "
        "A very active user may need archival/on-chain data for completeness.",
        file=sys.stderr,
    )
    return rows


def resolve_user_to_wallet(client: PolymarketClient, user_or_wallet: str) -> Tuple[str, Dict[str, Any]]:
    raw = user_or_wallet.strip()
    if WALLET_RE.match(raw):
        return raw, {"name": raw, "pseudonym": "", "proxyWallet": raw}

    handle = normalize_user(raw)
    result = client.get_json(
        GAMMA_BASE,
        "/public-search",
        {
            "q": handle,
            "search_profiles": True,
            "limit_per_type": 10,
            "page": 1,
        },
    )

    profiles = result.get("profiles") or [] if isinstance(result, dict) else []
    if not profiles:
        raise ApiError(f"No Polymarket profile found for {raw!r}")

    def candidate_names(profile: Dict[str, Any]) -> List[str]:
        return [
            str(profile.get("name") or ""),
            str(profile.get("pseudonym") or ""),
            str(profile.get("referral") or ""),
        ]

    exact_matches: List[Dict[str, Any]] = []
    for profile in profiles:
        names = [name.lower().lstrip("@") for name in candidate_names(profile)]
        if handle.lower() in names:
            exact_matches.append(profile)

    chosen = exact_matches[0] if exact_matches else profiles[0]
    wallet = chosen.get("proxyWallet")
    if not wallet or not WALLET_RE.match(wallet):
        raise ApiError(f"Found profile for {raw!r}, but no valid proxyWallet was returned")

    return wallet, chosen


def fetch_user_bundle(
    client: PolymarketClient,
    wallet: str,
    cfg: AnalyzerConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    current_positions = fetch_paginated(
        client,
        DATA_BASE,
        "/positions",
        {
            "user": wallet,
            "sizeThreshold": 0,
            "sortBy": "TOKENS",
            "sortDirection": "DESC",
        },
        limit=cfg.positions_page_limit,
        max_offset=cfg.positions_max_offset,
        progress_label=f"positions for {wallet}",
    )

    actionable_positions = [
        row for row in current_positions if (safe_float(row.get("size")) or 0.0) > 0.0
    ]
    if not actionable_positions:
        return [], current_positions, []

    trades = fetch_paginated(
        client,
        DATA_BASE,
        "/trades",
        {"user": wallet, "takerOnly": False},
        limit=cfg.trades_page_limit,
        max_offset=cfg.trades_max_offset,
        progress_label=f"trades for {wallet}",
    )
    for idx, trade in enumerate(trades):
        trade["_trade_uid"] = stable_trade_uid(trade, idx)

    closed_positions = fetch_paginated(
        client,
        DATA_BASE,
        "/closed-positions",
        {
            "user": wallet,
            "sortBy": "TIMESTAMP",
            "sortDirection": "DESC",
        },
        limit=cfg.closed_page_limit,
        max_offset=cfg.closed_max_offset,
        progress_label=f"closed positions for {wallet}",
    )

    return trades, current_positions, closed_positions


class MetadataCache:
    def __init__(self) -> None:
        self.markets_by_condition: Dict[str, Dict[str, Any]] = {}
        self.books_by_asset: Dict[str, Optional[Dict[str, Any]]] = {}
        self.histories_by_asset: Dict[str, List[Tuple[float, float]]] = {}
        self.history_ranges_by_asset: Dict[str, Tuple[float, float]] = {}

    def fetch_markets_for_conditions(
        self,
        client: PolymarketClient,
        condition_ids: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        valid_ids = sorted({str(cid) for cid in condition_ids if CONDITION_RE.match(str(cid))})
        missing = [cid for cid in valid_ids if cid not in self.markets_by_condition]

        for chunk in chunks(missing, 50):
            for closed in (False, True):
                found = self._fetch_market_chunk(client, chunk, closed=closed, mode="repeated")
                self.markets_by_condition.update(found)

                still_missing = [cid for cid in chunk if cid not in self.markets_by_condition]
                if still_missing:
                    found = self._fetch_market_chunk(client, still_missing, closed=closed, mode="comma")
                    self.markets_by_condition.update(found)

                still_missing = [cid for cid in chunk if cid not in self.markets_by_condition]
                for cid in still_missing:
                    found = self._fetch_market_chunk(client, [cid], closed=closed, mode="single")
                    self.markets_by_condition.update(found)

        return {cid: self.markets_by_condition[cid] for cid in valid_ids if cid in self.markets_by_condition}

    @staticmethod
    def _fetch_market_chunk(
        client: PolymarketClient,
        condition_ids: Sequence[str],
        *,
        closed: bool,
        mode: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not condition_ids:
            return {}
        if mode == "comma":
            condition_param: Any = ",".join(condition_ids)
        elif mode == "single":
            condition_param = condition_ids[0]
        else:
            condition_param = list(condition_ids)

        try:
            data = client.get_json(
                GAMMA_BASE,
                "/markets",
                {
                    "condition_ids": condition_param,
                    "closed": closed,
                    "limit": max(len(condition_ids), 1),
                },
            )
        except ApiError as exc:
            print(f"Market metadata fetch failed for {len(condition_ids)} ids: {exc}", file=sys.stderr)
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for market in data:
                if not isinstance(market, dict):
                    continue
                cid = market.get("conditionId")
                if cid:
                    result[str(cid)] = market
        return result

    def fetch_books_for_assets(
        self,
        client: PolymarketClient,
        assets: Iterable[str],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        valid_assets = sorted({str(a) for a in assets if str(a)})
        for asset in valid_assets:
            if asset in self.books_by_asset:
                continue
            try:
                book = client.get_json(CLOB_BASE, "/book", {"token_id": asset}, allow_404=True)
            except ApiError as exc:
                print(f"Orderbook fetch failed for asset {asset}: {exc}", file=sys.stderr)
                book = None
            self.books_by_asset[asset] = book if isinstance(book, dict) else None
        return {asset: self.books_by_asset.get(asset) for asset in valid_assets}

    def fetch_price_histories_for_trades(
        self,
        client: PolymarketClient,
        trades: Sequence[Dict[str, Any]],
        horizons_seconds: Sequence[int],
        cfg: AnalyzerConfig,
    ) -> None:
        by_asset: Dict[str, List[float]] = defaultdict(list)
        max_horizon = max(horizons_seconds) if horizons_seconds else 0

        for trade in trades:
            asset = str(trade.get("asset") or "")
            ts = safe_float(trade.get("timestamp"))
            if asset and ts is not None:
                by_asset[asset].append(ts)

        assets = sorted(by_asset.keys())
        if len(assets) > cfg.max_clv_assets_per_user:
            print(
                f"CLV asset cap reached: {len(assets)} assets found, "
                f"using first {cfg.max_clv_assets_per_user}. Increase --max-clv-assets if needed.",
                file=sys.stderr,
            )
            assets = assets[: cfg.max_clv_assets_per_user]

        for asset in assets:
            times = by_asset[asset]
            if not times:
                continue
            start_ts = max(0, min(times) - 3600)
            end_ts = max(times) + max_horizon + 3600
            existing_range = self.history_ranges_by_asset.get(asset)
            if existing_range and existing_range[0] <= start_ts and existing_range[1] >= end_ts:
                continue
            if existing_range:
                start_ts = min(start_ts, existing_range[0])
                end_ts = max(end_ts, existing_range[1])

            try:
                data = client.get_json(
                    CLOB_BASE,
                    "/prices-history",
                    {
                        "market": asset,
                        "startTs": start_ts,
                        "endTs": end_ts,
                        "interval": "all",
                        "fidelity": cfg.clv_fidelity_minutes,
                    },
                )
            except ApiError as exc:
                print(f"Price history fetch failed for asset {asset}: {exc}", file=sys.stderr)
                self.histories_by_asset[asset] = []
                self.history_ranges_by_asset[asset] = (start_ts, end_ts)
                continue

            history_raw = data.get("history") if isinstance(data, dict) else []
            history: List[Tuple[float, float]] = []
            for point in history_raw or []:
                if not isinstance(point, dict):
                    continue
                t = safe_float(point.get("t"))
                p = safe_float(point.get("p"))
                if t is None or p is None:
                    continue
                history.append((t, p))
            history.sort(key=lambda x: x[0])
            self.histories_by_asset[asset] = history
            self.history_ranges_by_asset[asset] = (start_ts, end_ts)


def enrich_trades_with_clv(
    trades: Sequence[Dict[str, Any]],
    cache: MetadataCache,
    horizons_seconds: Sequence[int],
) -> None:
    for trade in trades:
        asset = str(trade.get("asset") or "")
        ts = safe_float(trade.get("timestamp"))
        entry_price = safe_float(trade.get("price"))
        side = str(trade.get("side") or "").upper()
        history = cache.histories_by_asset.get(asset) or []
        times = [x[0] for x in history]

        for horizon in horizons_seconds:
            label = duration_label(horizon)
            trade[f"_ref_price_{label}"] = None
            trade[f"_ref_time_utc_{label}"] = ""
            trade[f"_clv_{label}"] = None

            if not history or ts is None or entry_price is None:
                continue
            target = ts + horizon
            idx = bisect.bisect_left(times, target)
            if idx >= len(history):
                continue
            ref_ts, ref_price = history[idx]
            clv = ref_price - entry_price if side == "BUY" else entry_price - ref_price
            trade[f"_ref_price_{label}"] = ref_price
            trade[f"_ref_time_utc_{label}"] = to_iso_utc(ref_ts)
            trade[f"_clv_{label}"] = clv
