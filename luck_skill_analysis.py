from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from config import AnalyzerConfig
from download_data import download_for_user
from runtime_state import update_watermark
from sqlite_store import (
    ensure_database,
    list_user_keys as sqlite_list_user_keys,
    load_all_market_rows as sqlite_load_all_market_rows,
    load_all_user_trades as sqlite_load_all_user_trades,
    load_price_history_points as sqlite_load_price_history_points,
    load_user_alias_rows as sqlite_load_user_alias_rows,
    load_user_profile_rows as sqlite_load_user_profile_rows,
    upsert_user_metric as sqlite_upsert_user_metric,
    user_has_trades as sqlite_user_has_trades,
)
from utils import safe_float


ProbabilityCalibrator = Callable[[float], float]


@dataclass(frozen=True)
class TradeDecomposition:
    trade_id: str
    condition_id: str
    outcome_key: str
    side: str
    size: float
    price: float
    closing_probability: float
    realized_outcome: float
    signed_shares: float
    notional_usdc: float
    skill_usdc: float
    luck_usdc: float
    pnl_usdc: float


@dataclass(frozen=True)
class MarketLuckBreakdown:
    condition_id: str
    market_luck: float
    variance: float
    expected_exposure: float
    net_exposures_by_outcome: Dict[str, float]
    probabilities_by_outcome: Dict[str, float]
    realized_by_outcome: Dict[str, float]


@dataclass(frozen=True)
class LuckSkillResult:
    luck_index: Optional[float]
    luck_z: Optional[float]
    skill_index: Optional[float]
    skill_z: Optional[float]
    skill_randomized_sd_usdc: float
    total_luck_usdc: float
    total_variance: float
    effective_markets: float
    number_of_resolved_markets: int
    skill_usdc: float
    skill_roi: Optional[float]
    raw_pnl_usdc: float
    gross_trade_notional_usdc: float
    trade_decompositions: List[TradeDecomposition]
    market_breakdowns: List[MarketLuckBreakdown]
    warnings: List[str]


def analyze_luck_and_skill(
    trades: Sequence[Dict[str, Any]],
    markets_by_condition: Dict[str, Dict[str, Any]],
    *,
    min_probability: float = 0.01,
    max_probability: float = 0.99,
    minimum_effective_markets: float = 10.0,
    probability_calibrator: Optional[ProbabilityCalibrator] = None,
) -> LuckSkillResult:
    """Decompose realized PnL into skill and luck using pre-resolution probabilities.

    Required trade fields:
    - conditionId
    - side (BUY/SELL)
    - size
    - price
    - optional: asset, outcome, outcomeIndex, id, _trade_uid

    Required market fields per condition:
    - resolved outcome signal via one of winningOutcomeIndex / winner / winning_outcome_index
      or via outcomes + winner label
    - outcome probabilities via one of outcomePrices / closingOutcomePrices / closing_probabilities
    - outcome identifiers via one of clobTokenIds / outcomes

    This implementation aggregates variance at the market level to avoid counting correlated
    trades as independent bets.
    """
    warnings: List[str] = []

    trade_decompositions: List[TradeDecomposition] = []
    exposure_by_market_outcome: Dict[str, Dict[str, float]] = {}
    variance_by_market: Dict[str, float] = {}
    outcome_info_by_market: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {}

    def outcome_info_for_market(condition_id: str, market: Dict[str, Any]) -> Optional[Dict[str, Dict[str, float]]]:
        if condition_id not in outcome_info_by_market:
            outcome_info_by_market[condition_id] = _extract_market_outcome_data(
                market,
                min_probability=min_probability,
                max_probability=max_probability,
                probability_calibrator=probability_calibrator,
            )
        return outcome_info_by_market[condition_id]

    skill_total = 0.0
    luck_total = 0.0
    pnl_total = 0.0
    gross_notional = 0.0
    skill_randomized_variance = 0.0

    for idx, trade in enumerate(trades):
        condition_id = str(trade.get("conditionId") or "")
        if not condition_id:
            warnings.append(f"Skipped trade index {idx}: missing conditionId")
            continue

        market = markets_by_condition.get(condition_id)
        if not market:
            warnings.append(
                f"Skipped trade index {idx} (conditionId={condition_id}): missing market metadata"
            )
            continue

        outcome_info = outcome_info_for_market(condition_id, market)
        if not outcome_info:
            warnings.append(
                f"Skipped trade index {idx} (conditionId={condition_id}): market has no valid resolved probabilities"
            )
            continue

        outcome_key = _trade_outcome_key(trade, market, outcome_info["outcome_keys"])  # type: ignore[index]
        if outcome_key not in outcome_info["probabilities"]:
            warnings.append(
                f"Skipped trade index {idx} (conditionId={condition_id}): could not map trade outcome"
            )
            continue

        side = str(trade.get("side") or "").upper()
        sign = 1.0 if side == "BUY" else -1.0 if side == "SELL" else 0.0
        if sign == 0.0:
            warnings.append(
                f"Skipped trade index {idx} (conditionId={condition_id}): unsupported side={trade.get('side')}"
            )
            continue

        size = safe_float(trade.get("size"))
        price = safe_float(trade.get("price"))
        if size is None or price is None or size <= 0:
            warnings.append(
                f"Skipped trade index {idx} (conditionId={condition_id}): invalid size/price"
            )
            continue

        c_i = outcome_info["probabilities"][outcome_key]
        y_i = outcome_info["realized"][outcome_key]

        signed_shares = sign * size
        notional = size * price
        skill = sign * size * (c_i - price)
        luck = sign * size * (y_i - c_i)
        pnl = sign * size * (y_i - price)

        # Proxy randomized baseline for skill: same trade opportunities but random side (+/-1).
        # Under this baseline Var(skill_i) = (q_i * (c_i - p_i))^2.
        edge_magnitude = size * (c_i - price)
        skill_randomized_variance += edge_magnitude * edge_magnitude

        trade_decompositions.append(
            TradeDecomposition(
                trade_id=_trade_id(trade, idx),
                condition_id=condition_id,
                outcome_key=outcome_key,
                side=side,
                size=size,
                price=price,
                closing_probability=c_i,
                realized_outcome=y_i,
                signed_shares=signed_shares,
                notional_usdc=notional,
                skill_usdc=skill,
                luck_usdc=luck,
                pnl_usdc=pnl,
            )
        )

        market_exposures = exposure_by_market_outcome.setdefault(condition_id, {})
        market_exposures[outcome_key] = market_exposures.get(outcome_key, 0.0) + signed_shares

        skill_total += skill
        luck_total += luck
        pnl_total += pnl
        gross_notional += notional

    market_breakdowns: List[MarketLuckBreakdown] = []
    for condition_id, exposures in exposure_by_market_outcome.items():
        market = markets_by_condition.get(condition_id)
        if not market:
            continue

        outcome_info = outcome_info_for_market(condition_id, market)
        if not outcome_info:
            continue

        probabilities: Dict[str, float] = outcome_info["probabilities"]  # type: ignore[assignment]
        realized: Dict[str, float] = outcome_info["realized"]  # type: ignore[assignment]

        # Use all known outcomes to preserve covariance in multi-outcome markets.
        all_outcomes = set(probabilities.keys()) | set(exposures.keys())
        q_by_outcome = {key: exposures.get(key, 0.0) for key in all_outcomes}

        market_luck = 0.0
        expected_exposure = 0.0
        for key in all_outcomes:
            c = probabilities.get(key, 0.0)
            y = realized.get(key, 0.0)
            q = q_by_outcome[key]
            market_luck += q * (y - c)
            expected_exposure += c * q

        variance = 0.0
        for key in all_outcomes:
            c = probabilities.get(key, 0.0)
            q = q_by_outcome[key]
            variance += c * ((q - expected_exposure) ** 2)

        variance_by_market[condition_id] = variance
        market_breakdowns.append(
            MarketLuckBreakdown(
                condition_id=condition_id,
                market_luck=market_luck,
                variance=variance,
                expected_exposure=expected_exposure,
                net_exposures_by_outcome=dict(sorted(q_by_outcome.items())),
                probabilities_by_outcome={k: probabilities.get(k, 0.0) for k in sorted(all_outcomes)},
                realized_by_outcome={k: realized.get(k, 0.0) for k in sorted(all_outcomes)},
            )
        )

    total_variance = sum(variance_by_market.values())
    luck_z = luck_total / math.sqrt(total_variance) if total_variance > 0 else None
    luck_index = 100.0 * _standard_normal_cdf(luck_z) if luck_z is not None else None

    sum_var_sq = sum(v * v for v in variance_by_market.values())
    effective_markets = (total_variance * total_variance / sum_var_sq) if sum_var_sq > 0 else 0.0

    if effective_markets < minimum_effective_markets:
        warnings.append(
            "EffectiveMarkets below threshold; interpret LuckIndex cautiously "
            f"({effective_markets:.2f} < {minimum_effective_markets:.2f})"
        )

    skill_roi = (skill_total / gross_notional) if gross_notional > 0 else None
    skill_randomized_sd = math.sqrt(skill_randomized_variance) if skill_randomized_variance > 0 else 0.0
    skill_z = (skill_total / skill_randomized_sd) if skill_randomized_sd > 0 else None
    skill_index = 100.0 * _standard_normal_cdf(skill_z) if skill_z is not None else None

    if skill_randomized_sd > 0:
        warnings.append(
            "SkillIndex uses a randomized-side proxy baseline; "
            "treat as approximate unless replaced by matched-trade bootstrap"
        )

    return LuckSkillResult(
        luck_index=luck_index,
        luck_z=luck_z,
        skill_index=skill_index,
        skill_z=skill_z,
        skill_randomized_sd_usdc=skill_randomized_sd,
        total_luck_usdc=luck_total,
        total_variance=total_variance,
        effective_markets=effective_markets,
        number_of_resolved_markets=len(variance_by_market),
        skill_usdc=skill_total,
        skill_roi=skill_roi,
        raw_pnl_usdc=pnl_total,
        gross_trade_notional_usdc=gross_notional,
        trade_decompositions=trade_decompositions,
        market_breakdowns=sorted(market_breakdowns, key=lambda m: m.condition_id),
        warnings=warnings,
    )


def _extract_market_outcome_data(
    market: Dict[str, Any],
    *,
    min_probability: float,
    max_probability: float,
    probability_calibrator: Optional[ProbabilityCalibrator],
) -> Optional[Dict[str, Dict[str, float]]]:
    outcome_keys = _market_outcome_keys(market)
    if not outcome_keys:
        return None

    winner_key = _winner_key(market, outcome_keys)
    if winner_key is None:
        return None

    raw_probs = _market_probabilities(market, outcome_keys)
    if not raw_probs:
        return None

    clipped = {
        key: _clip_probability(raw_probs.get(key, 0.0), min_probability, max_probability)
        for key in outcome_keys
    }

    if probability_calibrator is not None:
        clipped = {
            key: _clip_probability(probability_calibrator(prob), min_probability, max_probability)
            for key, prob in clipped.items()
        }

    probabilities = _normalize_probabilities(clipped)
    realized = {key: 1.0 if key == winner_key else 0.0 for key in outcome_keys}
    return {
        "outcome_keys": {key: 1.0 for key in outcome_keys},
        "probabilities": probabilities,
        "realized": realized,
    }


def _trade_id(trade: Dict[str, Any], idx: int) -> str:
    for key in ("id", "_trade_uid", "transactionHash"):
        value = trade.get(key)
        if value is not None and str(value):
            return str(value)
    return f"trade:{idx}"


def _trade_outcome_key(trade: Dict[str, Any], market: Dict[str, Any], valid_keys: Iterable[str]) -> str:
    valid_set = set(valid_keys)

    asset = str(trade.get("asset") or "")
    if asset and asset in valid_set:
        return asset

    outcome_index = trade.get("outcomeIndex")
    if outcome_index is not None:
        key = f"idx:{int(float(outcome_index))}"
        if key in valid_set:
            return key

    outcome = str(trade.get("outcome") or "").strip()
    if outcome:
        key = f"name:{outcome.lower()}"
        if key in valid_set:
            return key

    # Fallback to market outcome mapping by known token list.
    clob_ids = market.get("clobTokenIds")
    if isinstance(clob_ids, list) and asset:
        for i, token in enumerate(clob_ids):
            if str(token) == asset:
                key = f"idx:{i}"
                if key in valid_set:
                    return key

    return ""


def _market_outcome_keys(market: Dict[str, Any]) -> List[str]:
    clob_ids = market.get("clobTokenIds")
    if isinstance(clob_ids, list) and clob_ids:
        return [str(token) for token in clob_ids if str(token)]

    outcomes = market.get("outcomes")
    if isinstance(outcomes, list) and outcomes:
        return [f"name:{str(name).strip().lower()}" for name in outcomes]

    outcome_prices = market.get("outcomePrices")
    if isinstance(outcome_prices, list) and outcome_prices:
        return [f"idx:{i}" for i in range(len(outcome_prices))]

    return []


def _winner_key(market: Dict[str, Any], outcome_keys: Sequence[str]) -> Optional[str]:
    direct_index = market.get("winningOutcomeIndex")
    if direct_index is None:
        direct_index = market.get("winning_outcome_index")
    if direct_index is not None:
        idx = int(float(direct_index))
        key = _index_key_for_market(outcome_keys, idx)
        if key:
            return key

    winner = market.get("winner")
    if winner is not None:
        winner_text = str(winner).strip().lower()
        for key in outcome_keys:
            if key == winner_text or key == f"name:{winner_text}":
                return key

    outcomes = market.get("outcomes")
    if isinstance(outcomes, list) and winner is not None:
        winner_text = str(winner).strip().lower()
        for i, name in enumerate(outcomes):
            if str(name).strip().lower() == winner_text:
                key = _index_key_for_market(outcome_keys, i)
                if key:
                    return key

    # For closed/resolved markets: infer winner from outcomePrices if they show settlement (0 and 1)
    # This handles markets where the API doesn't provide winningOutcomeIndex
    if market.get("closed") or market.get("umaResolutionStatus") == "resolved":
        settlement_prices = market.get("outcomePrices")
        if isinstance(settlement_prices, list) and len(settlement_prices) == len(outcome_keys):
            try:
                # Convert to floats
                prices = [float(p) for p in settlement_prices]
                # Check if one price is 1.0 and the rest are 0.0 (settlement pattern)
                if sum(1 for p in prices if abs(p - 1.0) < 0.01) == 1:
                    winner_idx = next(i for i, p in enumerate(prices) if abs(p - 1.0) < 0.01)
                    key = _index_key_for_market(outcome_keys, winner_idx)
                    if key:
                        return key
            except (ValueError, TypeError, StopIteration):
                pass

    return None


def _index_key_for_market(outcome_keys: Sequence[str], idx: int) -> Optional[str]:
    if idx < 0:
        return None
    for key in outcome_keys:
        if key.startswith("idx:") and key == f"idx:{idx}":
            return key

    # If keys are clob token IDs or outcome names, index maps by position.
    if idx < len(outcome_keys):
        return outcome_keys[idx]
    return None


def _market_probabilities(market: Dict[str, Any], outcome_keys: Sequence[str]) -> Dict[str, float]:
    for field in ("closingOutcomePrices", "outcomePrices", "closing_probabilities"):
        values = market.get(field)
        if isinstance(values, list) and values:
            probs: Dict[str, float] = {}
            for i, key in enumerate(outcome_keys):
                if i >= len(values):
                    break
                p = safe_float(values[i])
                if p is None:
                    continue
                probs[key] = p
            if probs:
                return probs

        if isinstance(values, dict) and values:
            probs = {}
            for key in outcome_keys:
                p = safe_float(values.get(key))
                if p is not None:
                    probs[key] = p
            if probs:
                return probs

    return {}


def _clip_probability(value: float, min_probability: float, max_probability: float) -> float:
    if max_probability <= min_probability:
        raise ValueError("max_probability must be greater than min_probability")
    return max(min_probability, min(max_probability, value))


def _normalize_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        n = len(probabilities)
        if n == 0:
            return {}
        return {key: 1.0 / n for key in probabilities}
    return {key: value / total for key, value in probabilities.items()}


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def hydrate_missing_users(
    user_keys: Sequence[str],
    data_dir: Path,
    *,
    sleep_between_requests: float,
) -> Tuple[int, int]:
    """Fetch user data when the local database has no trades for that user."""
    hydrated = 0
    failed = 0
    cfg = AnalyzerConfig(sleep_between_requests=sleep_between_requests)

    for user_key in user_keys:
        if sqlite_user_has_trades(data_dir, user_key):
            continue

        print(f"Hydrating missing trades for {user_key}...", file=sys.stderr)
        try:
            download_for_user(
                user_key,
                cfg,
                data_dir,
                skip_price_history=True,
                price_fidelity_minutes=60,
                max_price_window_days=7,
                force_trades_refresh=False,
                force_market_refresh=False,
                force_price_refresh=False,
            )
            hydrated += 1
        except Exception as exc:
            failed += 1
            print(f"  Failed hydrating {user_key}: {exc}", file=sys.stderr)

    return hydrated, failed


def load_markets(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all market metadata from SQLite."""
    return sqlite_load_all_market_rows(data_dir)


def load_price_history(data_dir: Path) -> Dict[str, List[tuple[int, float]]]:
    """Load all price history records organized by asset from SQLite."""
    points = sqlite_load_price_history_points(data_dir)
    return {asset: [(int(ts), px) for ts, px in rows] for asset, rows in points.items()}


def parse_market_row(
    row: Dict[str, Any],
    price_history: Dict[str, List[tuple[int, float]]],
) -> Dict[str, Any]:
    """Normalize market row fields for luck_skill_analysis."""
    import json

    def try_parse_json(value: Any) -> Any:
        if not value or not isinstance(value, str):
            return value
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return value

    def to_int_or_none(value: Any) -> int | None:
        if not value:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return None

    condition_id = row.get("conditionId")
    closed_str = row.get("closed") or ""
    is_closed = closed_str == "True" or closed_str == "true"
    clob_token_ids = try_parse_json(row.get("clobTokenIds"))
    outcomes = try_parse_json(row.get("outcomes"))
    outcome_prices = try_parse_json(row.get("outcomePrices"))

    closing_prices = None
    if is_closed and clob_token_ids and len(clob_token_ids) > 0:
        asset_last_prices = {}
        for asset in clob_token_ids:
            asset_str = str(asset)
            prices = price_history.get(asset_str)
            if prices:
                asset_last_prices[asset_str] = prices[-1][1]

        if len(asset_last_prices) == len(clob_token_ids):
            closing_prices = [asset_last_prices.get(str(asset), 0.5) for asset in clob_token_ids]

    return {
        "conditionId": condition_id,
        "question": row.get("question"),
        "slug": row.get("slug"),
        "eventSlug": row.get("eventSlug"),
        "closed": is_closed,
        "active": row.get("active") == "True" or row.get("active") == "true",
        "closedTime": row.get("closedTime"),
        "endDate": row.get("endDate"),
        "winner": row.get("winner"),
        "winningOutcomeIndex": to_int_or_none(row.get("winningOutcomeIndex")),
        "outcomes": outcomes,
        "outcomePrices": outcome_prices,
        "closingOutcomePrices": closing_prices,
        "clobTokenIds": clob_token_ids,
        "liquidityNum": row.get("liquidityNum"),
        "volumeNum": row.get("volumeNum"),
        "volume24hr": row.get("volume24hr"),
        "umaResolutionStatus": row.get("umaResolutionStatus"),
    }


def parse_trade_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize trade row fields for luck_skill_analysis."""

    def to_float(value: Any) -> float | None:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    return {
        "id": row.get("id"),
        "conditionId": row.get("conditionId"),
        "asset": row.get("asset"),
        "side": row.get("side"),
        "size": to_float(row.get("size")),
        "price": to_float(row.get("price")),
        "timestamp": row.get("timestamp"),
        "outcome": row.get("outcome"),
        "outcomeIndex": row.get("outcomeIndex"),
        "transactionHash": row.get("transactionHash"),
    }


def analyze_user(
    user_key: str,
    data_dir: Path,
    markets_parsed: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Analyze a single user and save metrics to SQLite."""
    print(f"Analyzing {user_key}...", file=sys.stderr)

    trades_raw = sqlite_load_all_user_trades(data_dir, user_key)
    if not trades_raw:
        print(f"  No trades found for {user_key}", file=sys.stderr)
        return None

    trades = [parse_trade_row(t) for t in trades_raw]
    print(f"  Loaded {len(trades)} trades", file=sys.stderr)

    try:
        result = analyze_luck_and_skill(trades, markets_parsed, minimum_effective_markets=0)
        print(f"  Analyzed {result.number_of_resolved_markets} resolved markets", file=sys.stderr)
    except Exception as exc:
        print(f"  Analysis failed for {user_key}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None

    pnl_by_market: Dict[str, float] = defaultdict(float)
    for trade in result.trade_decompositions:
        pnl_by_market[trade.condition_id] += trade.pnl_usdc

    resolved_trade_count = len(result.trade_decompositions)
    profitable_markets = sum(1 for pnl in pnl_by_market.values() if pnl > 0)
    losing_markets = sum(1 for pnl in pnl_by_market.values() if pnl < 0)
    pnl_market_count = profitable_markets + losing_markets
    profitable_market_ratio = (
        profitable_markets / pnl_market_count
        if pnl_market_count > 0
        else None
    )
    largest_market_profit = max(pnl_by_market.values(), default=0.0)
    largest_market_loss = min(pnl_by_market.values(), default=0.0)
    pnl_per_trade = result.raw_pnl_usdc / len(trades) if trades else None
    pnl_per_resolved_trade = (
        result.raw_pnl_usdc / resolved_trade_count
        if resolved_trade_count > 0
        else None
    )
    skill_per_trade = result.skill_usdc / len(trades) if trades else None
    skill_per_resolved_trade = (
        result.skill_usdc / resolved_trade_count
        if resolved_trade_count > 0
        else None
    )
    if pnl_per_trade is not None and skill_per_trade is not None:
        monetized_edge_per_trade = min(pnl_per_trade, skill_per_trade)
    else:
        monetized_edge_per_trade = pnl_per_trade if pnl_per_trade is not None else skill_per_trade

    output_row = {
        "user_key": user_key,
        "total_trades": len(trades),
        "luck_index": result.luck_index,
        "luck_z": result.luck_z,
        "skill_index": result.skill_index,
        "skill_z": result.skill_z,
        "skill_randomized_sd_usdc": result.skill_randomized_sd_usdc,
        "total_luck_usdc": result.total_luck_usdc,
        "total_variance": result.total_variance,
        "effective_markets": result.effective_markets,
        "number_of_resolved_markets": result.number_of_resolved_markets,
        "skill_usdc": result.skill_usdc,
        "skill_roi": result.skill_roi,
        "raw_pnl_usdc": result.raw_pnl_usdc,
        "gross_trade_notional_usdc": result.gross_trade_notional_usdc,
        "resolved_trade_count": resolved_trade_count,
        "pnl_per_trade_usdc": pnl_per_trade,
        "pnl_per_resolved_trade_usdc": pnl_per_resolved_trade,
        "skill_per_trade_usdc": skill_per_trade,
        "skill_per_resolved_trade_usdc": skill_per_resolved_trade,
        "monetized_edge_per_trade_usdc": monetized_edge_per_trade,
        "profitable_resolved_markets": profitable_markets,
        "losing_resolved_markets": losing_markets,
        "profitable_market_ratio": profitable_market_ratio,
        "largest_market_profit_usdc": largest_market_profit,
        "largest_market_loss_usdc": largest_market_loss,
        "warning_count": len(result.warnings),
    }

    sqlite_upsert_user_metric(data_dir, output_row, warnings=result.warnings)
    print("  Saved metrics to SQLite", file=sys.stderr)

    return output_row


def run_all_users_analysis(
    data_dir: Path | None = None,
    reports_dir: Path | None = None,
    *,
    users: Sequence[str] | None = None,
    hydrate_missing: bool = False,
    sleep_between_requests: float = 0.05,
    use_price_history: bool = False,
) -> List[Dict[str, Any]]:
    """Run luck/skill analysis for users stored in SQLite."""
    data_dir = data_dir or Path("data")
    reports_dir = reports_dir or (Path("reports") / "luck_skill")
    ensure_database(data_dir)

    if not data_dir.exists():
        print("ERROR: ./data directory not found. Run update commands first.", file=sys.stderr)
        return []

    requested_users = {str(user or "").strip() for user in (users or []) if str(user or "").strip()}
    user_keys = sqlite_list_user_keys(data_dir, include_candidates=False)
    missing_requested_users: List[str] = []
    if requested_users:
        profiles = sqlite_load_user_profile_rows(data_dir)
        aliases = sqlite_load_user_alias_rows(data_dir)
        aliases_by_user = {str(row.get("user_key") or ""): row for row in aliases.values() if row.get("user_key")}
        user_keys = [
            key
            for key in user_keys
            if user_matches_requested(key, profiles.get(key, {}), requested_users, aliases_by_user.get(key))
        ]
        matched_inputs = {
            requested
            for requested in requested_users
            if any(
                user_matches_requested(key, profiles.get(key, {}), {requested}, aliases_by_user.get(key))
                for key in user_keys
            )
        }
        missing_requested_users = sorted(requested_users - matched_inputs)
        for user_key in missing_requested_users:
            print(f"WARNING: requested user has no SQLite data: {user_key}", file=sys.stderr)

    hydrate_targets = sorted(set(user_keys) | set(missing_requested_users)) if hydrate_missing else []
    if hydrate_targets:
        hydrated, failed = hydrate_missing_users(
            hydrate_targets,
            data_dir,
            sleep_between_requests=sleep_between_requests,
        )
        print(
            f"Hydration summary: hydrated={hydrated}, failed={failed}, total_users={len(hydrate_targets)}",
            file=sys.stderr,
        )
        user_keys = sqlite_list_user_keys(data_dir, include_candidates=False)
        if requested_users:
            profiles = sqlite_load_user_profile_rows(data_dir)
            aliases = sqlite_load_user_alias_rows(data_dir)
            aliases_by_user = {str(row.get("user_key") or ""): row for row in aliases.values() if row.get("user_key")}
            user_keys = [
                key
                for key in user_keys
                if user_matches_requested(key, profiles.get(key, {}), requested_users, aliases_by_user.get(key))
            ]
    elif user_keys:
        print(
            "Hydration skipped; luck/skill analysis is offline by default. "
            "Pass --hydrate-missing-users to download missing trades first.",
            file=sys.stderr,
        )

    print("Loading market data...", file=sys.stderr)
    markets = load_markets(data_dir)
    print(f"Loaded {len(markets)} markets", file=sys.stderr)

    price_history: Dict[str, List[tuple[int, float]]] = {}
    if use_price_history:
        print("Loading price history...", file=sys.stderr)
        price_history = load_price_history(data_dir)
        print(f"Loaded price history for {len(price_history)} assets", file=sys.stderr)
    else:
        print(
            "Skipping price history for fast offline analysis. "
            "Pass --use-price-history for the slower settlement-price fallback.",
            file=sys.stderr,
        )

    print("Parsing market metadata...", file=sys.stderr)
    markets_parsed = {cid: parse_market_row(m, price_history) for cid, m in markets.items()}
    print(f"Parsed {len(markets_parsed)} markets", file=sys.stderr)

    print(f"Found {len(user_keys)} user(s) to analyze", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []
    for user_key in user_keys:
        try:
            row = analyze_user(user_key, data_dir, markets_parsed)
            if row is not None:
                all_rows.append(row)
                update_watermark(
                    data_dir,
                    "analysis_users",
                    user_key,
                    {
                        "reports_dir": str(reports_dir),
                        "total_trades": row.get("total_trades"),
                        "resolved_markets": row.get("number_of_resolved_markets"),
                        "skill_index": row.get("skill_index"),
                    },
                )
        except Exception as exc:
            print(f"ERROR analyzing {user_key}: {exc}", file=sys.stderr)

    if all_rows:
        print(f"Saved {len(all_rows)} user metric row(s) to SQLite", file=sys.stderr)

    update_watermark(
        data_dir,
        "analysis",
        "luck_skill",
        {
            "reports_dir": str(reports_dir),
            "users_found": len(user_keys),
            "users_analyzed": len(all_rows),
            "requested_users": sorted(requested_users),
            "hydrate_missing": bool(hydrate_missing),
            "use_price_history": bool(use_price_history),
        },
    )

    print("Done.", file=sys.stderr)
    return all_rows


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
    seen = set()

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


def user_matches_requested(
    user_key: str,
    profile: Dict[str, Any],
    requested_users: set[str],
    alias: Dict[str, Any] | None = None,
) -> bool:
    alias = alias or {}
    requested = {value.lower().lstrip("@") for value in requested_users}
    candidates = {
        str(user_key or "").lower().lstrip("@"),
        str(profile.get("input_user") or "").lower().lstrip("@"),
        str(profile.get("name") or "").lower().lstrip("@"),
        str(profile.get("pseudonym") or "").lower().lstrip("@"),
        str(profile.get("profile_name") or "").lower().lstrip("@"),
        str(profile.get("profile_pseudonym") or "").lower().lstrip("@"),
        str(alias.get("alias") or "").lower().lstrip("@"),
        str(alias.get("display_name") or "").lower().lstrip("@"),
    }
    return bool(requested & {value for value in candidates if value})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run luck/skill analysis from the local database.")
    parser.add_argument("--data-dir", default="data", help="Data directory root (default: data)")
    parser.add_argument(
        "--reports-dir",
        default="reports/luck_skill",
        help="Output directory for luck/skill reports (default: reports/luck_skill)",
    )
    parser.add_argument(
        "--user",
        action="append",
        help="Analyze only this user key. Repeat to analyze more than one user.",
    )
    parser.add_argument(
        "--users-file",
        action="append",
        help="Text file containing user keys, one per line.",
    )
    parser.add_argument(
        "--hydrate-missing-users",
        action="store_true",
        help="Download missing user trade data before analyzing. Analysis is offline by default.",
    )
    parser.add_argument(
        "--no-hydrate-missing-users",
        action="store_true",
        help="Deprecated compatibility flag; hydration is already disabled by default.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between API calls when hydrating missing users.",
    )
    parser.add_argument(
        "--use-price-history",
        action="store_true",
        help=(
            "Load stored price-history rows as a fallback for settlement prices. "
            "This can be slow on large databases and is disabled by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all_users_analysis(
        data_dir=Path(args.data_dir),
        reports_dir=Path(args.reports_dir),
        users=collect_requested_users(args.user, args.users_file),
        hydrate_missing=args.hydrate_missing_users and not args.no_hydrate_missing_users,
        sleep_between_requests=args.sleep,
        use_price_history=args.use_price_history,
    )


if __name__ == "__main__":
    main()
