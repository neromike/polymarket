from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from cli import write_csv
from config import AnalyzerConfig
from download_data import download_for_user
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

        outcome_info = _extract_market_outcome_data(
            market,
            min_probability=min_probability,
            max_probability=max_probability,
            probability_calibrator=probability_calibrator,
        )
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

        outcome_info = _extract_market_outcome_data(
            market,
            min_probability=min_probability,
            max_probability=max_probability,
            probability_calibrator=probability_calibrator,
        )
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


def load_trades_for_user(user_dir: Path) -> List[Dict[str, Any]]:
    """Load all trades for a user across all months."""
    trades_root = user_dir / "trades"
    if not trades_root.exists():
        return []

    trades: List[Dict[str, Any]] = []
    for month_dir in sorted(trades_root.iterdir()):
        if not month_dir.is_dir():
            continue
        for csv_file in sorted(month_dir.glob("trade_*.csv")):
            try:
                with csv_file.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append(dict(row))
            except OSError as exc:
                print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)
    return trades


def hydrate_missing_users(
    user_dirs: Sequence[Path],
    data_dir: Path,
    *,
    sleep_between_requests: float,
) -> Tuple[int, int]:
    """Fetch user data for folders that do not yet contain trades."""
    hydrated = 0
    failed = 0
    cfg = AnalyzerConfig(sleep_between_requests=sleep_between_requests)

    for user_dir in user_dirs:
        user_key = user_dir.name
        if load_trades_for_user(user_dir):
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
    """Load all market metadata across all months."""
    markets_root = data_dir / "market" / "markets"
    if not markets_root.exists():
        return {}

    markets: Dict[str, Dict[str, Any]] = {}
    for month_dir in sorted(markets_root.iterdir()):
        if not month_dir.is_dir():
            continue
        for csv_file in sorted(month_dir.glob("market_*.csv")):
            try:
                with csv_file.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        condition_id = row.get("conditionId")
                        if condition_id:
                            markets[condition_id] = dict(row)
            except OSError as exc:
                print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)
    return markets


def load_price_history(data_dir: Path) -> Dict[str, List[tuple[int, float]]]:
    """Load all price history records organized by asset. Returns {asset: [(timestamp, price), ...]}"""
    prices_root = data_dir / "market" / "prices-history"
    if not prices_root.exists():
        return {}

    prices_by_asset: Dict[str, List[tuple[int, float]]] = defaultdict(list)
    for month_dir in sorted(prices_root.iterdir()):
        if not month_dir.is_dir():
            continue
        for csv_file in sorted(month_dir.glob("price_history_asset_*.csv")):
            try:
                with csv_file.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        asset = row.get("asset")
                        ts = row.get("timestamp")
                        price_str = row.get("price")
                        if asset and ts and price_str:
                            try:
                                ts_int = int(float(ts))
                                price = float(price_str)
                                prices_by_asset[asset].append((ts_int, price))
                            except (ValueError, TypeError):
                                continue
            except OSError as exc:
                print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)

    for asset in prices_by_asset:
        prices_by_asset[asset].sort(key=lambda x: x[0])

    return prices_by_asset


def parse_market_row(
    row: Dict[str, Any],
    price_history: Dict[str, List[tuple[int, float]]],
) -> Dict[str, Any]:
    """Convert CSV market row fields to proper types for luck_skill_analysis."""
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
    """Convert CSV trade row fields to proper types for luck_skill_analysis."""

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
    user_dir: Path,
    report_user_dir: Path,
    markets: Dict[str, Dict[str, Any]],
    price_history: Dict[str, List[tuple[int, float]]],
) -> Dict[str, Any] | None:
    """Analyze a single user and save metrics to CSV under reports/luck_skill/users."""
    print(f"Analyzing {user_key}...", file=sys.stderr)

    trades_raw = load_trades_for_user(user_dir)
    if not trades_raw:
        print(f"  No trades found for {user_key}", file=sys.stderr)
        return None

    trades = [parse_trade_row(t) for t in trades_raw]
    print(f"  Loaded {len(trades)} trades", file=sys.stderr)

    markets_parsed = {cid: parse_market_row(m, price_history) for cid, m in markets.items()}

    try:
        result = analyze_luck_and_skill(trades, markets_parsed, minimum_effective_markets=0)
        print(f"  Analyzed {result.number_of_resolved_markets} resolved markets", file=sys.stderr)
    except Exception as exc:
        print(f"  Analysis failed for {user_key}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None

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
        "warning_count": len(result.warnings),
    }

    report_user_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_user_dir / "metrics.csv"
    write_csv(metrics_path, [output_row])
    print(f"  Saved metrics to {metrics_path}", file=sys.stderr)

    if result.warnings:
        warnings_path = report_user_dir / "warnings.txt"
        with warnings_path.open("w", encoding="utf-8") as f:
            for warning in result.warnings:
                f.write(f"{warning}\n")
        print(f"  Saved {len(result.warnings)} warnings to {warnings_path}", file=sys.stderr)

    return output_row


def run_all_users_analysis(
    data_dir: Path | None = None,
    reports_dir: Path | None = None,
    *,
    hydrate_missing: bool = True,
    sleep_between_requests: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run luck/skill analysis for all users found under data/user and write reports outputs."""
    data_dir = data_dir or Path("data")
    reports_dir = reports_dir or (Path("reports") / "luck_skill")
    report_users_dir = reports_dir / "users"

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_users_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print("ERROR: ./data directory not found. Run download_luck_skill_data.py first.", file=sys.stderr)
        return []

    users_dir = data_dir / "user"
    if not users_dir.exists():
        print("ERROR: ./data/user directory not found.", file=sys.stderr)
        return []

    user_dirs = sorted([d for d in users_dir.iterdir() if d.is_dir()])
    if hydrate_missing and user_dirs:
        hydrated, failed = hydrate_missing_users(
            user_dirs,
            data_dir,
            sleep_between_requests=sleep_between_requests,
        )
        print(
            f"Hydration summary: hydrated={hydrated}, failed={failed}, total_users={len(user_dirs)}",
            file=sys.stderr,
        )

    print("Loading market data...", file=sys.stderr)
    markets = load_markets(data_dir)
    print(f"Loaded {len(markets)} markets", file=sys.stderr)

    print("Loading price history...", file=sys.stderr)
    price_history = load_price_history(data_dir)
    print(f"Loaded price history for {len(price_history)} assets", file=sys.stderr)

    user_dirs = sorted([d for d in users_dir.iterdir() if d.is_dir()])
    print(f"Found {len(user_dirs)} user(s) to analyze", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []
    for user_dir in user_dirs:
        user_key = user_dir.name
        try:
            report_user_dir = report_users_dir / user_key
            row = analyze_user(user_key, user_dir, report_user_dir, markets, price_history)
            if row is not None:
                all_rows.append(row)
        except Exception as exc:
            print(f"ERROR analyzing {user_key}: {exc}", file=sys.stderr)

    if all_rows:
        all_metrics_path = reports_dir / "metrics_all_users.csv"
        write_csv(all_metrics_path, all_rows)
        print(f"Saved aggregate metrics to {all_metrics_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run luck/skill analysis for users under data/user, hydrating missing trade data by default."
    )
    parser.add_argument("--data-dir", default="data", help="Data directory root (default: data)")
    parser.add_argument(
        "--reports-dir",
        default="reports/luck_skill",
        help="Output directory for luck/skill reports (default: reports/luck_skill)",
    )
    parser.add_argument(
        "--no-hydrate-missing-users",
        action="store_true",
        help="Do not auto-download data for users with empty trade folders.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between API calls when hydrating missing users.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all_users_analysis(
        data_dir=Path(args.data_dir),
        reports_dir=Path(args.reports_dir),
        hydrate_missing=not args.no_hydrate_missing_users,
        sleep_between_requests=args.sleep,
    )


if __name__ == "__main__":
    main()
