from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_user(user: str) -> str:
    return user.strip().lstrip("@")


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_iso_utc(ts: Any) -> str:
    f = safe_float(ts)
    if f is None:
        return "" if ts is None else str(ts)
    if f > 10_000_000_000:
        f /= 1000.0
    return datetime.fromtimestamp(f, tz=timezone.utc).isoformat()


def unix_now() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def parse_jsonish_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def weighted_mean(pairs: Iterable[Tuple[Optional[float], Optional[float]]]) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in pairs:
        if value is None or weight is None:
            continue
        if weight <= 0:
            continue
        total_value += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return total_value / total_weight


def z_score(total: float, variance: float) -> Optional[float]:
    if variance <= 0:
        return None
    return total / math.sqrt(variance)


def parse_duration_to_seconds(text: str) -> int:
    s = text.strip().lower()
    if not s:
        raise ValueError("Empty duration")
    multiplier = 1
    if s.endswith("h"):
        multiplier = 3600
        s = s[:-1]
    elif s.endswith("d"):
        multiplier = 86400
        s = s[:-1]
    elif s.endswith("m"):
        multiplier = 60
        s = s[:-1]
    elif s.endswith("s"):
        multiplier = 1
        s = s[:-1]
    return int(float(s) * multiplier)


def duration_label(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def make_position_key(obj: Dict[str, Any]) -> Tuple[str, str]:
    condition_id = str(obj.get("conditionId") or "")
    asset = str(obj.get("asset") or "")
    if asset:
        return condition_id, asset
    outcome_index = obj.get("outcomeIndex")
    if outcome_index is not None:
        return condition_id, f"outcomeIndex:{outcome_index}"
    outcome = str(obj.get("outcome") or "")
    return condition_id, f"outcome:{outcome}"


def chunks(items: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def stable_trade_uid(trade: Dict[str, Any], index: int) -> str:
    pieces = [
        str(trade.get("transactionHash") or ""),
        str(trade.get("conditionId") or ""),
        str(trade.get("asset") or ""),
        str(trade.get("side") or ""),
        str(trade.get("timestamp") or ""),
        str(trade.get("price") or ""),
        str(trade.get("size") or ""),
        str(index),
    ]
    return "|".join(pieces)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False
