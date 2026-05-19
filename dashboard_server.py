from __future__ import annotations

import html
import json
import math
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from config import CONDITION_RE
from runtime_state import (
    atomic_write_json,
    build_inventory,
    create_job,
    jobs_dir,
    list_jobs,
    list_latest_runs,
    load_watermarks,
    process_is_running,
    read_json,
    run_state_dir,
    update_job,
    utc_now_compact,
    utc_now_iso,
)
from sqlite_store import (
    count_market_rows as sqlite_count_market_rows,
    default_db_path,
    load_market_statuses as sqlite_load_market_statuses,
    load_market_raw_row as sqlite_load_market_raw_row,
    load_market_summary_rows as sqlite_load_market_summary_rows,
    market_condition_ids as sqlite_market_condition_ids,
    load_scanner_candidate_rows as sqlite_load_scanner_candidate_rows,
    load_scanner_jump_event_rows as sqlite_load_scanner_jump_event_rows,
    load_user_metric_rows as sqlite_load_user_metric_rows,
    load_user_alias_rows as sqlite_load_user_alias_rows,
    load_user_profile_row as sqlite_load_user_profile_row,
    load_user_profile_rows as sqlite_load_user_profile_rows,
    load_user_scanner_market_signals as sqlite_load_user_scanner_market_signals,
    load_user_trade_rows as sqlite_load_user_trade_rows,
    list_user_keys as sqlite_list_user_keys,
    sqlite_database_available,
)
from utils import parse_jsonish_list, safe_float, safe_int, to_iso_utc


ROOT = Path(__file__).resolve().parent
USER_LIST_CANDIDATE_CONFIDENCES = {"monitor", "candidate", "high", "very_high"}
WALLET_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def app_command(*parts: str) -> List[str]:
    return [sys.executable, str(ROOT / "app.py"), *parts]


JOB_COMMANDS: Dict[str, Tuple[str, List[str]]] = {
    "update_markets": ("Discover markets", app_command("update", "markets")),
    "update_users_quick": (
        "Update users and skills without price history",
        app_command("update", "users", "--skip-price-history", "--analyze-after"),
    ),
    "update_users_full": ("Update users and skills with price history", app_command("update", "users", "--analyze-after")),
    "update_market_trades": (
        "Update market trades",
        app_command("update", "market-trades"),
    ),
    "analyze_luck": ("Analyze luck/skill", app_command("analyze", "luck")),
    "analyze_scanner": ("Analyze markets", app_command("analyze", "scanner")),
}


def normalize_condition_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if CONDITION_RE.match(text) else ""


def market_list_path(root: Path | str = ROOT) -> Path:
    return run_state_dir(root) / "market_list.json"


def legacy_market_list_path(root: Path | str = ROOT) -> Path:
    return run_state_dir(root) / "market_watchlist.json"


def clean_market_entries(entries: Any) -> List[Dict[str, Any]]:
    if not isinstance(entries, list):
        entries = []
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        raw_id = entry.get("condition_id") if isinstance(entry, dict) else entry
        condition_id = normalize_condition_id(raw_id)
        if not condition_id or condition_id in seen:
            continue
        seen.add(condition_id)
        cleaned.append(
            {
                "condition_id": condition_id,
                "added_at": str(entry.get("added_at") or "") if isinstance(entry, dict) else "",
                "source": str(entry.get("source") or "manual") if isinstance(entry, dict) else "manual",
            }
        )
    return cleaned


def load_market_list(root: Path | str = ROOT) -> Dict[str, Any]:
    path = market_list_path(root)
    legacy_path = legacy_market_list_path(root)
    payload = read_json(path, default={"markets": []})
    entries = clean_market_entries(payload.get("markets"))

    if legacy_path.exists():
        seen = {entry["condition_id"] for entry in entries}
        legacy_payload = read_json(legacy_path, default={"markets": []})
        for entry in clean_market_entries(legacy_payload.get("markets")):
            if entry["condition_id"] in seen:
                continue
            seen.add(entry["condition_id"])
            entries.append(entry)
        if not path.exists() and entries:
            atomic_write_json(path, {"markets": entries})

    return {"markets": entries}


def save_market_list(payload: Dict[str, Any], root: Path | str = ROOT) -> None:
    atomic_write_json(market_list_path(root), payload)


def add_market_to_list(condition_id: str, *, source: str = "manual", root: Path | str = ROOT) -> Dict[str, Any]:
    normalized = normalize_condition_id(condition_id)
    if not normalized:
        raise ValueError("market id must be a 0x-prefixed conditionId")

    payload = load_market_list(root)
    entries = payload.setdefault("markets", [])
    for entry in entries:
        if entry.get("condition_id") == normalized:
            entry["source"] = entry.get("source") or source
            save_market_list(payload, root)
            return entry

    entry = {"condition_id": normalized, "added_at": utc_now_iso(), "source": source}
    entries.append(entry)
    save_market_list(payload, root)
    return entry


def fmt_float(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def sort_by_float(rows: Iterable[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: safe_float(row.get(key)) if safe_float(row.get(key)) is not None else -1e99, reverse=reverse)


def clamp_score(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_tier(score: float | None) -> str:
    if score is None:
        return "Unscored"
    if score >= 85:
        return "Strong signal"
    if score >= 70:
        return "Elevated"
    if score >= 55:
        return "Watch"
    return "Baseline"


def candidate_signal_score(row: Dict[str, Any] | None) -> float | None:
    if not row:
        return None
    confidence = str(row.get("confidence_level") or row.get("candidate_confidence") or "").strip().lower()
    confidence_base = {"monitor": 45.0, "candidate": 62.0, "high": 78.0, "very_high": 92.0}.get(confidence)
    timing_z = safe_float(row.get("timing_z")) if "timing_z" in row else safe_float(row.get("candidate_timing_z"))
    events = safe_int(row.get("number_of_independent_events")) if "number_of_independent_events" in row else safe_int(row.get("candidate_events"))
    capture = (
        safe_float(row.get("cluster_capped_total_jump_capture"))
        if "cluster_capped_total_jump_capture" in row
        else safe_float(row.get("candidate_jump_capture"))
    )
    if capture is None:
        capture = safe_float(row.get("total_jump_capture"))
    gross_notional = (
        safe_float(row.get("gross_notional"))
        if "gross_notional" in row
        else safe_float(row.get("candidate_gross_notional"))
    )
    directional_ratio = (
        safe_float(row.get("avg_directional_ratio"))
        if "avg_directional_ratio" in row
        else safe_float(row.get("candidate_directional_ratio"))
    )
    jumps_captured = (
        safe_int(row.get("number_of_jumps_captured"))
        if "number_of_jumps_captured" in row
        else safe_int(row.get("candidate_jumps_captured"))
    )
    if confidence_base is None and timing_z is None and events is None and capture is None:
        return None

    z_score = clamp_score(45.0 + (timing_z or 0.0) * 9.0)
    event_score = clamp_score((events or 0) * 10.0 + 25.0)
    capture_roi = (capture / gross_notional) if capture is not None and gross_notional and gross_notional > 0 else 0.0
    capture_score = clamp_score(45.0 + capture_roi * 900.0)
    direction_score = clamp_score(45.0 + (directional_ratio or 0.0) * 50.0)
    hit_score = clamp_score(35.0 + (jumps_captured or 0) * 8.0)
    if confidence_base is None:
        confidence_base = 50.0
    score = max(
        confidence_base,
        0.34 * z_score
        + 0.24 * event_score
        + 0.18 * capture_score
        + 0.14 * direction_score
        + 0.10 * hit_score,
    )
    if capture is not None and capture <= 0:
        score = min(score, 49.0)
    return clamp_score(score)


def market_jump_signal_score(move: Any, participants: Any = None) -> float:
    abs_move = abs(safe_float(move) or 0.0)
    participant_count = safe_int(participants) or 0
    return clamp_score(abs_move * 260.0 + min(participant_count, 60) * 0.75)


def metric_signal_score(row: Dict[str, Any]) -> float | None:
    skill_index = safe_float(row.get("skill_index"))
    skill_z = safe_float(row.get("skill_z"))
    luck_index = safe_float(row.get("luck_index"))
    raw_pnl = safe_float(row.get("raw_pnl_usdc"))
    skill_usdc = safe_float(row.get("skill_usdc"))
    skill_roi = safe_float(row.get("skill_roi"))
    gross_notional = safe_float(row.get("gross_trade_notional_usdc"))
    resolved = safe_int(row.get("resolved_markets") or row.get("number_of_resolved_markets")) or 0
    trades = safe_int(row.get("total_trades")) or 0
    profitable_market_ratio = safe_float(row.get("profitable_market_ratio"))
    edge_per_trade = monetized_edge_per_trade(row)

    if skill_index is None and skill_z is None and luck_index is None and raw_pnl is None and skill_usdc is None:
        return None

    skill_percentile = skill_index if skill_index is not None else clamp_score(50.0 + (skill_z or 0.0) * 16.0)
    skill_z_score = clamp_score(50.0 + (skill_z or 0.0) * 16.0)
    roi = (raw_pnl / gross_notional) if raw_pnl is not None and gross_notional and gross_notional > 0 else 0.0
    profit_roi_score = clamp_score(50.0 + roi * 700.0)
    skill_roi_score = clamp_score(50.0 + (skill_roi or 0.0) * 900.0)
    pnl_sign_score = 88.0 if raw_pnl is not None and raw_pnl > 0 else 25.0 if raw_pnl is not None and raw_pnl < 0 else 45.0
    profit_consistency_score = (
        clamp_score(profitable_market_ratio * 100.0)
        if profitable_market_ratio is not None
        else 50.0
    )
    if edge_per_trade is None:
        edge_per_trade_score = 50.0
    elif edge_per_trade <= 0:
        edge_per_trade_score = 25.0
    else:
        edge_per_trade_score = clamp_score(45.0 + math.log1p(edge_per_trade) * 12.0)
    sample_score = clamp_score(min(resolved, 25) * 2.4 + min(trades, 2000) / 2000 * 40.0)

    score = (
        0.22 * profit_roi_score
        + 0.18 * pnl_sign_score
        + 0.20 * skill_percentile
        + 0.12 * skill_z_score
        + 0.10 * skill_roi_score
        + 0.08 * profit_consistency_score
        + 0.10 * edge_per_trade_score
    )
    sample_multiplier = 0.80 + 0.20 * (sample_score / 100.0)
    score *= sample_multiplier

    if raw_pnl is not None and raw_pnl <= 0 and (skill_usdc is None or skill_usdc <= 0):
        score = min(score, 39.0)
    elif raw_pnl is not None and raw_pnl <= 0:
        score = min(score, 52.0)
    elif skill_usdc is not None and skill_usdc <= 0:
        score = min(score, 68.0)

    return clamp_score(score)


def per_trade_value(total: float | None, trades: int | None) -> float | None:
    if total is None or trades is None or trades <= 0:
        return None
    return total / trades


def monetized_edge_per_trade(row: Dict[str, Any]) -> float | None:
    explicit = safe_float(row.get("monetized_edge_per_trade_usdc"))
    if explicit is not None:
        return explicit

    trades = safe_int(row.get("total_trades"))
    pnl_per_trade = safe_float(row.get("pnl_per_trade_usdc"))
    if pnl_per_trade is None:
        pnl_per_trade = per_trade_value(safe_float(row.get("raw_pnl_usdc")), trades)

    skill_per_trade = safe_float(row.get("skill_per_trade_usdc"))
    if skill_per_trade is None:
        skill_per_trade = per_trade_value(safe_float(row.get("skill_usdc")), trades)

    if pnl_per_trade is not None and skill_per_trade is not None:
        return min(pnl_per_trade, skill_per_trade)
    return pnl_per_trade if pnl_per_trade is not None else skill_per_trade


def evidence_summary(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    confidence = str(row.get("candidate_confidence") or "").strip()
    timing_z = safe_float(row.get("candidate_timing_z"))
    candidate_events = safe_int(row.get("candidate_events"))
    jump_capture = safe_float(row.get("candidate_jump_capture"))
    jump_notional = safe_float(row.get("candidate_gross_notional"))
    if confidence:
        scanner = f"{confidence} timing"
        if timing_z is not None:
            scanner += f" z={timing_z:.2f}"
        if candidate_events is not None:
            scanner += f" across {candidate_events} events"
        if jump_capture is not None:
            scanner += f", jump capture ${jump_capture:,.0f}"
            if jump_notional:
                scanner += f" ({jump_capture / jump_notional:.1%})"
        parts.append(scanner)

    skill_index = safe_float(row.get("skill_index"))
    skill_z = safe_float(row.get("skill_z"))
    resolved = safe_int(row.get("resolved_markets"))
    if skill_index is not None:
        skill = f"skill pctl {skill_index:.0f}"
        if skill_z is not None:
            skill += f" (z={skill_z:.2f})"
        if resolved is not None:
            skill += f" over {resolved} resolved markets"
        parts.append(skill)

    pnl = safe_float(row.get("raw_pnl_usdc"))
    gross_notional = safe_float(row.get("gross_trade_notional_usdc"))
    if pnl is not None:
        pnl_text = f"PnL ${pnl:,.0f}"
        if gross_notional:
            pnl_text += f" ({pnl / gross_notional:.1%})"
        parts.append(pnl_text)

    skill_usdc = safe_float(row.get("skill_usdc"))
    if skill_usdc is not None:
        parts.append(f"expected edge ${skill_usdc:,.0f}")

    edge_per_trade = safe_float(row.get("monetized_edge_per_trade_usdc"))
    if edge_per_trade is not None:
        parts.append(f"info edge ${edge_per_trade:,.2f}/trade")

    cap_reason = str(row.get("score_cap_reason") or "").strip()
    if cap_reason:
        parts.append(cap_reason)

    return "; ".join(parts) if parts else "Needs user update"


def apply_information_edge_score(row: Dict[str, Any], candidate: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cand_score = candidate_signal_score(candidate or row)
    metric_score = metric_signal_score(row)
    has_metrics = bool(row.get("has_metrics"))
    raw_pnl = safe_float(row.get("raw_pnl_usdc"))
    skill_usdc = safe_float(row.get("skill_usdc"))

    if cand_score is not None and metric_score is not None:
        score = 0.45 * min(cand_score, metric_score) + 0.35 * cand_score + 0.20 * metric_score
    elif cand_score is not None:
        score = min(cand_score, 62.0)
    elif metric_score is not None:
        score = min(metric_score, 74.0)
    else:
        score = None

    cap_reason = ""
    if score is not None and cand_score is not None and not has_metrics:
        score = min(score, 62.0)
        cap_reason = "needs user update for realized profit"
    if score is not None and raw_pnl is not None and raw_pnl <= 0:
        if skill_usdc is not None and skill_usdc > 0:
            score = min(score, 70.0)
            cap_reason = "negative realized PnL caps insider score"
        else:
            score = min(score, 54.0)
            cap_reason = "loss-making profile caps insider score"
    if score is not None and raw_pnl is not None and raw_pnl > 0 and skill_usdc is not None and skill_usdc <= 0:
        score = min(score, 78.0)
        cap_reason = "profit lacks positive expected edge"

    row["scanner_signal_score"] = cand_score
    row["metric_signal_score"] = metric_score
    row["profit_signal_score"] = metric_score
    row["score_cap_reason"] = cap_reason
    row["info_edge_score"] = round(score, 1) if score is not None else None
    row["info_edge_tier"] = score_tier(score)
    row["evidence_summary"] = evidence_summary(row)
    return row


def row_matches(row: Dict[str, Any], query: str) -> bool:
    if not query:
        return True
    needle = query.lower()
    return any(needle in str(value).lower() for value in row.values())


def is_wallet(value: Any) -> bool:
    return bool(WALLET_RE.match(str(value or "").strip()))


def apply_alias_to_profile(profile: Dict[str, Any], alias: Dict[str, Any], user_key: str) -> None:
    if not alias:
        return
    alias_value = str(alias.get("alias") or "").strip()
    display_value = str(alias.get("display_name") or alias_value or "").strip()
    fallback_text = str(user_key or "").strip().lower()

    current_input = str(profile.get("input_user") or "").strip()
    if alias_value and (not current_input or current_input.lower() == fallback_text or is_wallet(current_input)):
        profile["input_user"] = alias_value

    if not display_value or is_wallet(display_value):
        return
    for key in ("display_name", "name"):
        current = str(profile.get(key) or "").strip()
        if not current or current.lower() == fallback_text or is_wallet(current):
            profile[key] = display_value


def profile_display_name(profile: Dict[str, Any], fallback: str) -> str:
    fallback_text = str(fallback or "").strip().lower()
    for key in ("display_name", "name", "pseudonym", "profile_name", "profile_pseudonym", "input_user", "referral"):
        value = str(profile.get(key) or "").strip()
        if value and value.lower() != fallback_text and not is_wallet(value):
            return value
    return fallback


def load_user_analysis(
    data_dir: str,
    reports_dir: str,
    query: str = "",
    limit: int = 200,
    candidate_only: bool = False,
) -> Dict[str, Any]:
    db_path = default_db_path(data_dir)
    if not sqlite_database_available(data_dir):
        return {
            "count": 0,
            "total_count": 0,
            "database_user_count": 0,
            "candidate_count": 0,
            "candidate_only_count": 0,
            "shown_candidate_count": 0,
            "candidate_filter_active": bool(candidate_only),
            "high_signal_count": 0,
            "rows": [],
            "top_info_edge": [],
            "top_luck": [],
            "top_skill": [],
            "top_pnl": [],
            "source": "database",
            "error": "database unavailable",
        }

    metric_rows = sqlite_load_user_metric_rows(db_path)
    user_profiles = sqlite_load_user_profile_rows(data_dir, db_path=db_path)
    user_aliases = sqlite_load_user_alias_rows(data_dir, db_path=db_path)
    aliases_by_user: Dict[str, Dict[str, Any]] = {}

    def alias_score(alias_row: Dict[str, Any]) -> Tuple[int, str]:
        alias_value = str(alias_row.get("alias") or "").strip()
        display_value = str(alias_row.get("display_name") or "").strip()
        user_key = str(alias_row.get("user_key") or "").strip()
        score = 0
        if alias_value and not is_wallet(alias_value):
            score += 2
        if display_value and display_value.lower() != user_key.lower() and not is_wallet(display_value):
            score += 3
        if str(alias_row.get("source") or "").lower() in {"manual", "polymarket_profile", "user_update"}:
            score += 1
        return score, str(alias_row.get("updated_at") or "")

    for alias_row in user_aliases.values():
        key = str(alias_row.get("user_key") or "")
        if not key:
            continue
        existing_alias = aliases_by_user.get(key)
        if not existing_alias or alias_score(alias_row) > alias_score(existing_alias):
            aliases_by_user[key] = alias_row

    total_count = len(metric_rows)
    try:
        database_user_count = len(sqlite_list_user_keys(data_dir, include_candidates=False, db_path=db_path))
    except Exception:
        database_user_count = total_count
    scanner_candidates = sqlite_load_scanner_candidate_rows(db_path)
    watermarks = load_watermarks(data_dir)
    user_watermarks = watermarks.get("users", {}) if isinstance(watermarks.get("users"), dict) else {}
    analysis_user_watermarks = (
        watermarks.get("analysis_users", {}) if isinstance(watermarks.get("analysis_users"), dict) else {}
    )

    def compact(row: Dict[str, Any]) -> Dict[str, Any]:
        user_key = str(row.get("user_key", "") or "")
        freshness = user_watermarks.get(user_key, {}) if isinstance(user_watermarks.get(user_key), dict) else {}
        analysis_freshness = (
            analysis_user_watermarks.get(user_key, {})
            if isinstance(analysis_user_watermarks.get(user_key), dict)
            else {}
        )
        metrics_mtime = str(row.get("_db_updated_at") or "")
        profile = dict(user_profiles.get(user_key, {}))
        alias = aliases_by_user.get(user_key, {})
        apply_alias_to_profile(profile, alias, user_key)
        display_name = profile_display_name(profile, user_key)
        return {
            "user_key": user_key,
            "display_name": display_name,
            "profile_name": str(profile.get("name") or profile.get("profile_name") or ""),
            "profile_pseudonym": str(profile.get("pseudonym") or profile.get("profile_pseudonym") or ""),
            "input_user": str(profile.get("input_user") or ""),
            "total_trades": safe_int(row.get("total_trades")),
            "resolved_trade_count": safe_int(row.get("resolved_trade_count")),
            "luck_index": safe_float(row.get("luck_index")),
            "luck_z": safe_float(row.get("luck_z")),
            "skill_index": safe_float(row.get("skill_index")),
            "skill_z": safe_float(row.get("skill_z")),
            "raw_pnl_usdc": safe_float(row.get("raw_pnl_usdc")),
            "skill_usdc": safe_float(row.get("skill_usdc")),
            "skill_roi": safe_float(row.get("skill_roi")),
            "pnl_per_trade_usdc": safe_float(row.get("pnl_per_trade_usdc")),
            "pnl_per_resolved_trade_usdc": safe_float(row.get("pnl_per_resolved_trade_usdc")),
            "skill_per_trade_usdc": safe_float(row.get("skill_per_trade_usdc")),
            "skill_per_resolved_trade_usdc": safe_float(row.get("skill_per_resolved_trade_usdc")),
            "monetized_edge_per_trade_usdc": safe_float(row.get("monetized_edge_per_trade_usdc")),
            "gross_trade_notional_usdc": safe_float(row.get("gross_trade_notional_usdc")),
            "total_luck_usdc": safe_float(row.get("total_luck_usdc")),
            "resolved_markets": safe_int(row.get("number_of_resolved_markets")),
            "profitable_resolved_markets": safe_int(row.get("profitable_resolved_markets")),
            "losing_resolved_markets": safe_int(row.get("losing_resolved_markets")),
            "profitable_market_ratio": safe_float(row.get("profitable_market_ratio")),
            "largest_market_profit_usdc": safe_float(row.get("largest_market_profit_usdc")),
            "largest_market_loss_usdc": safe_float(row.get("largest_market_loss_usdc")),
            "warning_count": safe_int(row.get("warning_count")),
            "last_updated_at": str(analysis_freshness.get("updated_at") or metrics_mtime or freshness.get("updated_at") or ""),
            "data_updated_at": str(freshness.get("updated_at") or ""),
            "analysis_updated_at": str(analysis_freshness.get("updated_at") or metrics_mtime or ""),
            "last_seen_trade_utc": str(freshness.get("last_seen_trade_utc") or ""),
            "new_trades_written": safe_int(freshness.get("new_trades_written")),
            "source": "analyzed",
            "source_label": "Analyzed",
            "has_metrics": True,
            "candidate_confidence": "",
            "candidate_timing_z": None,
            "candidate_events": None,
            "candidate_jumps_captured": None,
            "candidate_jump_capture": None,
            "candidate_gross_notional": None,
            "candidate_directional_ratio": None,
        }

    def apply_candidate_fields(row: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        confidence = str(candidate.get("confidence_level") or "").strip().lower()
        row["source_label"] = f"Analyzed + {confidence}" if row.get("has_metrics") and confidence else row.get("source_label")
        row["candidate_confidence"] = confidence
        row["candidate_timing_z"] = safe_float(candidate.get("timing_z"))
        row["candidate_events"] = safe_int(candidate.get("number_of_independent_events"))
        row["candidate_jumps_captured"] = safe_int(candidate.get("number_of_jumps_captured"))
        row["candidate_jump_capture"] = safe_float(
            candidate.get("cluster_capped_total_jump_capture") or candidate.get("total_jump_capture")
        )
        row["candidate_gross_notional"] = safe_float(candidate.get("gross_notional"))
        row["candidate_directional_ratio"] = safe_float(candidate.get("avg_directional_ratio"))
        return row

    def confidence_rank(row: Dict[str, Any]) -> Tuple[int, float]:
        ranks = {"very_high": 4, "high": 3, "candidate": 2, "monitor": 1}
        confidence = str(row.get("confidence_level") or "").strip().lower()
        return (ranks.get(confidence, 0), safe_float(row.get("timing_z")) or float("-inf"))

    candidate_by_user: Dict[str, Dict[str, Any]] = {}
    for row in scanner_candidates:
        confidence = str(row.get("confidence_level") or "").strip().lower()
        if confidence not in USER_LIST_CANDIDATE_CONFIDENCES:
            continue
        user_key = safe_user_key(str(row.get("wallet") or "").lower())
        if not user_key:
            continue
        existing = candidate_by_user.get(user_key)
        if existing is None or confidence_rank(row) > confidence_rank(existing):
            candidate_by_user[user_key] = row

    metric_user_keys = {str(row.get("user_key") or "") for row in metric_rows if row.get("user_key")}
    compacted_metric_rows = [compact(row) for row in metric_rows]

    for row in compacted_metric_rows:
        candidate = candidate_by_user.get(str(row.get("user_key") or ""))
        if not candidate:
            continue
        apply_candidate_fields(row, candidate)

    def compact_candidate(user_key: str, row: Dict[str, Any]) -> Dict[str, Any]:
        freshness = user_watermarks.get(user_key, {}) if isinstance(user_watermarks.get(user_key), dict) else {}
        confidence = str(row.get("confidence_level") or "").strip().lower()
        profile = dict(user_profiles.get(user_key, {}))
        alias = aliases_by_user.get(user_key, {})
        apply_alias_to_profile(profile, alias, user_key)
        display_name = profile_display_name(profile, user_key)
        return {
            "user_key": user_key,
            "display_name": display_name,
            "profile_name": str(profile.get("name") or profile.get("profile_name") or ""),
            "profile_pseudonym": str(profile.get("pseudonym") or profile.get("profile_pseudonym") or ""),
            "input_user": str(profile.get("input_user") or ""),
            "total_trades": None,
            "resolved_trade_count": None,
            "luck_index": None,
            "luck_z": None,
            "skill_index": None,
            "skill_z": None,
            "raw_pnl_usdc": None,
            "skill_usdc": None,
            "skill_roi": None,
            "pnl_per_trade_usdc": None,
            "pnl_per_resolved_trade_usdc": None,
            "skill_per_trade_usdc": None,
            "skill_per_resolved_trade_usdc": None,
            "monetized_edge_per_trade_usdc": None,
            "gross_trade_notional_usdc": None,
            "total_luck_usdc": None,
            "resolved_markets": None,
            "profitable_resolved_markets": None,
            "losing_resolved_markets": None,
            "profitable_market_ratio": None,
            "largest_market_profit_usdc": None,
            "largest_market_loss_usdc": None,
            "warning_count": None,
            "last_updated_at": "",
            "data_updated_at": str(freshness.get("updated_at") or ""),
            "analysis_updated_at": "",
            "last_seen_trade_utc": str(freshness.get("last_seen_trade_utc") or ""),
            "new_trades_written": safe_int(freshness.get("new_trades_written")),
            "source": "candidate",
            "source_label": f"Candidate {confidence}" if confidence else "Candidate",
            "has_metrics": False,
            "candidate_confidence": confidence,
            "candidate_timing_z": safe_float(row.get("timing_z")),
            "candidate_events": safe_int(row.get("number_of_independent_events")),
            "candidate_jumps_captured": safe_int(row.get("number_of_jumps_captured")),
            "candidate_jump_capture": safe_float(row.get("cluster_capped_total_jump_capture") or row.get("total_jump_capture")),
            "candidate_gross_notional": safe_float(row.get("gross_notional")),
            "candidate_directional_ratio": safe_float(row.get("avg_directional_ratio")),
        }

    candidate_only_rows = [
        compact_candidate(user_key, row)
        for user_key, row in candidate_by_user.items()
        if user_key not in metric_user_keys
    ]

    profile_only_rows = []
    profile_user_keys = set(user_profiles.keys()) | set(aliases_by_user.keys())
    for user_key in profile_user_keys:
        profile = dict(user_profiles.get(user_key, {}))
        alias = aliases_by_user.get(user_key, {})
        apply_alias_to_profile(profile, alias, user_key)
        if user_key in metric_user_keys or user_key in candidate_by_user:
            continue
        freshness = user_watermarks.get(user_key, {}) if isinstance(user_watermarks.get(user_key), dict) else {}
        display_name = profile_display_name(profile, user_key)
        profile_only_rows.append(
            {
                "user_key": user_key,
                "display_name": display_name,
                "profile_name": str(profile.get("name") or profile.get("profile_name") or ""),
                "profile_pseudonym": str(profile.get("pseudonym") or profile.get("profile_pseudonym") or ""),
                "input_user": str(profile.get("input_user") or ""),
                "total_trades": None,
                "resolved_trade_count": None,
                "luck_index": None,
                "luck_z": None,
                "skill_index": None,
                "skill_z": None,
                "raw_pnl_usdc": None,
                "skill_usdc": None,
                "skill_roi": None,
                "pnl_per_trade_usdc": None,
                "pnl_per_resolved_trade_usdc": None,
                "skill_per_trade_usdc": None,
                "skill_per_resolved_trade_usdc": None,
                "monetized_edge_per_trade_usdc": None,
                "gross_trade_notional_usdc": None,
                "total_luck_usdc": None,
                "resolved_markets": None,
                "profitable_resolved_markets": None,
                "losing_resolved_markets": None,
                "profitable_market_ratio": None,
                "largest_market_profit_usdc": None,
                "largest_market_loss_usdc": None,
                "warning_count": None,
                "last_updated_at": str(freshness.get("updated_at") or profile.get("_db_updated_at") or ""),
                "data_updated_at": str(freshness.get("updated_at") or ""),
                "analysis_updated_at": "",
                "last_seen_trade_utc": str(freshness.get("last_seen_trade_utc") or ""),
                "new_trades_written": safe_int(freshness.get("new_trades_written")),
                "source": "profile",
                "source_label": "Needs update",
                "has_metrics": False,
                "candidate_confidence": "",
                "candidate_timing_z": None,
                "candidate_events": None,
                "candidate_jumps_captured": None,
                "candidate_jump_capture": None,
                "candidate_gross_notional": None,
                "candidate_directional_ratio": None,
            }
        )

    def compacted_candidate_rank(row: Dict[str, Any]) -> Tuple[int, float]:
        ranks = {"very_high": 4, "high": 3, "candidate": 2, "monitor": 1}
        confidence = str(row.get("candidate_confidence") or "").strip().lower()
        return (ranks.get(confidence, 0), safe_float(row.get("candidate_timing_z")) or float("-inf"))

    candidate_only_rows.sort(key=compacted_candidate_rank, reverse=True)

    for row in compacted_metric_rows:
        apply_information_edge_score(row, candidate_by_user.get(str(row.get("user_key") or "")))
        row["monetized_edge_per_trade_usdc"] = monetized_edge_per_trade(row)
        row["evidence_summary"] = evidence_summary(row)
    for row in candidate_only_rows:
        apply_information_edge_score(row)
        row["monetized_edge_per_trade_usdc"] = monetized_edge_per_trade(row)
        row["evidence_summary"] = evidence_summary(row)
    for row in profile_only_rows:
        apply_information_edge_score(row)
        row["info_edge_tier"] = "Needs update"
        row["monetized_edge_per_trade_usdc"] = monetized_edge_per_trade(row)
        row["evidence_summary"] = "Alias/profile exists in SQLite, but trade and analysis rows are missing. Update this user to fetch trades and rerun analysis."

    combined_rows = compacted_metric_rows + candidate_only_rows + profile_only_rows
    candidate_rows_all = [row for row in combined_rows if row.get("candidate_confidence")]
    rows_for_filter = candidate_rows_all if candidate_only else combined_rows
    visible_rows = [row for row in rows_for_filter if row_matches(row, query)]
    visible_metric_rows = [
        row
        for row in compacted_metric_rows
        if row_matches(row, query) and (not candidate_only or row.get("candidate_confidence"))
    ]
    visible_candidate_rows = [row for row in candidate_rows_all if row_matches(row, query)]
    top_luck = sort_by_float(visible_metric_rows, "luck_index")[:limit]
    top_skill = sort_by_float(visible_metric_rows, "skill_index")[:limit]
    top_pnl = sort_by_float(visible_metric_rows, "raw_pnl_usdc")[:limit]
    top_info_edge = sort_by_float(visible_rows, "info_edge_score")[:limit]
    rows_for_table = list(top_info_edge)
    if not candidate_only:
        shown_keys = {str(row.get("user_key") or "") for row in rows_for_table}
        for row in profile_only_rows:
            user_key = str(row.get("user_key") or "")
            if user_key and user_key not in shown_keys and row_matches(row, query):
                rows_for_table.append(row)
                shown_keys.add(user_key)

    return {
        "count": len(visible_rows),
        "total_count": total_count,
        "database_user_count": database_user_count,
        "candidate_count": len(candidate_by_user),
        "candidate_only_count": len(candidate_only_rows),
        "shown_candidate_count": len(visible_candidate_rows),
        "candidate_filter_active": bool(candidate_only),
        "high_signal_count": sum(1 for row in combined_rows if (safe_float(row.get("info_edge_score")) or 0.0) >= 70.0),
        "rows": rows_for_table,
        "top_info_edge": top_info_edge,
        "top_luck": top_luck,
        "top_skill": top_skill,
        "top_pnl": top_pnl,
        "source": "database",
    }


def load_scanner(reports_dir: str, query: str = "", limit: int = 200, data_dir: str = "data") -> Dict[str, Any]:
    db_path = default_db_path(data_dir)
    if not sqlite_database_available(data_dir):
        return {
            "candidate_count": 0,
            "event_count": 0,
            "confidence_counts": {},
            "candidates": [],
            "events": [],
            "source": "database",
            "error": "database unavailable",
        }
    candidate_rows = sqlite_load_scanner_candidate_rows(db_path)
    event_rows = sqlite_load_scanner_jump_event_rows(db_path)

    candidate_rows = [row for row in candidate_rows if row_matches(row, query)]
    event_rows = [row for row in event_rows if row_matches(row, query)]

    confidence_counts: Dict[str, int] = {}
    for row in candidate_rows:
        confidence = str(row.get("confidence_level") or "unknown")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

    candidates = []
    for row in sort_by_float(candidate_rows, "timing_z")[:limit]:
        signal_score = candidate_signal_score(row)
        candidates.append(
            {
                "wallet": row.get("wallet", ""),
                "timing_z": safe_float(row.get("timing_z")),
                "timing_index": safe_float(row.get("timing_index")),
                "confidence_level": row.get("confidence_level", ""),
                "signal_score": signal_score,
                "signal_tier": score_tier(signal_score),
                "independent_events": safe_int(row.get("number_of_independent_events")),
                "jumps_captured": safe_int(row.get("number_of_jumps_captured")),
                "total_jump_capture": safe_float(row.get("total_jump_capture")),
                "gross_notional": safe_float(row.get("gross_notional")),
                "avg_directional_ratio": safe_float(row.get("avg_directional_ratio")),
            }
        )

    def event_score(row: Dict[str, str]) -> float:
        return market_jump_signal_score(row.get("delta_p"), row.get("participants"))

    events = []
    for row in sorted(event_rows, key=event_score, reverse=True)[:limit]:
        signal_score = event_score(row)
        events.append(
            {
                "question": row.get("question", ""),
                "slug": row.get("slug", ""),
                "event_slug": row.get("event_slug", "") or row.get("eventSlug", ""),
                "jump_time_utc": row.get("jump_time_utc", ""),
                "delta_p": safe_float(row.get("delta_p")),
                "p_before": safe_float(row.get("p_before")),
                "p_after": safe_float(row.get("p_after")),
                "participants": safe_int(row.get("participants")),
                "volume_during_jump": safe_float(row.get("volume_during_jump")),
                "signal_score": signal_score,
                "signal_tier": score_tier(signal_score),
                "condition_id": row.get("condition_id", ""),
            }
        )

    return {
        "candidate_count": len(candidate_rows),
        "event_count": len(event_rows),
        "confidence_counts": confidence_counts,
        "candidates": candidates,
        "events": events,
        "source": "database",
    }


def safe_user_key(value: str) -> str:
    text = str(value or "").strip()
    return "".join(ch for ch in text if ch.isalnum() or ch in "._-")


def clean_user_identifier(value: Any) -> str:
    text = str(value or "").strip().lstrip("@").lower()
    return safe_user_key(text)


def scanner_candidate_user_keys(
    reports_dir: str,
    *,
    confidences: Iterable[str] = USER_LIST_CANDIDATE_CONFIDENCES,
    data_dir: str = "data",
) -> List[str]:
    accepted = {str(confidence).strip().lower() for confidence in confidences if str(confidence).strip()}
    ranks = {"very_high": 4, "high": 3, "candidate": 2, "monitor": 1}
    best_by_user: Dict[str, Dict[str, Any]] = {}

    if not sqlite_database_available(data_dir):
        return []
    rows = sqlite_load_scanner_candidate_rows(default_db_path(data_dir))
    for row in rows:
        confidence = str(row.get("confidence_level") or "").strip().lower()
        if confidence not in accepted:
            continue
        user_key = safe_user_key(str(row.get("wallet") or "").lower())
        if not user_key:
            continue
        current = best_by_user.get(user_key)
        row_rank = (ranks.get(confidence, 0), safe_float(row.get("timing_z")) or float("-inf"))
        current_rank = (
            ranks.get(str((current or {}).get("confidence_level") or "").strip().lower(), 0),
            safe_float((current or {}).get("timing_z")) or float("-inf"),
        )
        if current is None or row_rank > current_rank:
            best_by_user[user_key] = row

    return [
        user_key
        for user_key, row in sorted(
            best_by_user.items(),
            key=lambda item: (
                ranks.get(str(item[1].get("confidence_level") or "").strip().lower(), 0),
                safe_float(item[1].get("timing_z")) or float("-inf"),
                item[0],
            ),
            reverse=True,
        )
    ]


def load_user_profile(data_dir: str, user_key: str) -> Dict[str, Any]:
    return sqlite_load_user_profile_row(data_dir, user_key) if sqlite_database_available(data_dir) else {}


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_market_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if text.endswith("+00"):
        text = text[:-3] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def market_status_from_row(row: Dict[str, Any]) -> Dict[str, str]:
    if truthy(row.get("closed")) or str(row.get("umaResolutionStatus") or "").strip().lower() == "resolved":
        return {"market_status": "closed", "market_status_label": "Closed"}

    end_dt = parse_market_datetime(row.get("endDate"))
    if truthy(row.get("active")):
        return {"market_status": "open", "market_status_label": "Open"}
    if end_dt is not None:
        if end_dt > datetime.now(timezone.utc):
            return {"market_status": "open", "market_status_label": "Open"}
        return {"market_status": "ended", "market_status_label": "Ended"}

    return {"market_status": "unknown", "market_status_label": "Unknown"}


def market_summary_from_database(
    data_dir: str,
    condition_id: str,
    *,
    listed: bool = False,
    added_at: str = "",
    source: str = "database",
    db_row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized = normalize_condition_id(condition_id)
    if db_row is not None:
        db_row = dict(db_row)
    elif sqlite_database_available(data_dir):
        db_row = sqlite_load_market_raw_row(data_dir, normalized)
    else:
        db_row = {}
    row = db_row or {}
    status = market_status_from_row(row) if row else {"market_status": "unknown", "market_status_label": "Unknown"}
    title = (
        row.get("question")
        or row.get("title")
        or row.get("slug")
        or normalized
        or str(condition_id or "")
    )
    slug = row.get("slug") or ""
    event_slug = row.get("eventSlug") or row.get("event_slug") or ""
    return {
        "condition_id": normalized or str(condition_id or ""),
        "title": title,
        "slug": slug,
        "event_slug": event_slug,
        "category": row.get("category") or "",
        "volume": safe_float(row.get("volumeNum")) or safe_float(row.get("volume")),
        "liquidity": safe_float(row.get("liquidityNum")) or safe_float(row.get("liquidity")),
        "end_date": row.get("endDate") or row.get("endDateIso") or "",
        "last_updated_at": str(row.get("_db_updated_at") or ""),
        "in_database": bool(row),
        "listed": bool(listed),
        "scanner_signal": False,
        "added_at": added_at,
        "source": source,
        **status,
    }


def load_markets(data_dir: str, reports_dir: str, query: str = "", limit: int = 1000) -> Dict[str, Any]:
    if not sqlite_database_available(data_dir):
        return {
            "count": 0,
            "shown_count": 0,
            "loaded_count": 0,
            "total_market_count": 0,
            "listed_count": 0,
            "added_count": 0,
            "database_count": 0,
            "rows": [],
            "scanner_events": [],
            "scanner_event_count": 0,
            "source": "database",
            "error": "database unavailable",
        }

    market_list = load_market_list(ROOT)
    entries = market_list.get("markets", [])
    listed_ids = {entry.get("condition_id") for entry in entries if entry.get("condition_id")}
    rows_by_id: Dict[str, Dict[str, Any]] = {}

    for row in sqlite_load_market_summary_rows(data_dir, query=query, limit=limit):
        condition_id = normalize_condition_id(row.get("conditionId") or row.get("condition_id"))
        if condition_id and condition_id not in rows_by_id:
            rows_by_id[condition_id] = market_summary_from_database(data_dir, condition_id, source="database", db_row=row)

    for entry in entries:
        condition_id = str(entry.get("condition_id") or "")
        if not condition_id:
            continue
        summary = rows_by_id.get(condition_id)
        if summary is None:
            summary = market_summary_from_database(
                data_dir,
                condition_id,
                listed=True,
                added_at=str(entry.get("added_at") or ""),
                source=str(entry.get("source") or "manual"),
            )
            rows_by_id[condition_id] = summary
        summary["listed"] = True
        summary["added_at"] = summary.get("added_at") or str(entry.get("added_at") or "")
        summary["source"] = str(entry.get("source") or summary.get("source") or "manual")

    scanner = load_scanner(reports_dir, query="", limit=limit, data_dir=data_dir)
    scanner_events: List[Dict[str, Any]] = []
    seen_scanner = set()
    for event in scanner.get("events", []):
        condition_id = normalize_condition_id(event.get("condition_id"))
        if not condition_id or condition_id in seen_scanner:
            continue
        seen_scanner.add(condition_id)
        summary = rows_by_id.get(condition_id)
        if summary is None:
            summary = market_summary_from_database(
                data_dir,
                condition_id,
                listed=condition_id in listed_ids,
                source="scanner",
            )
            rows_by_id[condition_id] = summary
        summary.update(
            {
                "scanner_signal": True,
                "scanner_jump_time_utc": event.get("jump_time_utc") or "",
                "scanner_move": safe_float(event.get("delta_p")),
                "scanner_participants": safe_int(event.get("participants")),
                "signal_score": event.get("signal_score")
                or market_jump_signal_score(event.get("delta_p"), event.get("participants")),
                "signal_tier": event.get("signal_tier")
                or score_tier(market_jump_signal_score(event.get("delta_p"), event.get("participants"))),
                "slug": summary.get("slug") or event.get("slug") or "",
                "event_slug": summary.get("event_slug") or event.get("event_slug") or "",
                "title": summary.get("title") or event.get("question") or event.get("slug") or condition_id,
            }
        )
        scanner_events.append(summary)

    for row in rows_by_id.values():
        sources: List[str] = []
        if row.get("listed"):
            sources.append("Added")
        if row.get("scanner_signal"):
            sources.append("Signal")
        if not sources and row.get("in_database"):
            sources.append("Database")
        row["source_summary"] = ", ".join(sources) or "Unknown"

    rows = [row for row in rows_by_id.values() if row_matches(row, query)]
    rows.sort(
        key=lambda row: (
            1 if row.get("scanner_signal") else 0,
            safe_float(row.get("signal_score")) or float("-inf"),
            str(row.get("last_updated_at") or row.get("scanner_jump_time_utc") or ""),
        ),
        reverse=True,
    )
    row_limit = max(1, min(limit, 1000))
    shown_rows = rows[:row_limit]
    total_market_count = sqlite_count_market_rows(data_dir)

    watermarks = load_watermarks(data_dir)
    updates = watermarks.get("updates", {}) if isinstance(watermarks.get("updates"), dict) else {}
    analysis = watermarks.get("analysis", {}) if isinstance(watermarks.get("analysis"), dict) else {}
    return {
        "count": len(rows),
        "shown_count": len(shown_rows),
        "loaded_count": len(rows_by_id),
        "total_market_count": total_market_count,
        "listed_count": total_market_count,
        "added_count": len(listed_ids),
        "database_count": total_market_count,
        "rows": shown_rows,
        "scanner_events": scanner_events[:row_limit],
        "scanner_event_count": scanner.get("event_count", 0),
        "watermark": updates.get("markets", {}) if isinstance(updates.get("markets"), dict) else {},
        "trade_watermark": (
            updates.get("market_trades", {}) if isinstance(updates.get("market_trades"), dict) else {}
        ),
        "analysis_watermark": (
            analysis.get("market_scanner", {}) if isinstance(analysis.get("market_scanner"), dict) else {}
        ),
        "source": "database",
    }


def listed_market_ids(data_dir: str, reports_dir: str, limit: int | None = None) -> List[str]:
    ids: List[str] = []
    seen = set()

    def add(value: Any) -> None:
        condition_id = normalize_condition_id(value)
        if not condition_id or condition_id in seen:
            return
        seen.add(condition_id)
        ids.append(condition_id)

    market_list = load_market_list(ROOT)
    for entry in market_list.get("markets", []):
        add(entry.get("condition_id") if isinstance(entry, dict) else entry)

    if sqlite_database_available(data_dir):
        db_path = default_db_path(data_dir)
        try:
            for event in sqlite_load_scanner_jump_event_rows(db_path):
                add(event.get("condition_id"))
        except Exception:
            pass
        for condition_id in sorted(sqlite_market_condition_ids(data_dir, db_path=db_path)):
            add(condition_id)
    else:
        payload = load_markets(data_dir, reports_dir, limit=limit or 1000)
        for row in payload.get("rows", []):
            add(row.get("condition_id"))

    if limit is not None and limit > 0:
        return ids[:limit]
    return ids


def write_market_ids_file(market_ids: Iterable[str], root: Path | str = ROOT) -> Path:
    ids = []
    seen = set()
    for market_id in market_ids:
        condition_id = normalize_condition_id(market_id)
        if not condition_id or condition_id in seen:
            continue
        seen.add(condition_id)
        ids.append(condition_id)

    path = run_state_dir(root) / "listed_markets_update.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return path


def write_user_keys_file(user_keys: Iterable[str], root: Path | str = ROOT, prefix: str = "users") -> Path:
    keys: List[str] = []
    seen = set()
    for user_key in user_keys:
        key = safe_user_key(str(user_key or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)

    path = run_state_dir(root) / f"{prefix}_{utc_now_compact()}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    return path


def normalize_outcome_label(value: Any) -> str:
    return str(value or "").strip().lower()


def infer_winning_outcome(status: Dict[str, Any]) -> Tuple[int | None, str]:
    outcomes = [str(value) for value in status.get("market_outcomes") or []]
    raw_index = safe_int(status.get("market_winning_outcome_index"))
    if raw_index is not None and raw_index >= 0:
        label = outcomes[raw_index] if raw_index < len(outcomes) else str(status.get("market_winner") or "")
        return raw_index, label

    winner = str(status.get("market_winner") or "").strip()
    if winner:
        winner_key = normalize_outcome_label(winner)
        for idx, label in enumerate(outcomes):
            if normalize_outcome_label(label) == winner_key:
                return idx, label
        return None, winner

    prices = [safe_float(value) for value in status.get("market_outcome_prices") or []]
    finite_prices = [(idx, price) for idx, price in enumerate(prices) if price is not None]
    if finite_prices:
        idx, price = max(finite_prices, key=lambda item: item[1])
        if price is not None and price >= 0.99:
            label = outcomes[idx] if idx < len(outcomes) else ""
            return idx, label

    return None, ""


def closed_market_pnl(group: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
    if status.get("market_status") != "closed":
        return {}

    winner_index, winner_label = infer_winning_outcome(status)
    if winner_index is None and not winner_label:
        return {
            "closed_pnl": None,
            "closed_pnl_note": "Closed market; winning outcome not available in cache",
        }

    positions_by_index = group.get("positions_by_index") if isinstance(group.get("positions_by_index"), dict) else {}
    positions_by_label = group.get("positions_by_label") if isinstance(group.get("positions_by_label"), dict) else {}

    winning_position: float | None = None
    if winner_index is not None:
        index_key = str(winner_index)
        if index_key in positions_by_index:
            winning_position = safe_float(positions_by_index.get(index_key)) or 0.0

    if winning_position is None and winner_label:
        winning_position = safe_float(positions_by_label.get(normalize_outcome_label(winner_label))) or 0.0

    if winning_position is None:
        winning_position = 0.0

    payout = max(0.0, winning_position)
    cash_flow = safe_float(group.get("cash_flow")) or 0.0
    pnl = cash_flow + payout
    note_parts = []
    if winner_label:
        note_parts.append(f"Winner: {winner_label}")
    note_parts.append(f"Cash flow: {cash_flow:.2f}")
    note_parts.append(f"Final payout: {payout:.2f}")
    return {
        "closed_pnl": pnl,
        "closed_pnl_note": "; ".join(note_parts),
        "winning_outcome": winner_label,
        "winning_position": winning_position,
        "resolution_payout": payout,
    }


def user_market_scanner_signal_score(
    *,
    capture: float | None,
    gross_notional: float | None,
    events: int | None,
    positive_events: int | None,
    directional_ratio: float | None,
) -> float | None:
    if capture is None and not events:
        return None
    capture_roi = (
        capture / gross_notional
        if capture is not None and gross_notional is not None and gross_notional > 0
        else 0.0
    )
    capture_score = clamp_score(45.0 + capture_roi * 900.0)
    event_score = clamp_score(35.0 + (events or 0) * 12.0)
    hit_score = clamp_score(35.0 + (positive_events or 0) * 10.0)
    direction_score = clamp_score(45.0 + (directional_ratio or 0.0) * 50.0)
    score = 0.42 * capture_score + 0.24 * event_score + 0.18 * hit_score + 0.16 * direction_score
    if capture is not None and capture <= 0:
        score = min(score, 49.0)
    return round(clamp_score(score), 1)


def load_user_scanner_market_signals(data_dir: str, reports_dir: str, user_key: str) -> Dict[str, Dict[str, Any]]:
    db_path = default_db_path(data_dir)
    if not sqlite_database_available(data_dir):
        return {}
    signals = sqlite_load_user_scanner_market_signals(db_path, user_key)
    for rec in signals.values():
        rec["scanner_signal_score"] = user_market_scanner_signal_score(
            capture=safe_float(rec.get("scanner_user_jump_capture")),
            gross_notional=safe_float(rec.get("scanner_user_gross_notional")),
            events=safe_int(rec.get("scanner_user_event_count")),
            positive_events=safe_int(rec.get("scanner_user_positive_events")),
            directional_ratio=safe_float(rec.get("scanner_user_avg_directional_ratio")),
        )
    return signals


def load_user_trade_history(
    data_dir: str,
    user_key: str,
    *,
    reports_dir: str = "reports",
    query: str = "",
    limit: int = 1000,
) -> Dict[str, Any]:
    key = safe_user_key(user_key)
    if not key or key != user_key:
        return {"error": "invalid user key", "user_key": user_key, "total": 0, "rows": []}

    db_path = default_db_path(data_dir)
    if not sqlite_database_available(data_dir):
        return {"error": "database unavailable", "user_key": key, "total": 0, "rows": []}
    rows, total, _matched = sqlite_load_user_trade_rows(db_path, key, query=query)

    rows.sort(key=lambda row: safe_float(row.get("timestamp")) or -1.0, reverse=True)
    limited_rows = rows[: max(1, min(limit, 5000))]
    condition_ids = {str(row.get("condition_id") or "").strip().lower() for row in rows if row.get("condition_id")}
    market_statuses = sqlite_load_market_statuses(db_path, condition_ids)
    scanner_signals = load_user_scanner_market_signals(data_dir, reports_dir, key)
    groups_by_market: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        market_key = str(row.get("condition_id") or row.get("slug") or row.get("title") or "unknown")
        group = groups_by_market.get(market_key)
        if group is None:
            group = {
                "market_key": market_key,
                "title": row.get("title") or "",
                "slug": row.get("slug") or "",
                "event_slug": row.get("event_slug") or "",
                "condition_id": row.get("condition_id") or "",
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "total_size": 0.0,
                "buy_size": 0.0,
                "sell_size": 0.0,
                "net_size": 0.0,
                "total_notional": 0.0,
                "cash_flow": 0.0,
                "first_timestamp": row.get("timestamp"),
                "last_timestamp": row.get("timestamp"),
                "outcomes": {},
                "positions_by_index": {},
                "positions_by_label": {},
                "trades": [],
            }
            groups_by_market[market_key] = group

        side = str(row.get("side") or "").upper()
        size = safe_float(row.get("size")) or 0.0
        notional = safe_float(row.get("notional")) or 0.0
        ts = safe_float(row.get("timestamp"))
        group["trade_count"] += 1
        group["total_size"] += size
        group["total_notional"] += notional
        group["trades"].append(row)
        if not group.get("event_slug") and row.get("event_slug"):
            group["event_slug"] = row.get("event_slug")
        if side == "BUY":
            group["buy_count"] += 1
            group["buy_size"] += size
            group["net_size"] += size
            group["cash_flow"] -= notional
        elif side == "SELL":
            group["sell_count"] += 1
            group["sell_size"] += size
            group["net_size"] -= size
            group["cash_flow"] += notional
        position_delta = size if side == "BUY" else -size if side == "SELL" else 0.0
        if position_delta:
            outcome_index = row.get("outcome_index")
            if outcome_index is not None:
                positions_by_index = group["positions_by_index"]
                index_key = str(outcome_index)
                positions_by_index[index_key] = positions_by_index.get(index_key, 0.0) + position_delta
            label_key = normalize_outcome_label(row.get("outcome"))
            if label_key:
                positions_by_label = group["positions_by_label"]
                positions_by_label[label_key] = positions_by_label.get(label_key, 0.0) + position_delta
        if ts is not None:
            first_ts = safe_float(group.get("first_timestamp"))
            last_ts = safe_float(group.get("last_timestamp"))
            group["first_timestamp"] = ts if first_ts is None else min(first_ts, ts)
            group["last_timestamp"] = ts if last_ts is None else max(last_ts, ts)
        outcome = str(row.get("outcome") or "").strip() or "Unknown"
        outcomes = group["outcomes"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    groups: List[Dict[str, Any]] = []
    for group in groups_by_market.values():
        total_size = safe_float(group.get("total_size")) or 0.0
        avg_price = (safe_float(group.get("total_notional")) or 0.0) / total_size if total_size > 0 else None
        outcomes = group.get("outcomes") if isinstance(group.get("outcomes"), dict) else {}
        outcome_summary = ", ".join(f"{name} ({count})" for name, count in sorted(outcomes.items()))
        group["avg_price"] = avg_price
        group["first_time_utc"] = to_iso_utc(group.get("first_timestamp"))
        group["last_time_utc"] = to_iso_utc(group.get("last_timestamp"))
        group["outcome_summary"] = outcome_summary
        status = market_statuses.get(str(group.get("condition_id") or "").strip().lower())
        if status:
            group.update(status)
            group.update(closed_market_pnl(group, status))
        else:
            group["market_status"] = "unknown"
            group["market_status_label"] = "Unknown"
            group["market_end_date"] = ""
        scanner_signal = scanner_signals.get(str(group.get("condition_id") or "").strip().lower())
        if scanner_signal:
            group.update(scanner_signal)
        group.pop("positions_by_index", None)
        group.pop("positions_by_label", None)
        group["trades"].sort(key=lambda trade: safe_float(trade.get("timestamp")) or -1.0, reverse=True)
        groups.append(group)

    groups.sort(key=lambda group: safe_float(group.get("last_timestamp")) or -1.0, reverse=True)
    limited_groups = groups[: max(1, min(limit, 1000))]
    profile = load_user_profile(data_dir, key)
    return {
        "user_key": key,
        "profile": profile,
        "total": total,
        "matched": len(rows),
        "market_count": len(groups),
        "limit": len(limited_groups),
        "trade_limit": len(limited_rows),
        "groups": limited_groups,
        "rows": limited_rows,
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Polymarket Information Edge Dashboard</title>
  <script>
    (() => {
      try {
        const theme = localStorage.getItem("polymarket-dashboard-theme");
        if (theme === "dark" || theme === "light") document.documentElement.dataset.theme = theme;
      } catch (_error) {}
    })();
  </script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d6dde8;
      --soft: #eef2f6;
      --accent: #0b6e69;
      --accent-strong: #095c58;
      --blue: #2058a8;
      --warn: #9a5b00;
      --bad: #b42318;
      --good: #087443;
      --accent-ink: #ffffff;
      --control: #ffffff;
      --control-active: #edf7f6;
      --table-head: #fafbfc;
      --row-hover: #fbfcfe;
      --row-selected: #edf7f6;
      --detail-bg: #fbfcfe;
      --toast-bg: #101828;
      --toast-ink: #ffffff;
      --toast-shadow: rgba(16, 24, 40, 0.22);
      --log-bg: #101828;
      --log-ink: #eff4ff;
      --sort-muted: #98a2b3;
      --market-row-open: #f0fdf4;
      --market-row-closed: #f8fafc;
      --market-row-ended: #fffbeb;
      --market-row-unknown: #f9fafb;
      --market-open-bg: #dcfce7;
      --market-open-ink: #166534;
      --market-open-line: #86efac;
      --market-closed-bg: #e5e7eb;
      --market-closed-ink: #374151;
      --market-closed-line: #cbd5e1;
      --market-ended-bg: #fef3c7;
      --market-ended-ink: #92400e;
      --market-ended-line: #fcd34d;
      --market-unknown-bg: #eef2f6;
      --market-unknown-ink: #475467;
      --score-bg: #eef2f6;
      --score-strong-bg: #fee4e2;
      --score-strong-ink: #912018;
      --score-elevated-bg: #fffaeb;
      --score-elevated-ink: #93370d;
      --score-watch-bg: #eff8ff;
      --score-watch-ink: #175cd3;
      --score-baseline-bg: #f2f4f7;
      --score-baseline-ink: #475467;
    }
    html[data-theme="dark"] {
      color-scheme: dark;
      --bg: #0d1116;
      --panel: #151b22;
      --ink: #e7edf3;
      --muted: #9aa8b5;
      --line: #2d3944;
      --soft: #202a33;
      --accent: #39b7aa;
      --accent-strong: #67d7cc;
      --blue: #7db3ff;
      --warn: #f2b458;
      --bad: #ff897f;
      --good: #66d39a;
      --accent-ink: #071312;
      --control: #1a222b;
      --control-active: #123b38;
      --table-head: #1b242d;
      --row-hover: #202a33;
      --row-selected: #123b38;
      --detail-bg: #111820;
      --toast-bg: #e7edf3;
      --toast-ink: #0d1116;
      --toast-shadow: rgba(0, 0, 0, 0.42);
      --log-bg: #090d12;
      --log-ink: #dbe7f3;
      --sort-muted: #6f7e8c;
      --market-row-open: #102419;
      --market-row-closed: #1b2229;
      --market-row-ended: #2a2211;
      --market-row-unknown: #192129;
      --market-open-bg: #123b25;
      --market-open-ink: #96e6b8;
      --market-open-line: #247348;
      --market-closed-bg: #26313a;
      --market-closed-ink: #c5d0db;
      --market-closed-line: #40505f;
      --market-ended-bg: #3b2c0d;
      --market-ended-ink: #ffd98a;
      --market-ended-line: #8f6817;
      --market-unknown-bg: #222c35;
      --market-unknown-ink: #c1ccd7;
      --score-bg: #222c35;
      --score-strong-bg: #451d1a;
      --score-strong-ink: #ffb0a8;
      --score-elevated-bg: #3b2c0d;
      --score-elevated-ink: #ffd98a;
      --score-watch-bg: #142d4a;
      --score-watch-ink: #9ed0ff;
      --score-baseline-bg: #26313a;
      --score-baseline-ink: #c5d0db;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    main {
      max-width: 1600px;
      margin: 0 auto;
      padding: 4px 16px 12px;
      display: grid;
      gap: 8px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 6px 10px;
      border-bottom: 1px solid var(--line);
      min-height: 38px;
    }
    .bar h2 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .muted { color: var(--muted); }
    .tabs {
      display: flex;
      gap: 4px;
      padding: 5px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      overflow-x: auto;
    }
    .tab {
      border: 1px solid var(--line);
      background: var(--control);
      color: var(--ink);
      border-radius: 6px;
      min-height: 28px;
      padding: 4px 9px;
      font-weight: 650;
      cursor: pointer;
    }
    .tab.active {
      border-color: var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
    }
    .theme-toggle {
      margin-left: auto;
      flex: 0 0 auto;
      min-height: 28px;
      min-width: 68px;
    }
    .tab.loading::after {
      content: "";
      width: 10px;
      height: 10px;
      margin-left: 7px;
      border-radius: 50%;
      border: 2px solid currentColor;
      border-top-color: transparent;
      animation: spin 0.75s linear infinite;
    }
    .view { display: none; }
    .view.active { display: grid; gap: 10px; }
    .view.loading::after {
      content: attr(data-loading);
      position: fixed;
      top: 46px;
      right: 18px;
      z-index: 30;
      min-width: 150px;
      max-width: min(360px, calc(100vw - 36px));
      padding: 7px 10px;
      border-radius: 6px;
      background: var(--toast-bg);
      color: var(--toast-ink);
      box-shadow: 0 8px 24px var(--toast-shadow);
      font-size: 12px;
      font-weight: 750;
      line-height: 1.2;
    }
    #view-users.active {
      height: calc(100vh - 72px);
      min-height: 480px;
      overflow: hidden;
    }
    #view-jobs.active {
      height: calc(100vh - 72px);
      min-height: 480px;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 1px;
      background: var(--line);
    }
    .metric {
      background: var(--panel);
      padding: 8px 10px;
      min-height: 56px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(7, minmax(130px, 1fr));
      gap: 6px;
      padding: 8px;
    }
    button, a.button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: var(--control);
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      min-height: 30px;
      padding: 4px 8px;
      text-decoration: none;
      white-space: normal;
      text-align: center;
    }
    button.primary, a.button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: var(--accent-ink);
    }
    button:hover, a.button:hover { border-color: var(--accent); }
    button.primary:hover, a.button.primary:hover { background: var(--accent-strong); }
    button.toggle.active {
      background: var(--control-active);
      border-color: var(--accent);
      color: var(--accent-strong);
    }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    .tools {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    input[type="search"], input[type="text"] {
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 8px;
      font: inherit;
      min-width: 260px;
      background: var(--control);
      color: var(--ink);
    }
    input::placeholder { color: var(--muted); }
    button.mini {
      min-height: 24px;
      padding: 2px 6px;
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      text-align: left;
      padding: 5px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      overflow: hidden;
      text-overflow: ellipsis;
      line-height: 1.25;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: var(--table-head);
    }
    th.sortable {
      cursor: pointer;
      user-select: none;
    }
    th.sortable::after {
      content: " <> ";
      color: var(--sort-muted);
      font-weight: 700;
    }
    th.sortable.sorted.asc::after {
      content: " ^";
      color: var(--accent);
    }
    th.sortable.sorted.desc::after {
      content: " v";
      color: var(--accent);
    }
    tbody tr:hover td { background: var(--row-hover); }
    tr.clickable { cursor: pointer; }
    tr.clickable.selected td {
      background: var(--row-selected);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .status-success, .good { color: var(--good); font-weight: 700; }
    .status-running, .status-queued, .status-cancelled, .warn { color: var(--warn); font-weight: 700; }
    .status-failed, .bad { color: var(--bad); font-weight: 700; }
    .expand {
      width: 26px;
      min-height: 24px;
      padding: 0;
      font-weight: 800;
    }
    .detail-row td {
      background: var(--detail-bg);
      padding: 0;
    }
    .nested {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      border-top: 1px solid var(--line);
    }
    .nested th, .nested td {
      padding: 4px 8px;
      font-size: 12px;
    }
    .market-row-open td { background: var(--market-row-open); }
    .market-row-closed td { background: var(--market-row-closed); }
    .market-row-ended td { background: var(--market-row-ended); }
    .market-row-unknown td { background: var(--market-row-unknown); }
    .market-status-pill {
      display: inline-block;
      border-radius: 999px;
      padding: 1px 6px;
      margin-right: 5px;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid var(--line);
    }
    .market-status-open {
      background: var(--market-open-bg);
      color: var(--market-open-ink);
      border-color: var(--market-open-line);
    }
    .market-status-closed {
      background: var(--market-closed-bg);
      color: var(--market-closed-ink);
      border-color: var(--market-closed-line);
    }
    .market-status-ended {
      background: var(--market-ended-bg);
      color: var(--market-ended-ink);
      border-color: var(--market-ended-line);
    }
    .market-status-unknown {
      background: var(--market-unknown-bg);
      color: var(--market-unknown-ink);
    }
    .score {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 48px;
      border-radius: 6px;
      padding: 2px 6px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      background: var(--score-bg);
      color: var(--ink);
    }
    .score-strong {
      background: var(--score-strong-bg);
      color: var(--score-strong-ink);
    }
    .score-elevated {
      background: var(--score-elevated-bg);
      color: var(--score-elevated-ink);
    }
    .score-watch {
      background: var(--score-watch-bg);
      color: var(--score-watch-ink);
    }
    .score-baseline {
      background: var(--score-baseline-bg);
      color: var(--score-baseline-ink);
    }
    .evidence {
      line-height: 1.25;
      white-space: normal;
    }
    .evidence-compact {
      display: block;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 1px 6px;
      background: var(--soft);
      font-size: 12px;
      font-weight: 700;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    a.external {
      color: var(--blue);
      font-weight: 700;
      text-decoration: none;
    }
    a.external:hover { text-decoration: underline; }
    .grid-2 {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
      gap: 10px;
    }
    .workspace-grid {
      display: grid;
      grid-template-columns: minmax(460px, 0.9fr) minmax(0, 1.1fr);
      gap: 10px;
      align-items: start;
    }
    .user-workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: minmax(0, 7fr) minmax(220px, 3fr);
      gap: 10px;
      align-items: stretch;
      min-height: 0;
      height: 100%;
      overflow: hidden;
    }
    .user-workspace section {
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }
    .table-wrap {
      overflow-x: auto;
    }
    .user-workspace .table-wrap {
      flex: 1;
      min-height: 0;
      overflow: auto;
    }
    .user-workspace table {
      min-width: 1420px;
    }
    .user-workspace section:nth-child(2) table {
      min-width: 1600px;
    }
    .user-workspace th {
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .user-workspace td {
      white-space: nowrap;
    }
    .jobs-table-wrap {
      max-height: 205px;
      overflow: auto;
    }
    .jobs-log-panel {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .jobs-log-panel .log {
      flex: 1;
      min-height: 0;
      max-height: none;
    }
    .log {
      margin: 0;
      padding: 10px;
      background: var(--log-bg);
      color: var(--log-ink);
      min-height: 220px;
      max-height: 520px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.35;
    }
    .empty {
      padding: 14px;
      color: var(--muted);
    }
    .loading-cell {
      color: var(--muted);
      font-weight: 700;
    }
    .loading-spinner {
      display: inline-block;
      width: 12px;
      height: 12px;
      margin-right: 8px;
      border-radius: 50%;
      border: 2px solid var(--sort-muted);
      border-top-color: var(--accent);
      vertical-align: -2px;
      animation: spin 0.75s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @media (max-width: 1100px) {
      main { padding: 8px; }
      .metrics, .actions, .grid-2, .workspace-grid, .user-workspace { grid-template-columns: 1fr; }
      input[type="search"] { min-width: 0; width: 100%; }
      input[type="text"] { min-width: 0; width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <nav class="tabs">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="markets">Markets</button>
      <button class="tab" data-tab="users">Users</button>
      <button class="tab" data-tab="jobs">Jobs</button>
      <button class="theme-toggle" id="theme-toggle" type="button" aria-pressed="false" title="Switch to dark mode">Dark</button>
    </nav>

    <div class="view active" id="view-overview">
      <section>
        <div class="bar">
          <h2>Information Edge Signals</h2>
          <span class="muted">Timing, repeatability, realized edge, and profitability</span>
        </div>
        <div class="metrics">
          <div class="metric"><span>Elevated Users</span><strong id="signal-high-users">0</strong></div>
          <div class="metric"><span>Scanner Candidates</span><strong id="signal-candidates">0</strong></div>
          <div class="metric"><span>Market Jump Events</span><strong id="signal-events">0</strong></div>
          <div class="metric"><span>Analyzed Users</span><strong id="signal-analyzed">0</strong></div>
          <div class="metric"><span>Last Market Analysis</span><strong id="signal-last-analysis">None</strong></div>
        </div>
      </section>
      <section>
        <div class="bar">
          <h2>Data Coverage</h2>
          <span class="muted" id="job-state">Idle</span>
        </div>
        <div class="metrics">
          <div class="metric"><span>Users</span><strong id="m-users">0</strong></div>
          <div class="metric"><span>User Trade Rows</span><strong id="m-user-trades">0</strong></div>
          <div class="metric"><span>Market Rows</span><strong id="m-markets">0</strong></div>
          <div class="metric"><span>Price Assets</span><strong id="m-prices">0</strong></div>
          <div class="metric"><span>Market Trade Sets</span><strong id="m-trades">0</strong></div>
        </div>
      </section>
      <section>
        <div class="bar"><h2>Run</h2><span class="muted" id="run-note"></span></div>
        <div class="actions">
          <button class="primary" data-action="update_markets">Discover Markets</button>
          <button data-action="update_listed_markets">Update Markets</button>
          <button data-action="update_market_trades">Update Trades</button>
          <button data-action="analyze_scanner">Analyze Markets</button>
          <button data-action="update_users_quick">Update Users Fast</button>
          <button data-action="update_users_full">Update Users Full</button>
        </div>
      </section>
      <div class="grid-2">
        <section>
          <div class="bar"><h2>Market Events To Investigate</h2><span class="muted" id="market-count"></span></div>
          <table>
            <thead><tr><th class="sortable" data-sort-table="overviewMarkets" data-sort-key="signal_score" data-sort-type="number">Signal</th><th class="sortable" data-sort-table="overviewMarkets" data-sort-key="title" data-sort-type="text">Market</th><th class="sortable" data-sort-table="overviewMarkets" data-sort-key="scanner_participants" data-sort-type="number">Users</th><th class="sortable" data-sort-table="overviewMarkets" data-sort-key="scanner_jump_time_utc" data-sort-type="text">Jump Time</th></tr></thead>
            <tbody id="overview-markets"></tbody>
          </table>
        </section>
        <section>
          <div class="bar"><h2>Strongest User Signals</h2><span class="muted" id="users-count"></span></div>
          <table>
            <thead><tr><th class="sortable" data-sort-table="overviewUsers" data-sort-key="info_edge_score" data-sort-type="number">Edge</th><th class="sortable" data-sort-table="overviewUsers" data-sort-key="user_key" data-sort-type="text">User</th><th class="sortable" data-sort-table="overviewUsers" data-sort-key="info_edge_tier" data-sort-type="text">Tier</th><th class="sortable" data-sort-table="overviewUsers" data-sort-key="evidence_summary" data-sort-type="text">Evidence</th></tr></thead>
            <tbody id="overview-users"></tbody>
          </table>
        </section>
      </div>
    </div>

    <div class="view" id="view-markets">
      <section>
        <div class="bar">
          <h2>Markets</h2>
          <div class="tools">
            <input type="text" id="market-id-input" placeholder="Paste market conditionId">
            <button class="primary" id="add-market">Add Market</button>
            <button data-action="update_markets">Discover Markets</button>
            <button data-action="update_listed_markets">Update Markets</button>
            <button data-action="update_market_trades">Update Trades</button>
            <button data-action="analyze_scanner">Analyze Markets</button>
          </div>
        </div>
        <div class="metrics">
          <div class="metric"><span>Markets</span><strong id="mk-listed">0</strong></div>
          <div class="metric"><span>Shown</span><strong id="mk-shown">0</strong></div>
          <div class="metric"><span>In Database</span><strong id="mk-database">0</strong></div>
          <div class="metric"><span>Signals</span><strong id="mk-events">0</strong></div>
          <div class="metric"><span>Market Update</span><strong id="mk-watermark">None</strong></div>
          <div class="metric"><span>Trade Update</span><strong id="mk-trade-watermark">None</strong></div>
          <div class="metric"><span>Analyzed</span><strong id="mk-analysis-watermark">None</strong></div>
          <div class="metric"><span>Manual Adds</span><strong id="mk-added">0</strong></div>
        </div>
      </section>
      <div>
        <section>
          <div class="bar">
            <h2>Market List</h2>
            <div class="tools">
              <input type="search" id="market-search" placeholder="Search markets">
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th class="sortable" data-sort-table="markets" data-sort-key="title" data-sort-type="text">Market</th>
                  <th class="sortable" data-sort-table="markets" data-sort-key="source_summary" data-sort-type="text">Source</th>
                  <th class="sortable" data-sort-table="markets" data-sort-key="market_status" data-sort-type="text">Status</th>
                  <th class="sortable" data-sort-table="markets" data-sort-key="end_date" data-sort-type="text">Closes</th>
                  <th class="sortable" data-sort-table="markets" data-sort-key="last_updated_at" data-sort-type="text">Updated</th>
                  <th class="num sortable" data-sort-table="markets" data-sort-key="volume" data-sort-type="number">Volume</th>
                  <th class="num sortable" data-sort-table="markets" data-sort-key="signal_score" data-sort-type="number">Signal</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody id="markets-table"></tbody>
            </table>
          </div>
        </section>
      </div>
    </div>

    <div class="view" id="view-users">
      <div class="user-workspace">
        <section>
          <div class="bar">
            <h2>Users</h2>
            <div class="tools">
              <input type="search" id="user-search" placeholder="Search users">
              <input type="text" id="user-id-input" placeholder="User handle or wallet">
              <button class="primary" id="update-user-input">Update User</button>
              <button class="toggle" id="candidate-filter">Candidate Users</button>
              <button data-action="update_candidate_users" id="update-candidate-users">Update Candidate Users</button>
              <button data-action="update_users_quick">Update All Users</button>
            </div>
          </div>
          <div class="metrics">
            <div class="metric"><span>Analyzed Users</span><strong id="user-total-count">0</strong></div>
            <div class="metric"><span>Shown</span><strong id="user-shown-count">0</strong></div>
            <div class="metric"><span>Users In Database</span><strong id="user-database-count">0</strong></div>
            <div class="metric"><span>Scanner Candidates</span><strong id="user-candidate-count">0</strong></div>
            <div class="metric"><span>Shown Candidates</span><strong id="user-shown-candidate-count">0</strong></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th class="sortable" data-sort-table="users" data-sort-key="user_key" data-sort-type="text">User</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="info_edge_score" data-sort-type="number">Edge</th>
                  <th class="sortable" data-sort-table="users" data-sort-key="info_edge_tier" data-sort-type="text">Tier</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="monetized_edge_per_trade_usdc" data-sort-type="number">Info $/Trade</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="candidate_jump_capture" data-sort-type="number">Jump $</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="candidate_events" data-sort-type="number">Events</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="skill_index" data-sort-type="number">Skill</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="raw_pnl_usdc" data-sort-type="number">PnL</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="total_trades" data-sort-type="number">Trades</th>
                  <th class="sortable" data-sort-table="users" data-sort-key="last_updated_at" data-sort-type="text">Analyzed</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody id="users-table"></tbody>
            </table>
          </div>
        </section>
        <section>
          <div class="bar">
            <h2 id="trade-title">Trading History</h2>
            <div class="tools">
              <span class="muted" id="trade-summary">Select a user</span>
              <input type="search" id="trade-search" placeholder="Search selected user trades">
              <button id="update-selected-user" disabled>Update User</button>
              <button id="analyze-selected-user" disabled>Analyze User</button>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th class="sortable" data-sort-table="trades" data-sort-key="title" data-sort-type="text">Market</th>
                  <th class="num sortable" data-sort-table="trades" data-sort-key="scanner_signal_score" data-sort-type="number">Signal</th>
                  <th class="sortable" data-sort-table="trades" data-sort-key="market_status" data-sort-type="text">Status</th>
                  <th class="sortable" data-sort-table="trades" data-sort-key="market_end_date" data-sort-type="text">Closes</th>
                  <th class="sortable" data-sort-table="trades" data-sort-key="outcome_summary" data-sort-type="text">Outcomes</th>
                  <th class="num sortable" data-sort-table="trades" data-sort-key="trade_count" data-sort-type="number">Trades</th>
                  <th class="num sortable" data-sort-table="trades" data-sort-key="net_size" data-sort-type="number">Net Shares</th>
                  <th class="num sortable" data-sort-table="trades" data-sort-key="avg_price" data-sort-type="number">Avg Price</th>
                  <th class="num sortable" data-sort-table="trades" data-sort-key="total_notional" data-sort-type="number">Notional</th>
                  <th class="sortable" data-sort-table="trades" data-sort-key="last_timestamp" data-sort-type="number">Last Trade</th>
                  <th class="num sortable" data-sort-table="trades" data-sort-key="closed_pnl" data-sort-type="number">Closed P/L</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody id="trades-table"></tbody>
            </table>
          </div>
        </section>
      </div>
    </div>

    <div class="view" id="view-jobs">
      <section class="jobs-run-list">
        <div class="bar"><h2>Runs</h2><span class="muted">Newest first</span></div>
        <div class="table-wrap jobs-table-wrap">
          <table>
            <thead><tr><th class="sortable" data-sort-table="jobs" data-sort-key="created_at" data-sort-type="text">Started</th><th class="sortable" data-sort-table="jobs" data-sort-key="label" data-sort-type="text">Run</th><th class="sortable" data-sort-table="jobs" data-sort-key="status" data-sort-type="text">Status</th><th></th></tr></thead>
            <tbody id="jobs"></tbody>
          </table>
        </div>
      </section>
      <section class="jobs-log-panel">
        <div class="bar">
          <h2>Log</h2>
          <span class="muted" id="log-title">No run selected</span>
        </div>
        <pre class="log" id="log"></pre>
      </section>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const themeStorageKey = "polymarket-dashboard-theme";
    function themeName() {
      return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    }
    function applyTheme(theme) {
      const nextTheme = theme === "dark" ? "dark" : "light";
      document.documentElement.dataset.theme = nextTheme;
      try { localStorage.setItem(themeStorageKey, nextTheme); } catch (_error) {}
      const button = $("theme-toggle");
      if (button) {
        const dark = nextTheme === "dark";
        button.textContent = dark ? "Light" : "Dark";
        button.setAttribute("aria-pressed", dark ? "true" : "false");
        button.title = dark ? "Switch to light mode" : "Switch to dark mode";
      }
    }
    const actionHelp = {
      update_markets: "API calls: discover/store markets from Polymarket and refresh market metadata and price history in local SQLite.",
      update_listed_markets: "API calls: refresh the markets already listed in this dashboard from Polymarket, then store the results locally.",
      update_market_trades: "API calls: refresh market-level trade rows from Polymarket for the listed markets. Stores results in local SQLite.",
      analyze_scanner: "Local only: scan cached SQLite market prices/trades for jump events and candidate users. No Polymarket API calls.",
      update_users_quick: "API calls: refresh user trades and related market metadata from Polymarket, then rerun local user analysis. Skips price history.",
      update_users_full: "API calls: refresh user trades, related markets, and price history from Polymarket, then rerun local user analysis.",
      update_candidate_users: "API calls: update users identified by Analyze Markets as candidates, then rerun local user analysis for them."
    };
    const tabHelp = {
      overview: "Local view: show the cached dashboard overview from SQLite and run-state files.",
      markets: "Local view: show markets stored in SQLite and controls for market API updates and local analysis.",
      users: "Local view: show analyzed users from SQLite and controls for user API updates or local-only analysis.",
      jobs: "Local view: show background runs and logs from local run-state files."
    };
    const staticButtonHelp = {
      "add-market": "Local only: add this conditionId to the dashboard market list. Use Update Markets afterward to fetch Polymarket data.",
      "update-user-input": "API calls: update the entered user from Polymarket, then rerun local user analysis. Skips price history.",
      "candidate-filter": "Local only: filter the user table to scanner candidate users. No Polymarket API calls.",
      "update-selected-user": "API calls: update the selected user from Polymarket, then rerun local user analysis. Skips price history.",
      "analyze-selected-user": "Local only: rerun scoring for the selected user from cached SQLite data. No Polymarket API calls."
    };
    function setButtonHelp(button, text) {
      if (!button || !text) return;
      button.title = text;
      button.setAttribute("aria-description", text);
    }
    function applyButtonHints(root = document) {
      root.querySelectorAll(".tab[data-tab]").forEach(button => {
        setButtonHelp(button, tabHelp[button.dataset.tab]);
      });
      root.querySelectorAll("[data-action]").forEach(button => {
        setButtonHelp(button, actionHelp[button.getAttribute("data-action")]);
      });
      Object.entries(staticButtonHelp).forEach(([id, text]) => setButtonHelp($(id), text));
    }
    let selectedJob = null;
    let selectedUser = null;
    let currentTab = "overview";
    let activeJobCount = 0;
    let userCandidateOnly = false;
    const loadingState = {
      status: "",
      markets: "",
      users: "",
      trades: "",
      log: "",
    };
    const expandedMarkets = new Set();
    const state = { status: null, users: null, trades: null, markets: null };
    const sortState = {
      overviewMarkets: { key: "signal_score", dir: "desc", type: "number" },
      overviewUsers: { key: "info_edge_score", dir: "desc", type: "number" },
      markets: { key: "signal_score", dir: "desc", type: "number" },
      users: { key: "info_edge_score", dir: "desc", type: "number" },
      jobs: { key: "created_at", dir: "desc", type: "text" },
      trades: { key: "last_timestamp", dir: "desc", type: "number" },
      tradeDetails: { key: "timestamp", dir: "desc", type: "number" }
    };

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[ch]));
    }
    function num(value, digits = 2) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "";
      return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }
    function intNum(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "";
      return Math.round(n).toLocaleString();
    }
    function cls(status) {
      return "status-" + String(status || "").toLowerCase();
    }
    function scoreClass(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "score score-baseline";
      if (n >= 85) return "score score-strong";
      if (n >= 70) return "score score-elevated";
      if (n >= 55) return "score score-watch";
      return "score score-baseline";
    }
    function scoreCell(value, title = "") {
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (!Number.isFinite(n)) return "";
      const titleAttr = title ? ` title="${esc(title)}" alt="${esc(title)}"` : "";
      return `<span class="${scoreClass(n)}"${titleAttr}>${num(n, 1)}</span>`;
    }
    function money(value, digits = 2) {
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (!Number.isFinite(n)) return "";
      const sign = n < 0 ? "-" : "";
      return sign + "$" + Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }
    function shortWallet(value) {
      const text = String(value || "");
      return text.length > 18 ? text.slice(0, 10) + "..." + text.slice(-6) : text;
    }
    function isWallet(value) {
      return /^0x[a-fA-F0-9]{40}$/.test(String(value || "").trim());
    }
    function shortDate(value) {
      const text = String(value || "");
      if (!text) return "";
      return text.replace("T", " ").replace("+00:00", "Z").slice(0, 19);
    }
    function externalLink(url, label, title) {
      const text = label || url || "";
      if (!url) return esc(text);
      return `<a class="external" href="${esc(url)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()" title="${esc(title || text)}">${esc(text)}</a>`;
    }
    function polymarketUserUrl(userKey) {
      const key = String(userKey || "").trim();
      if (!key) return "";
      const profileKey = /^0x[a-fA-F0-9]{40}$/.test(key) || key.startsWith("@") ? key : "@" + key;
      return "https://polymarket.com/profile/" + encodeURIComponent(profileKey);
    }
    function polymarketMarketUrl(row) {
      if (!row) return "";
      const marketSlug = String(row.slug || "").trim();
      const eventSlug = String(row.event_slug || "").trim();
      if (eventSlug && marketSlug && eventSlug !== marketSlug) {
        return "https://polymarket.com/event/" + encodeURIComponent(eventSlug) + "/" + encodeURIComponent(marketSlug);
      }
      const slug = eventSlug || marketSlug;
      return slug ? "https://polymarket.com/event/" + encodeURIComponent(slug) : "";
    }
    function userLink(userKey, short = false, label = "") {
      const text = label || (short ? shortWallet(userKey) : userKey);
      return externalLink(polymarketUserUrl(userKey), text, userKey);
    }
    function userAliasName(row) {
      const key = String(row && row.user_key ? row.user_key : "").trim();
      const fallbackShort = shortWallet(key).toLowerCase();
      const fallbackKey = key.toLowerCase();
      const values = row ? [row.display_name, row.profile_name, row.profile_pseudonym, row.input_user] : [];
      for (const value of values) {
        const text = String(value || "").trim();
        if (!text || isWallet(text)) continue;
        const comparable = text.toLowerCase();
        if (comparable === fallbackKey || comparable === fallbackShort) continue;
        return text;
      }
      return "";
    }
    function userLabel(row, short = false) {
      const key = row && row.user_key ? row.user_key : "";
      const name = userAliasName(row);
      return name || (short ? shortWallet(key) : key);
    }
    function userCell(row, short = false) {
      const key = row && row.user_key ? row.user_key : "";
      const name = userAliasName(row);
      const label = userLabel(row, short);
      const sub = name ? `<div class="muted" title="${esc(key)}">${esc(shortWallet(key))}</div>` : "";
      return `${userLink(key, short, label)}${sub}`;
    }
    function marketLink(row) {
      const label = (row && (row.title || row.slug || row.condition_id)) || "";
      return externalLink(polymarketMarketUrl(row), label || shortWallet(row && row.condition_id), label);
    }
    function safeId(value) {
      return String(value || "").replace(/[^a-zA-Z0-9_-]/g, "_");
    }
    async function api(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error(await response.text());
      return await response.json();
    }
    function emptyRow(cols, label = "No rows") {
      return `<tr><td colspan="${cols}" class="empty">${esc(label)}</td></tr>`;
    }
    function loadingRow(cols, label = "Loading") {
      return `<tr><td colspan="${cols}" class="loading-cell"><span class="loading-spinner"></span>${esc(label)}</td></tr>`;
    }
    function errorRow(cols, error, fallback = "Unable to load rows") {
      const message = error && error.message ? error.message : String(error || fallback);
      return `<tr><td colspan="${cols}" class="empty bad">${esc(message)}</td></tr>`;
    }
    function loadingTab(name) {
      if (name === "status" || name === "log") return "jobs";
      if (name === "trades") return "users";
      return name;
    }
    function updateLoadingChrome() {
      ["overview", "markets", "users", "jobs"].forEach(tabName => {
        const labels = Object.entries(loadingState)
          .filter(([name, label]) => label && loadingTab(name) === tabName)
          .map(([, label]) => label);
        const active = labels.length > 0;
        const button = document.querySelector(`.tab[data-tab="${tabName}"]`);
        const view = $("view-" + tabName);
        if (button) {
          button.classList.toggle("loading", active);
          button.setAttribute("aria-busy", active ? "true" : "false");
        }
        if (view) {
          view.classList.toggle("loading", active);
          view.dataset.loading = labels[0] || "";
        }
      });
    }
    function setLoading(name, active, label = "Loading") {
      loadingState[name] = active ? label : "";
      updateLoadingChrome();
    }
    function nextFrame() {
      return new Promise(resolve => requestAnimationFrame(resolve));
    }
    function sortValue(row, key, type) {
      const value = row ? row[key] : null;
      if (type === "number") {
        const number = Number(value);
        return Number.isFinite(number) ? number : Number.NEGATIVE_INFINITY;
      }
      return String(value ?? "").toLowerCase();
    }
    function sortedRows(rows, tableName) {
      const cfg = sortState[tableName];
      const copy = Array.isArray(rows) ? rows.slice() : [];
      if (!cfg) return copy;
      copy.sort((a, b) => {
        const av = sortValue(a, cfg.key, cfg.type);
        const bv = sortValue(b, cfg.key, cfg.type);
        let cmp = 0;
        if (cfg.type === "number") {
          cmp = av === bv ? 0 : av < bv ? -1 : 1;
        } else {
          cmp = String(av).localeCompare(String(bv));
        }
        return cfg.dir === "asc" ? cmp : -cmp;
      });
      return copy;
    }
    function updateSortIndicators() {
      document.querySelectorAll("th.sortable").forEach(th => {
        const cfg = sortState[th.dataset.sortTable];
        const active = cfg && cfg.key === th.dataset.sortKey;
        th.classList.toggle("sorted", Boolean(active));
        th.classList.toggle("asc", Boolean(active && cfg.dir === "asc"));
        th.classList.toggle("desc", Boolean(active && cfg.dir === "desc"));
      });
    }
    function rerenderTable(tableName) {
      if (["overviewUsers", "users"].includes(tableName) && state.users) renderUsers(state.users);
      if (["overviewMarkets", "markets"].includes(tableName) && state.markets) renderMarkets(state.markets);
      if (tableName === "jobs" && state.status) renderStatus(state.status);
      if (tableName === "trades" && state.trades) renderTrades(state.trades);
      updateSortIndicators();
    }
    function setTab(name) {
      currentTab = name;
      document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === name));
      document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === "view-" + name));
      if (name === "markets" && !state.markets && !loadingState.markets) refreshMarkets();
      if (name === "users" && !state.users && !loadingState.users) refreshUsers();
      if (name === "jobs" && !state.status && !loadingState.status) refreshStatus(false);
    }
    function renderStatus(data) {
      const inv = data.inventory || {};
      const generated = $("generated");
      if (generated) generated.textContent = "Updated " + (inv.generated_at || data.now || "");
      $("m-users").textContent = intNum(inv.user_count || 0);
      $("m-user-trades").textContent = intNum(inv.user_trade_rows || 0);
      $("m-markets").textContent = intNum(inv.market_metadata_rows || 0);
      $("m-prices").textContent = intNum(inv.price_history_assets || 0);
      $("m-trades").textContent = intNum(inv.market_trade_sets || 0);

      const running = (data.jobs || []).filter(j => ["queued", "running"].includes(j.status));
      $("job-state").textContent = running.length ? running.length + " active job" + (running.length === 1 ? "" : "s") : "Idle";
      $("run-note").textContent = running.length ? "Background work is running" : "";

      $("jobs").innerHTML = sortedRows(data.jobs || [], "jobs").map(job => {
        const cancellable = ["queued", "running"].includes(job.status);
        const cancelTitle = "Local job control: stop this background process if it is still running. Does not start a Polymarket API call.";
        const cancel = cancellable ? ` <button data-cancel-job="${esc(job.id)}" title="${esc(cancelTitle)}" aria-description="${esc(cancelTitle)}">Cancel</button>` : "";
        const selected = selectedJob === job.id ? " selected" : "";
        const started = job.started_at || job.created_at || "";
        const logTitle = "Local only: show this run's saved log file. No Polymarket API calls.";
        return `<tr class="clickable${selected}" data-job-row="${esc(job.id)}">
          <td title="${esc(started)}">${esc(shortDate(started))}</td>
          <td title="${esc(job.label)}">${esc(job.label)}</td>
          <td class="${cls(job.status)}">${esc(job.status)}</td>
          <td><button data-job="${esc(job.id)}" title="${esc(logTitle)}" aria-description="${esc(logTitle)}">Log</button>${cancel}</td>
        </tr>`;
      }).join("") || emptyRow(4, "No runs recorded");

      document.querySelectorAll("[data-job-row]").forEach(row => {
        row.onclick = () => selectJob(row.getAttribute("data-job-row"));
      });
      document.querySelectorAll("[data-job]").forEach(button => {
        button.onclick = (event) => {
          event.stopPropagation();
          selectJob(button.getAttribute("data-job"));
        };
      });
      document.querySelectorAll("[data-cancel-job]").forEach(button => {
        button.onclick = async (event) => {
          event.stopPropagation();
          button.disabled = true;
          try {
            await api("/api/jobs/" + encodeURIComponent(button.getAttribute("data-cancel-job")) + "/cancel", { method: "POST" });
            await refreshStatus(true);
          } finally {
            button.disabled = false;
          }
        };
      });
      updateSortIndicators();
    }
    function renderUsers(data) {
      const candidateButton = $("candidate-filter");
      candidateButton.classList.toggle("active", userCandidateOnly);
      candidateButton.textContent = userCandidateOnly ? "All Users" : "Candidate Users";
      setButtonHelp(
        candidateButton,
        userCandidateOnly
          ? "Local only: clear the candidate filter and show all users from SQLite. No Polymarket API calls."
          : "Local only: filter the user table to scanner candidate users. No Polymarket API calls."
      );
      $("users-count").textContent = intNum(data.count || 0) + (userCandidateOnly ? " candidate users" : " users");
      $("user-total-count").textContent = intNum(data.total_count || 0);
      $("user-shown-count").textContent = intNum(data.count || 0);
      $("user-database-count").textContent = intNum(data.database_user_count || 0);
      $("user-candidate-count").textContent = intNum(data.candidate_count || 0);
      $("user-shown-candidate-count").textContent = intNum(data.shown_candidate_count || 0);
      $("update-candidate-users").disabled = !Number(data.candidate_count || 0);
      $("signal-high-users").textContent = intNum(data.high_signal_count || 0);
      $("signal-candidates").textContent = intNum(data.candidate_count || 0);
      $("signal-analyzed").textContent = intNum(data.total_count || 0);
      const overviewRows = sortedRows(data.top_info_edge || data.rows || [], "overviewUsers");
      $("overview-users").innerHTML = overviewRows.slice(0, 8).map(row => {
        const selected = selectedUser === row.user_key ? " selected" : "";
        return `<tr class="clickable${selected}" data-user-key="${esc(row.user_key)}">
          <td class="num">${scoreCell(row.info_edge_score, row.evidence_summary)}</td>
          <td title="${esc([row.display_name, row.user_key, row.profile_name, row.profile_pseudonym, row.input_user].filter(Boolean).join(' | '))}">${userCell(row, true)}</td>
          <td><span class="pill">${esc(row.info_edge_tier || "Unscored")}</span></td>
          <td class="evidence" title="${esc(row.evidence_summary)}">${esc(row.evidence_summary)}</td>
        </tr>`;
      }).join("") || emptyRow(4, "No user signals yet");
      $("users-table").innerHTML = sortedRows(data.rows || data.top_info_edge || [], "users").map(row => {
        const selected = selectedUser === row.user_key ? " selected" : "";
        const scannerText = row.candidate_confidence ? `Scanner: ${row.candidate_confidence} | z ${num(row.candidate_timing_z, 2)} | ${intNum(row.candidate_events)} events` : "";
        const evidenceTitle = [row.evidence_summary, scannerText, row.source_label].filter(Boolean).join(" | ");
        const edgePerTradeTitle = row.monetized_edge_per_trade_usdc == null ? "Run user analysis to calculate info edge per trade" : `Conservative per-trade monetization: lower of realized PnL/trade and expected edge/trade. ${evidenceTitle}`;
        const updateTitle = "API calls: fetch latest Polymarket trades and related market metadata for this user, then rerun local user analysis. Skips price history.";
        const analyzeTitle = "Local only: rerun this user's scoring from cached SQLite data. No Polymarket API calls.";
        return `<tr class="clickable${selected}" data-user-key="${esc(row.user_key)}">
          <td title="${esc([row.display_name, row.user_key, row.profile_name, row.profile_pseudonym, row.input_user].filter(Boolean).join(' | '))}">${userCell(row, true)}</td>
          <td class="num">${scoreCell(row.info_edge_score, evidenceTitle)}</td>
          <td><span class="pill" title="${esc(evidenceTitle)}">${esc(row.info_edge_tier || "Unscored")}</span></td>
          <td class="num" title="${esc(edgePerTradeTitle)}">${money(row.monetized_edge_per_trade_usdc)}</td>
          <td class="num" title="${esc(scannerText)}">${money(row.candidate_jump_capture, 0)}</td>
          <td class="num" title="${esc(scannerText)}">${intNum(row.candidate_events)}</td>
          <td class="num">${num(row.skill_index)}</td>
          <td class="num">${money(row.raw_pnl_usdc, 0)}</td>
          <td class="num">${intNum(row.total_trades)}</td>
          <td title="${esc(row.last_updated_at)}">${esc(shortDate(row.last_updated_at))}</td>
          <td><button class="mini" data-update-user="${esc(row.user_key)}" title="${esc(updateTitle)}" aria-description="${esc(updateTitle)}">Update</button> <button class="mini" data-analyze-user="${esc(row.user_key)}" title="${esc(analyzeTitle)}" aria-description="${esc(analyzeTitle)}">Analyze</button></td>
        </tr>`;
      }).join("") || emptyRow(11, userCandidateOnly ? "No candidate users matched" : "No users yet");
      document.querySelectorAll("[data-user-key]").forEach(row => {
        row.onclick = () => selectUser(row.getAttribute("data-user-key"));
      });
      document.querySelectorAll("[data-update-user]").forEach(button => {
        button.onclick = async (event) => {
          event.stopPropagation();
          button.disabled = true;
          try {
            await startJob("update_user", { user_key: button.getAttribute("data-update-user") });
          } finally {
            button.disabled = false;
          }
        };
      });
      document.querySelectorAll("[data-analyze-user]").forEach(button => {
        button.onclick = async (event) => {
          event.stopPropagation();
          button.disabled = true;
          try {
            await startJob("analyze_user", { user_key: button.getAttribute("data-analyze-user") });
          } finally {
            button.disabled = false;
          }
        };
      });
      updateSortIndicators();
    }
    function renderTrades(data) {
      const groups = sortedRows(data && data.groups ? data.groups : [], "trades");
      const userKey = data && data.user_key ? data.user_key : selectedUser;
      $("trade-title").textContent = userKey ? "Trading History: " + userKey : "Trading History";
      $("update-selected-user").disabled = !userKey;
      $("analyze-selected-user").disabled = !userKey;
      if (!userKey) {
        $("trade-summary").textContent = "Select a user";
      } else if (data && data.error) {
        $("trade-summary").textContent = data.error;
      } else if (data) {
        $("trade-summary").textContent = intNum(data.market_count || 0) + " markets, " + intNum(data.matched || 0) + " trades shown of " + intNum(data.total || 0) + " database trades";
      }
      $("trades-table").innerHTML = groups.map(group => {
        const marketKey = group.market_key || group.condition_id || group.title;
        const expanded = expandedMarkets.has(marketKey);
        const buttonLabel = expanded ? "-" : "+";
        const expandTitle = expanded
          ? "Local only: collapse cached trade fills for this market. No Polymarket API calls."
          : "Local only: expand cached trade fills for this market. No Polymarket API calls.";
        const detail = expanded ? renderTradeDetail(group.trades || []) : "";
        const status = group.market_status || "unknown";
        const statusLabel = group.market_status_label || "Unknown";
        return `<tr class="market-row-${esc(status)}">
          <td><button class="expand" data-market-key="${esc(marketKey)}" title="${esc(expandTitle)}" aria-description="${esc(expandTitle)}">${buttonLabel}</button></td>
          <td title="${esc(group.title)}">${marketLink(group)}</td>
          <td class="num">${tradeSignalCell(group)}</td>
          <td><span class="market-status-pill market-status-${esc(status)}">${esc(statusLabel)}</span></td>
          <td title="${esc(group.market_end_date)}">${esc(shortDate(group.market_end_date))}</td>
          <td title="${esc(group.outcome_summary)}">${esc(group.outcome_summary)}</td>
          <td class="num">${intNum(group.trade_count)}</td>
          <td class="num">${num(group.net_size)}</td>
          <td class="num">${num(group.avg_price, 4)}</td>
          <td class="num">${num(group.total_notional)}</td>
          <td title="${esc(group.last_time_utc)}">${esc(group.last_time_utc)}</td>
          <td class="num">${closedPnlCell(group)}</td>
          <td>${marketActionButtons(group)}</td>
        </tr>${detail}`;
      }).join("") || emptyRow(13, selectedUser ? "No database trades found" : "Select a user to load trades");
      document.querySelectorAll("[data-market-key]").forEach(button => {
        button.onclick = () => {
          const key = button.getAttribute("data-market-key");
          if (!key) return;
          if (expandedMarkets.has(key)) expandedMarkets.delete(key);
          else expandedMarkets.add(key);
          renderTrades(state.trades);
        };
      });
      bindMarketUpdateButtons();
      updateSortIndicators();
    }
    function renderTradeDetail(rows) {
      const body = sortedRows(rows || [], "tradeDetails").map(row => {
        const tx = row.transaction_hash ? `<div class="muted" title="${esc(row.transaction_hash)}">${esc(shortWallet(row.transaction_hash))}</div>` : "";
        return `<tr>
          <td title="${esc(row.timestamp_utc)}">${esc(row.timestamp_utc)}</td>
          <td><span class="pill">${esc(row.side)}</span></td>
          <td>${esc(row.outcome)}</td>
          <td class="num">${num(row.price, 4)}</td>
          <td class="num">${num(row.size)}</td>
          <td class="num">${num(row.notional)}</td>
          <td>${tx}</td>
        </tr>`;
      }).join("") || `<tr><td colspan="7" class="empty">No fills for this market</td></tr>`;
      return `<tr class="detail-row"><td colspan="13">
        <table class="nested">
          <thead><tr><th>Time</th><th>Side</th><th>Outcome</th><th class="num">Price</th><th class="num">Size</th><th class="num">Notional</th><th>Tx</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </td></tr>`;
    }
    function marketStatusPill(row) {
      const status = row.market_status || "unknown";
      const label = row.market_status_label || "Unknown";
      return `<span class="market-status-pill market-status-${esc(status)}">${esc(label)}</span>`;
    }
    function closedPnlCell(row) {
      const value = Number(row && row.closed_pnl);
      if (!Number.isFinite(value)) return "";
      const tone = value > 0 ? "good" : value < 0 ? "bad" : "muted";
      const title = (row && row.closed_pnl_note) || "";
      return `<span class="${tone}" title="${esc(title)}">${num(value)}</span>`;
    }
    function tradeSignalCell(row) {
      const eventCount = Number(row && row.scanner_user_event_count);
      const capture = Number(row && row.scanner_user_jump_capture);
      const score = row ? row.scanner_signal_score : null;
      if (!Number.isFinite(eventCount) || eventCount <= 0) return "";
      const captureTone = Number.isFinite(capture) && capture > 0 ? "good" : Number.isFinite(capture) && capture < 0 ? "bad" : "muted";
      const parts = [
        "Scanner jump exposure for this selected user/market",
        Number.isFinite(capture) ? "jump capture " + money(capture, 2) : "",
        eventCount ? intNum(eventCount) + " event" + (eventCount === 1 ? "" : "s") : "",
        row.scanner_user_last_jump_time_utc ? "last jump " + row.scanner_user_last_jump_time_utc : "",
        Number.isFinite(Number(row.scanner_user_max_move)) ? "max move " + num(row.scanner_user_max_move, 3) : "",
        Number.isFinite(Number(row.scanner_user_avg_directional_ratio)) ? "directional ratio " + num(row.scanner_user_avg_directional_ratio, 2) : ""
      ].filter(Boolean).join(" | ");
      return `<div title="${esc(parts)}">${scoreCell(score, parts)}<div class="${captureTone}">${money(capture, 0)}</div><div class="muted">${intNum(eventCount)} evt</div></div>`;
    }
    function marketActionButtons(row) {
      const id = row.condition_id || "";
      const updateTitle = "API calls: refresh this market's stored metadata and price history from Polymarket. Does not rerun scanner analysis by itself.";
      if (!id) return "";
      const update = `<button class="mini" data-update-market="${esc(id)}" title="${esc(updateTitle)}" aria-description="${esc(updateTitle)}">Update</button>`;
      return update;
    }
    function bindMarketUpdateButtons() {
      document.querySelectorAll("[data-update-market]").forEach(button => {
        button.onclick = async (event) => {
          event.stopPropagation();
          button.disabled = true;
          try {
            await startJob("update_market", { market_id: button.getAttribute("data-update-market") });
          } finally {
            button.disabled = false;
          }
        };
      });
    }
    function sourcePills(row) {
      const labels = String(row.source_summary || "").split(",").map(item => item.trim()).filter(Boolean);
      return labels.map(label => `<span class="pill">${esc(label)}</span>`).join(" ") || `<span class="pill">Unknown</span>`;
    }
    function signalSummary(row) {
      if (!row.scanner_signal) return "";
      const move = Number(row.scanner_move);
      const moveText = Number.isFinite(move) ? num(move, 3) : "";
      const timeText = shortDate(row.scanner_jump_time_utc);
      const score = scoreCell(row.signal_score);
      const tier = row.signal_tier ? `<div class="muted">${esc(row.signal_tier)}</div>` : "";
      return `${score}${moveText ? `<div class="muted">move ${esc(moveText)}</div>` : ""}${timeText ? `<div class="muted">${esc(timeText)}</div>` : ""}${tier}`;
    }
    function renderMarkets(data) {
      const rows = sortedRows(data.rows || [], "markets");
      const totalMarkets = Number(data.total_market_count ?? data.listed_count ?? data.count ?? rows.length);
      const shownMarkets = Number(data.shown_count ?? rows.length);
      $("market-count").textContent = shownMarkets === totalMarkets
        ? intNum(totalMarkets || 0) + " markets"
        : "Showing " + intNum(shownMarkets || 0) + " of " + intNum(totalMarkets || 0) + " markets";
      $("mk-listed").textContent = intNum(totalMarkets || 0);
      $("mk-shown").textContent = intNum(shownMarkets || 0);
      $("mk-database").textContent = intNum(data.database_count || 0);
      $("mk-events").textContent = intNum(data.scanner_event_count || 0);
      $("mk-watermark").textContent = shortDate((data.watermark || {}).updated_at) || "None";
      $("mk-trade-watermark").textContent = shortDate((data.trade_watermark || {}).updated_at) || "None";
      $("mk-analysis-watermark").textContent = shortDate((data.analysis_watermark || {}).updated_at) || "None";
      $("signal-events").textContent = intNum(data.scanner_event_count || 0);
      $("signal-last-analysis").textContent = shortDate((data.analysis_watermark || {}).updated_at) || "None";
      $("mk-added").textContent = intNum(data.added_count || 0);
      $("overview-markets").innerHTML = sortedRows(data.rows || [], "overviewMarkets").filter(row => row.scanner_signal).slice(0, 8).map(row => {
        return `<tr class="market-row-${esc(row.market_status || "unknown")}">
          <td class="num">${scoreCell(row.signal_score)}</td>
          <td title="${esc(row.title)}">${marketLink(row)}</td>
          <td class="num">${intNum(row.scanner_participants)}</td>
          <td title="${esc(row.scanner_jump_time_utc)}">${esc(shortDate(row.scanner_jump_time_utc))}</td>
        </tr>`;
      }).join("") || emptyRow(4, "No market signal events yet");
      $("markets-table").innerHTML = rows.map(row => {
        return `<tr class="market-row-${esc(row.market_status || "unknown")}">
          <td title="${esc(row.title)}">${marketLink(row)}<div class="muted" title="${esc(row.condition_id)}">${esc(shortWallet(row.condition_id))}</div></td>
          <td>${sourcePills(row)}</td>
          <td>${marketStatusPill(row)}</td>
          <td title="${esc(row.end_date)}">${esc(shortDate(row.end_date))}</td>
          <td title="${esc(row.last_updated_at)}">${esc(shortDate(row.last_updated_at))}</td>
          <td class="num">${num(row.volume)}</td>
          <td class="num">${signalSummary(row)}</td>
          <td>${marketActionButtons(row)}</td>
        </tr>`;
      }).join("") || emptyRow(8, "No markets yet");
      bindMarketUpdateButtons();
      updateSortIndicators();
    }
    async function refreshStatus(force = false, quiet = false) {
      const previousActiveJobs = activeJobCount;
      if (!quiet) {
        setLoading("status", true, force ? "Refreshing runs" : "Loading runs");
        if (!state.status) $("jobs").innerHTML = loadingRow(4, "Loading runs");
        await nextFrame();
      }
      try {
        state.status = await api("/api/status" + (force ? "?refresh=1" : ""));
        activeJobCount = (state.status.jobs || []).filter(j => ["queued", "running"].includes(j.status)).length;
        renderStatus(state.status);
        if (selectedJob) refreshLog(quiet);
        if (previousActiveJobs > 0 && activeJobCount === 0) {
          await Promise.all([refreshUsers(), refreshMarkets()]);
          if (selectedUser) await refreshTrades();
        }
      } catch (error) {
        if (!quiet) {
          $("jobs").innerHTML = errorRow(4, error, "Unable to load runs");
          $("job-state").textContent = "Run load failed";
        }
      } finally {
        if (!quiet) setLoading("status", false);
      }
    }
    async function refreshUsers() {
      const query = $("user-search").value.trim();
      const params = new URLSearchParams({ q: query, limit: "1000" });
      if (userCandidateOnly) params.set("candidates", "1");
      const label = userCandidateOnly ? "Loading candidate users" : "Loading users";
      setLoading("users", true, label);
      if (!state.users) {
        $("users-count").textContent = label;
        $("overview-users").innerHTML = loadingRow(4, label);
        $("users-table").innerHTML = loadingRow(11, label);
      }
      await nextFrame();
      try {
        state.users = await api("/api/users?" + params.toString());
        renderUsers(state.users);
      } catch (error) {
        $("overview-users").innerHTML = errorRow(4, error, "Unable to load users");
        $("users-table").innerHTML = errorRow(11, error, "Unable to load users");
      } finally {
        setLoading("users", false);
      }
    }
    async function refreshMarkets() {
      const query = $("market-search") ? $("market-search").value.trim() : "";
      const label = "Loading markets";
      setLoading("markets", true, label);
      if (!state.markets) {
        $("market-count").textContent = label;
        $("overview-markets").innerHTML = loadingRow(4, label);
        $("markets-table").innerHTML = loadingRow(8, label);
      }
      await nextFrame();
      try {
        state.markets = await api("/api/markets?q=" + encodeURIComponent(query));
        renderMarkets(state.markets);
      } catch (error) {
        $("overview-markets").innerHTML = errorRow(4, error, "Unable to load markets");
        $("markets-table").innerHTML = errorRow(8, error, "Unable to load markets");
      } finally {
        setLoading("markets", false);
      }
    }
    async function addMarket(marketId, source = "manual") {
      const id = String(marketId || "").trim();
      if (!id) return;
      setLoading("markets", true, "Adding market");
      await nextFrame();
      try {
        state.markets = await api("/api/markets", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ market_id: id, source })
        });
        renderMarkets(state.markets);
      } finally {
        setLoading("markets", false);
      }
    }
    async function selectUser(userKey) {
      if (!userKey) return;
      selectedUser = userKey;
      expandedMarkets.clear();
      setTab("users");
      renderUsers(state.users || {});
      $("trade-title").textContent = "Trading History: " + userKey;
      await refreshTrades();
      $("update-selected-user").disabled = false;
      $("analyze-selected-user").disabled = false;
    }
    async function refreshTrades() {
      if (!selectedUser) {
        renderTrades(null);
        return;
      }
      const query = $("trade-search").value.trim();
      setLoading("trades", true, "Loading trades");
      $("trade-summary").textContent = "Loading trades";
      $("trades-table").innerHTML = loadingRow(13, "Loading trades");
      await nextFrame();
      try {
        state.trades = await api("/api/users/" + encodeURIComponent(selectedUser) + "/trades?limit=1000&q=" + encodeURIComponent(query));
        renderTrades(state.trades);
      } catch (error) {
        $("trade-summary").textContent = "Trade load failed";
        $("trades-table").innerHTML = errorRow(13, error, "Unable to load trades");
      } finally {
        setLoading("trades", false);
      }
    }
    async function refreshAll(force = false) {
      await Promise.all([refreshStatus(force), refreshUsers(), refreshMarkets()]);
    }
    async function selectJob(id) {
      selectedJob = id;
      $("log-title").textContent = id;
      setTab("jobs");
      if (state.status) renderStatus(state.status);
      await refreshLog();
    }
    async function refreshLog(quiet = false) {
      if (!selectedJob) return;
      if (!quiet) {
        setLoading("log", true, "Loading log");
        $("log").textContent = "Loading log...";
        await nextFrame();
      }
      try {
        const data = await api("/api/jobs/" + encodeURIComponent(selectedJob) + "/log");
        $("log").textContent = data.log || "";
        $("log").scrollTop = $("log").scrollHeight;
      } catch (error) {
        $("log").textContent = String(error);
      } finally {
        if (!quiet) setLoading("log", false);
      }
    }
    async function startJob(action, params = {}) {
      const data = await api("/api/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action, ...params})
      });
      selectedJob = data.job && data.job.id ? data.job.id : selectedJob;
      await refreshStatus(true);
      if (currentTab === "jobs") await refreshLog();
    }
    document.querySelectorAll(".tab").forEach(button => {
      button.onclick = () => setTab(button.dataset.tab);
    });
    const themeButton = $("theme-toggle");
    if (themeButton) {
      applyTheme(themeName());
      themeButton.onclick = () => applyTheme(themeName() === "dark" ? "light" : "dark");
    }
    applyButtonHints();
    document.querySelectorAll("th.sortable").forEach(th => {
      th.onclick = () => {
        const tableName = th.dataset.sortTable;
        const key = th.dataset.sortKey;
        const type = th.dataset.sortType || "text";
        const cfg = sortState[tableName];
        if (!cfg || !key) return;
        if (cfg.key === key) {
          cfg.dir = cfg.dir === "asc" ? "desc" : "asc";
        } else {
          cfg.key = key;
          cfg.type = type;
          cfg.dir = type === "number" ? "desc" : "asc";
        }
        if (tableName === "trades") expandedMarkets.clear();
        rerenderTable(tableName);
      };
    });
    document.querySelectorAll("[data-action]").forEach(button => {
      button.onclick = async () => {
        button.disabled = true;
        try {
          await startJob(button.getAttribute("data-action"));
        } catch (error) {
          $("log").textContent = String(error);
          setTab("jobs");
        } finally {
          button.disabled = false;
        }
      };
    });
    const refreshButton = $("refresh");
    if (refreshButton) refreshButton.onclick = () => refreshAll(true);
    $("add-market").onclick = async () => {
      const input = $("market-id-input");
      const value = input.value.trim();
      if (!value) return;
      try {
        await addMarket(value, "manual");
        input.value = "";
      } catch (error) {
        $("log").textContent = String(error);
        setTab("jobs");
      }
    };
    $("market-id-input").onkeydown = (event) => {
      if (event.key === "Enter") $("add-market").click();
    };
    $("user-search").oninput = () => refreshUsers();
    $("candidate-filter").onclick = () => {
      userCandidateOnly = !userCandidateOnly;
      refreshUsers();
    };
    $("update-user-input").onclick = async () => {
      const input = $("user-id-input");
      const value = input.value.trim();
      if (!value) return;
      try {
        await startJob("update_user", { user_key: value });
        input.value = "";
      } catch (error) {
        $("log").textContent = String(error);
        setTab("jobs");
      }
    };
    $("user-id-input").onkeydown = (event) => {
      if (event.key === "Enter") $("update-user-input").click();
    };
    $("trade-search").oninput = () => refreshTrades();
    $("market-search").oninput = () => refreshMarkets();
    $("update-selected-user").onclick = async () => {
      if (!selectedUser) return;
      await startJob("update_user", { user_key: selectedUser });
    };
    $("analyze-selected-user").onclick = async () => {
      if (!selectedUser) return;
      await startJob("analyze_user", { user_key: selectedUser });
    };
    renderTrades(null);
    refreshAll(false);
    setInterval(() => refreshStatus(false, true), 5000);
  </script>
</body>
</html>
"""


CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


def _send_response(handler: BaseHTTPRequestHandler, body: bytes, content_type: str, status: int = 200) -> None:
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except CLIENT_DISCONNECT_ERRORS:
        return


def _json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    _send_response(handler, body, "application/json; charset=utf-8", status=status)


def _text_response(handler: BaseHTTPRequestHandler, body: str, content_type: str = "text/html") -> None:
    data = body.encode("utf-8")
    _send_response(handler, data, f"{content_type}; charset=utf-8")


def build_job_command(
    action: str,
    body: Dict[str, Any],
    *,
    data_dir: str = "data",
    reports_dir: str = "reports",
) -> Tuple[str, List[str]]:
    if action in JOB_COMMANDS:
        return JOB_COMMANDS[action]

    if action == "update_user":
        raw_user_key = str(body.get("user_key") or "").strip()
        user_key = clean_user_identifier(raw_user_key)
        if not user_key:
            raise ValueError("user_key is required")
        return (
            f"Update and analyze user {user_key}",
            app_command("update", "users", "--user", user_key, "--skip-price-history", "--analyze-after"),
        )

    if action == "analyze_user":
        raw_user_key = str(body.get("user_key") or "").strip()
        user_key = clean_user_identifier(raw_user_key)
        if not user_key:
            raise ValueError("user_key is required")
        return (
            f"Analyze user {user_key}",
            app_command("analyze", "luck", "--user", user_key),
        )

    if action == "update_candidate_users":
        user_keys = scanner_candidate_user_keys(reports_dir, data_dir=data_dir)
        if not user_keys:
            raise ValueError("no scanner candidate users found; run Analyze Markets first")
        users_file = write_user_keys_file(user_keys, ROOT, prefix="candidate_users")
        return (
            f"Update and analyze {len(user_keys)} candidate users",
            app_command(
                "update",
                "users",
                "--users-file",
                str(users_file),
                "--skip-price-history",
                "--analyze-after",
                "--no-seed-from-candidates",
            ),
        )

    if action == "update_market":
        condition_id = normalize_condition_id(body.get("market_id"))
        if not condition_id:
            raise ValueError("market_id must be a 0x-prefixed conditionId")
        add_market_to_list(condition_id, source="update")
        return (
            f"Update market {condition_id}",
            app_command(
                "update",
                "markets",
                "--market",
                condition_id,
                "--force-market-refresh",
            ),
        )

    if action == "update_listed_markets":
        market_ids = listed_market_ids(data_dir, reports_dir)
        if not market_ids:
            raise ValueError("the market list is empty")
        markets_file = write_market_ids_file(market_ids, ROOT)
        command = app_command(
            "update",
            "markets",
            "--force-market-refresh",
            "--markets-file",
            str(markets_file),
        )
        return (f"Update {len(market_ids)} markets", command)

    raise ValueError(f"unknown action: {action}")


def safe_job_id(value: str) -> str:
    return "".join(ch for ch in str(value or "") if ch.isalnum() or ch in "._-")


def cancel_job(job_id: str) -> Dict[str, Any]:
    safe_id = safe_job_id(job_id)
    if not safe_id:
        raise ValueError("job id is required")
    job_path = jobs_dir(ROOT) / f"{safe_id}.json"
    job = read_json(job_path, default={})
    if not job:
        raise ValueError("job not found")

    pid = job.get("worker_pid")
    if process_is_running(pid):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.kill(int(pid), 15)
            except (OSError, TypeError, ValueError):
                pass

    return update_job(
        ROOT,
        job,
        status="cancelled",
        returncode=job.get("returncode") if job.get("returncode") is not None else -1,
        finished_at=utc_now_iso(),
    )


def _run_job(job: Dict[str, Any]) -> None:
    log_path = Path(str(job["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(job["command"]),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        update_job(
            ROOT,
            job,
            status="running",
            started_at=utc_now_iso(),
            dashboard_pid=os.getpid(),
            worker_pid=process.pid,
        )
        returncode = process.wait()
    current_job = read_json(jobs_dir(ROOT) / f"{job.get('id')}.json", default={})
    if current_job.get("status") == "cancelled":
        return
    status = "success" if returncode == 0 else "failed"
    if returncode in {130, -2, -1073741510}:
        status = "cancelled"
    update_job(
        ROOT,
        job,
        status=status,
        returncode=returncode,
        finished_at=utc_now_iso(),
    )


class DashboardHandler(BaseHTTPRequestHandler):
    data_dir = "data"
    reports_dir = "reports"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/":
            _text_response(self, INDEX_HTML)
            return
        if parsed.path == "/api/status":
            force = params.get("refresh") == ["1"]
            payload = {
                "now": utc_now_iso(),
                "inventory": build_inventory(
                    self.data_dir,
                    self.reports_dir,
                    root=ROOT,
                    cache_seconds=0 if force else 60,
                ),
                "runs": list_latest_runs(ROOT),
                "jobs": list_jobs(ROOT, limit=50),
                "watermarks": load_watermarks(self.data_dir),
            }
            _json_response(self, payload)
            return
        if parsed.path == "/api/users":
            query = str((params.get("q") or [""])[0])
            limit = safe_int((params.get("limit") or ["200"])[0]) or 200
            candidate_only = (params.get("candidates") or ["0"])[0] in {"1", "true", "yes"}
            _json_response(
                self,
                load_user_analysis(
                    self.data_dir,
                    self.reports_dir,
                    query=query,
                    limit=limit,
                    candidate_only=candidate_only,
                ),
            )
            return
        if parsed.path.startswith("/api/users/") and parsed.path.endswith("/trades"):
            raw_user = parsed.path[len("/api/users/") : -len("/trades")]
            user_key = unquote(raw_user).strip("/")
            query = str((params.get("q") or [""])[0])
            limit = safe_int((params.get("limit") or ["1000"])[0]) or 1000
            payload = load_user_trade_history(
                self.data_dir,
                user_key,
                reports_dir=self.reports_dir,
                query=query,
                limit=limit,
            )
            status = 400 if payload.get("error") == "invalid user key" else 200
            _json_response(self, payload, status=status)
            return
        if parsed.path == "/api/scanner":
            query = str((params.get("q") or [""])[0])
            _json_response(self, load_scanner(self.reports_dir, query=query, data_dir=self.data_dir))
            return
        if parsed.path == "/api/markets":
            query = str((params.get("q") or [""])[0])
            limit = safe_int((params.get("limit") or ["1000"])[0]) or 1000
            _json_response(self, load_markets(self.data_dir, self.reports_dir, query=query, limit=limit))
            return
        if parsed.path == "/api/jobs":
            _json_response(self, {"jobs": list_jobs(ROOT, limit=30)})
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/log"):
            job_id = parsed.path[len("/api/jobs/") : -len("/log")]
            safe_id = "".join(ch for ch in job_id if ch.isalnum() or ch in "._-")
            log_path = jobs_dir(ROOT) / f"{safe_id}.log"
            log = ""
            if log_path.exists():
                log = log_path.read_text(encoding="utf-8", errors="replace")[-120_000:]
            _json_response(self, {"id": safe_id, "log": log})
            return
        _json_response(self, {"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            job_id = parsed.path[len("/api/jobs/") : -len("/cancel")]
            try:
                job = cancel_job(job_id)
            except ValueError as exc:
                _json_response(self, {"error": html.escape(str(exc))}, status=400)
                return
            _json_response(self, {"job": job})
            return

        if parsed.path not in {"/api/jobs", "/api/markets"}:
            _json_response(self, {"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            _json_response(self, {"error": "invalid json"}, status=400)
            return

        if parsed.path == "/api/markets":
            condition_id = normalize_condition_id(body.get("market_id"))
            if not condition_id:
                _json_response(self, {"error": "market_id must be a 0x-prefixed conditionId"}, status=400)
                return
            source = str(body.get("source") or "manual")
            add_market_to_list(condition_id, source=source)
            _json_response(self, load_markets(self.data_dir, self.reports_dir))
            return

        action = str(body.get("action") or "")
        try:
            label, command = build_job_command(
                action,
                body,
                data_dir=self.data_dir,
                reports_dir=self.reports_dir,
            )
        except ValueError as exc:
            _json_response(self, {"error": html.escape(str(exc))}, status=400)
            return

        job = create_job(ROOT, label=label, command=command, cwd=ROOT)
        thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
        thread.start()
        _json_response(self, {"job": job})


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    data_dir: str = "data",
    reports_dir: str = "reports",
) -> None:
    DashboardHandler.data_dir = data_dir
    DashboardHandler.reports_dir = reports_dir
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}"
    print(f"Dashboard server running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve_dashboard()
