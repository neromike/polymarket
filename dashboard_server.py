from __future__ import annotations

import csv
import html
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from functools import lru_cache
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
    utc_now_iso,
)
from utils import parse_jsonish_list, safe_float, safe_int, to_iso_utc


ROOT = Path(__file__).resolve().parent
USER_LIST_CANDIDATE_CONFIDENCES = {"monitor", "candidate", "high", "very_high"}


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
    "build_dashboards": ("Build static dashboards", app_command("dashboard", "build")),
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


def read_csv_rows(path: Path, limit: int = 5000) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


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
    if confidence_base is None and timing_z is None and events is None:
        return None

    z_score = clamp_score(45.0 + (timing_z or 0.0) * 9.0)
    event_score = clamp_score((events or 0) * 10.0 + 25.0)
    if confidence_base is None:
        confidence_base = 50.0
    return clamp_score(max(confidence_base, 0.45 * z_score + 0.35 * event_score + 0.20 * confidence_base))


def market_jump_signal_score(move: Any, participants: Any = None) -> float:
    abs_move = abs(safe_float(move) or 0.0)
    participant_count = safe_int(participants) or 0
    return clamp_score(abs_move * 260.0 + min(participant_count, 60) * 0.75)


def metric_signal_score(row: Dict[str, Any]) -> float | None:
    skill_index = safe_float(row.get("skill_index"))
    skill_z = safe_float(row.get("skill_z"))
    luck_index = safe_float(row.get("luck_index"))
    raw_pnl = safe_float(row.get("raw_pnl_usdc"))
    gross_notional = safe_float(row.get("gross_trade_notional_usdc"))
    resolved = safe_int(row.get("resolved_markets") or row.get("number_of_resolved_markets")) or 0
    trades = safe_int(row.get("total_trades")) or 0

    if skill_index is None and skill_z is None and luck_index is None and raw_pnl is None:
        return None

    skill_percentile = skill_index if skill_index is not None else clamp_score(50.0 + (skill_z or 0.0) * 16.0)
    skill_z_score = clamp_score(50.0 + (skill_z or 0.0) * 16.0)
    luck_outlier = luck_index if luck_index is not None else 50.0
    roi = (raw_pnl / gross_notional) if raw_pnl is not None and gross_notional and gross_notional > 0 else 0.0
    profit_score = clamp_score(50.0 + roi * 350.0)
    sample_score = clamp_score(min(resolved, 40) * 1.6 + min(trades, 5000) / 5000 * 36.0)

    return clamp_score(
        0.38 * skill_percentile
        + 0.20 * skill_z_score
        + 0.15 * luck_outlier
        + 0.15 * profit_score
        + 0.12 * sample_score
    )


def evidence_summary(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    confidence = str(row.get("candidate_confidence") or "").strip()
    timing_z = safe_float(row.get("candidate_timing_z"))
    candidate_events = safe_int(row.get("candidate_events"))
    if confidence:
        scanner = f"{confidence} timing"
        if timing_z is not None:
            scanner += f" z={timing_z:.2f}"
        if candidate_events is not None:
            scanner += f" across {candidate_events} events"
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
    if pnl is not None:
        parts.append(f"PnL ${pnl:,.0f}")

    return "; ".join(parts) if parts else "Needs user update"


def apply_information_edge_score(row: Dict[str, Any], candidate: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cand_score = candidate_signal_score(candidate or row)
    metric_score = metric_signal_score(row)
    if cand_score is not None and metric_score is not None:
        score = max(cand_score, 0.62 * cand_score + 0.38 * metric_score)
    elif cand_score is not None:
        score = cand_score
    elif metric_score is not None:
        score = metric_score
    else:
        score = None

    row["scanner_signal_score"] = cand_score
    row["metric_signal_score"] = metric_score
    row["info_edge_score"] = round(score, 1) if score is not None else None
    row["info_edge_tier"] = score_tier(score)
    row["evidence_summary"] = evidence_summary(row)
    return row


def row_matches(row: Dict[str, Any], query: str) -> bool:
    if not query:
        return True
    needle = query.lower()
    return any(needle in str(value).lower() for value in row.values())


def load_user_analysis(data_dir: str, reports_dir: str, query: str = "", limit: int = 200) -> Dict[str, Any]:
    reports = Path(reports_dir)
    metrics_path = reports / "luck_skill" / "metrics_all_users.csv"
    metric_rows = read_csv_rows(metrics_path, limit=10_000)
    if not metric_rows:
        users_dir = reports / "luck_skill" / "users"
        if users_dir.exists():
            for metrics_file in sorted(users_dir.glob("*/metrics.csv")):
                metric_rows.extend(read_csv_rows(metrics_file, limit=1))

    total_count = len(metric_rows)
    scanner_candidates = read_csv_rows(reports / "market_scanner" / "candidate_users.csv", limit=100_000)
    users_root = Path(data_dir) / "user"
    try:
        cached_user_count = sum(1 for path in users_root.iterdir() if path.is_dir()) if users_root.exists() else 0
    except OSError:
        cached_user_count = 0
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
        metrics_mtime = file_mtime_iso(reports / "luck_skill" / "users" / user_key / "metrics.csv")
        return {
            "user_key": user_key,
            "total_trades": safe_int(row.get("total_trades")),
            "luck_index": safe_float(row.get("luck_index")),
            "luck_z": safe_float(row.get("luck_z")),
            "skill_index": safe_float(row.get("skill_index")),
            "skill_z": safe_float(row.get("skill_z")),
            "raw_pnl_usdc": safe_float(row.get("raw_pnl_usdc")),
            "skill_usdc": safe_float(row.get("skill_usdc")),
            "skill_roi": safe_float(row.get("skill_roi")),
            "gross_trade_notional_usdc": safe_float(row.get("gross_trade_notional_usdc")),
            "total_luck_usdc": safe_float(row.get("total_luck_usdc")),
            "resolved_markets": safe_int(row.get("number_of_resolved_markets")),
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
        }

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
        confidence = str(candidate.get("confidence_level") or "").strip().lower()
        row["source_label"] = f"Analyzed + {confidence}" if confidence else "Analyzed + candidate"
        row["candidate_confidence"] = confidence
        row["candidate_timing_z"] = safe_float(candidate.get("timing_z"))
        row["candidate_events"] = safe_int(candidate.get("number_of_independent_events"))

    def compact_candidate(user_key: str, row: Dict[str, Any]) -> Dict[str, Any]:
        freshness = user_watermarks.get(user_key, {}) if isinstance(user_watermarks.get(user_key), dict) else {}
        confidence = str(row.get("confidence_level") or "").strip().lower()
        return {
            "user_key": user_key,
            "total_trades": None,
            "luck_index": None,
            "luck_z": None,
            "skill_index": None,
            "skill_z": None,
            "raw_pnl_usdc": None,
            "skill_usdc": None,
            "skill_roi": None,
            "gross_trade_notional_usdc": None,
            "total_luck_usdc": None,
            "resolved_markets": None,
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
        }

    candidate_only_rows = [
        compact_candidate(user_key, row)
        for user_key, row in candidate_by_user.items()
        if user_key not in metric_user_keys
    ]
    def compacted_candidate_rank(row: Dict[str, Any]) -> Tuple[int, float]:
        ranks = {"very_high": 4, "high": 3, "candidate": 2, "monitor": 1}
        confidence = str(row.get("candidate_confidence") or "").strip().lower()
        return (ranks.get(confidence, 0), safe_float(row.get("candidate_timing_z")) or float("-inf"))

    candidate_only_rows.sort(key=compacted_candidate_rank, reverse=True)

    for row in compacted_metric_rows:
        apply_information_edge_score(row, candidate_by_user.get(str(row.get("user_key") or "")))
    for row in candidate_only_rows:
        apply_information_edge_score(row)

    combined_rows = compacted_metric_rows + candidate_only_rows
    visible_rows = [row for row in combined_rows if row_matches(row, query)]
    visible_metric_rows = [row for row in compacted_metric_rows if row_matches(row, query)]
    visible_candidate_rows = [row for row in candidate_only_rows if row_matches(row, query)]
    top_luck = sort_by_float(visible_metric_rows, "luck_index")[:limit]
    top_skill = sort_by_float(visible_metric_rows, "skill_index")[:limit]
    top_pnl = sort_by_float(visible_metric_rows, "raw_pnl_usdc")[:limit]
    top_info_edge = sort_by_float(visible_rows, "info_edge_score")[:limit]
    rows_for_table = top_info_edge

    return {
        "count": len(visible_rows),
        "total_count": total_count,
        "cached_user_count": cached_user_count,
        "candidate_count": len(candidate_by_user),
        "candidate_only_count": len(candidate_only_rows),
        "shown_candidate_count": len(visible_candidate_rows),
        "high_signal_count": sum(1 for row in combined_rows if (safe_float(row.get("info_edge_score")) or 0.0) >= 70.0),
        "rows": rows_for_table,
        "top_info_edge": top_info_edge,
        "top_luck": top_luck,
        "top_skill": top_skill,
        "top_pnl": top_pnl,
        "metrics_path": str(metrics_path),
    }


def load_scanner(reports_dir: str, query: str = "", limit: int = 200) -> Dict[str, Any]:
    scanner_dir = Path(reports_dir) / "market_scanner"
    candidate_rows = read_csv_rows(scanner_dir / "candidate_users.csv", limit=100_000)
    event_rows = read_csv_rows(scanner_dir / "jump_events.csv", limit=100_000)

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
        "scanner_dir": str(scanner_dir),
    }


def safe_user_key(value: str) -> str:
    text = str(value or "").strip()
    return "".join(ch for ch in text if ch.isalnum() or ch in "._-")


def load_user_profile(data_dir: str, user_key: str) -> Dict[str, Any]:
    profile_path = Path(data_dir) / "user" / user_key / "profile.csv"
    rows = read_csv_rows(profile_path, limit=1)
    return dict(rows[0]) if rows else {}


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


def load_market_statuses(data_dir: str, condition_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    markets_root = Path(data_dir) / "market" / "markets"
    statuses: Dict[str, Dict[str, str]] = {}
    if not markets_root.exists():
        return statuses

    for raw_condition_id in sorted({str(cid or "").strip().lower() for cid in condition_ids if str(cid or "").strip()}):
        market_files = sorted(markets_root.glob(f"*/market_{raw_condition_id}.csv"), reverse=True)
        if not market_files:
            continue
        rows = read_csv_rows(market_files[0], limit=1)
        if not rows:
            continue
        row = rows[0]
        status = market_status_from_row(row)
        status["market_end_date"] = str(row.get("endDate") or "")
        status["market_active"] = str(row.get("active") or "")
        status["market_closed"] = str(row.get("closed") or "")
        status["market_title"] = str(row.get("question") or row.get("title") or row.get("slug") or "")
        status["market_slug"] = str(row.get("slug") or "")
        status["market_winner"] = str(row.get("winner") or "")
        status["market_winning_outcome_index"] = str(row.get("winningOutcomeIndex") or "")
        status["market_outcomes"] = parse_jsonish_list(row.get("outcomes"))
        status["market_outcome_prices"] = parse_jsonish_list(row.get("outcomePrices"))
        statuses[raw_condition_id] = status
    return statuses


def latest_market_cache_file(data_dir: str, condition_id: str) -> Path | None:
    normalized = normalize_condition_id(condition_id)
    if not normalized:
        return None
    markets_root = Path(data_dir) / "market" / "markets"
    if not markets_root.exists():
        return None
    files = list(markets_root.glob(f"*/market_{normalized}.csv"))
    if not files:
        return None
    try:
        return max(files, key=lambda path: path.stat().st_mtime)
    except OSError:
        return files[0]


def file_mtime_iso(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def market_trade_cache_summary(data_dir: str, condition_id: str) -> Dict[str, str]:
    normalized = normalize_condition_id(condition_id)
    if not normalized:
        return {}

    trade_path = Path(data_dir) / "market" / "trades" / normalized / "trades.csv"
    rows = read_csv_rows(trade_path, limit=1)
    if not rows:
        return {}

    row = rows[0]
    return {
        "title": row.get("title") or row.get("market_title") or "",
        "slug": row.get("slug") or "",
        "event_slug": row.get("eventSlug") or row.get("event_slug") or "",
    }


EVENT_LINK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "win",
}


def event_link_tokens(*values: Any) -> set[str]:
    text = " ".join(str(value or "") for value in values).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return {
        token
        for token in tokens
        if len(token) > 1
        and token not in EVENT_LINK_STOP_WORDS
        and (not token.isdigit() or len(token) == 4)
    }


@lru_cache(maxsize=16)
def related_event_slug_candidates(data_dir: str) -> Tuple[Tuple[str, str, str, frozenset[str]], ...]:
    trades_root = Path(data_dir) / "market" / "trades"
    if not trades_root.exists():
        return ()

    candidates: List[Tuple[str, str, str, frozenset[str]]] = []
    for trade_path in trades_root.glob("*/trades.csv"):
        rows = read_csv_rows(trade_path, limit=1)
        if not rows:
            continue
        row = rows[0]
        event_slug = str(row.get("eventSlug") or row.get("event_slug") or "").strip()
        slug = str(row.get("slug") or "").strip()
        title = str(row.get("title") or row.get("market_title") or "").strip()
        if not event_slug or not (slug or title):
            continue
        candidates.append((event_slug, slug, title, frozenset(event_link_tokens(slug, title))))
    return tuple(candidates)


def infer_related_event_slug(data_dir: str, title: str, slug: str) -> str:
    """Recover missing parent event slugs from related cached trade rows."""
    slug_key = str(slug or "").strip().lower()
    target_tokens = event_link_tokens(title, slug)
    if not slug_key and not target_tokens:
        return ""

    best_by_event: Dict[str, Tuple[int, float, float]] = {}
    for event_slug, candidate_slug, _candidate_title, candidate_tokens in related_event_slug_candidates(data_dir):
        if slug_key and slug_key == candidate_slug.lower():
            return event_slug
        if not target_tokens or not candidate_tokens:
            continue
        shared = target_tokens & candidate_tokens
        if len(shared) < 4:
            continue
        union_size = len(target_tokens | candidate_tokens)
        jaccard = len(shared) / union_size if union_size else 0.0
        coverage = min(len(shared) / len(target_tokens), len(shared) / len(candidate_tokens))
        if jaccard < 0.30 or coverage < 0.45:
            continue
        score = (len(shared), jaccard, coverage)
        if score > best_by_event.get(event_slug, (0, 0.0, 0.0)):
            best_by_event[event_slug] = score

    ranked = sorted(best_by_event.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return ""
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return ""
    return ranked[0][0]


def market_summary_from_cache(
    data_dir: str,
    condition_id: str,
    *,
    listed: bool = False,
    added_at: str = "",
    source: str = "cache",
) -> Dict[str, Any]:
    normalized = normalize_condition_id(condition_id)
    cache_path = latest_market_cache_file(data_dir, normalized)
    row = read_csv_rows(cache_path, limit=1)[0] if cache_path else {}
    trade_summary = market_trade_cache_summary(data_dir, normalized)
    status = market_status_from_row(row) if row else {"market_status": "unknown", "market_status_label": "Unknown"}
    title = (
        row.get("question")
        or row.get("title")
        or trade_summary.get("title")
        or row.get("slug")
        or trade_summary.get("slug")
        or normalized
        or str(condition_id or "")
    )
    slug = row.get("slug") or trade_summary.get("slug") or ""
    event_slug = (
        row.get("eventSlug")
        or trade_summary.get("event_slug")
        or infer_related_event_slug(data_dir, str(title or ""), str(slug or ""))
    )
    return {
        "condition_id": normalized or str(condition_id or ""),
        "title": title,
        "slug": slug,
        "event_slug": event_slug,
        "category": row.get("category") or "",
        "volume": safe_float(row.get("volumeNum")) or safe_float(row.get("volume")),
        "liquidity": safe_float(row.get("liquidityNum")) or safe_float(row.get("liquidity")),
        "end_date": row.get("endDate") or row.get("endDateIso") or "",
        "last_updated_at": file_mtime_iso(cache_path),
        "cache_path": str(cache_path) if cache_path else "",
        "cached": bool(cache_path),
        "listed": bool(listed),
        "scanner_signal": False,
        "added_at": added_at,
        "source": source,
        **status,
    }


def recent_cached_markets(data_dir: str, limit: int = 200) -> List[Dict[str, Any]]:
    markets_root = Path(data_dir) / "market" / "markets"
    if not markets_root.exists():
        return []
    files = list(markets_root.rglob("market_*.csv"))
    try:
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        files.sort(reverse=True)
    rows: List[Dict[str, Any]] = []
    for path in files:
        condition_id = path.stem.removeprefix("market_").lower()
        normalized = normalize_condition_id(condition_id)
        if not normalized:
            continue
        rows.append(market_summary_from_cache(data_dir, normalized, source="recent-cache"))
        if len(rows) >= limit:
            break
    return rows


def load_markets(data_dir: str, reports_dir: str, query: str = "", limit: int = 1000) -> Dict[str, Any]:
    market_list = load_market_list(ROOT)
    entries = market_list.get("markets", [])
    listed_ids = {entry.get("condition_id") for entry in entries if entry.get("condition_id")}
    rows_by_id: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        condition_id = str(entry.get("condition_id") or "")
        rows_by_id[condition_id] = market_summary_from_cache(
            data_dir,
            condition_id,
            listed=True,
            added_at=str(entry.get("added_at") or ""),
            source=str(entry.get("source") or "manual"),
        )

    for row in recent_cached_markets(data_dir, limit=limit):
        condition_id = str(row.get("condition_id") or "")
        if condition_id and condition_id not in rows_by_id:
            rows_by_id[condition_id] = row

    scanner = load_scanner(reports_dir, query="", limit=limit)
    scanner_events: List[Dict[str, Any]] = []
    seen_scanner = set()
    for event in scanner.get("events", []):
        condition_id = normalize_condition_id(event.get("condition_id"))
        if not condition_id or condition_id in seen_scanner:
            continue
        seen_scanner.add(condition_id)
        summary = rows_by_id.get(condition_id)
        if summary is None:
            summary = market_summary_from_cache(
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
        if not sources and row.get("source") == "recent-cache":
            sources.append("Recent")
        if not sources and row.get("cached"):
            sources.append("Cached")
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

    watermarks = load_watermarks(data_dir)
    updates = watermarks.get("updates", {}) if isinstance(watermarks.get("updates"), dict) else {}
    analysis = watermarks.get("analysis", {}) if isinstance(watermarks.get("analysis"), dict) else {}
    return {
        "count": len(rows),
        "listed_count": len(rows_by_id),
        "added_count": len(listed_ids),
        "cached_count": sum(1 for row in rows_by_id.values() if row.get("cached")),
        "rows": rows[: max(1, min(limit, 1000))],
        "scanner_events": scanner_events[: max(1, min(limit, 1000))],
        "scanner_event_count": scanner.get("event_count", 0),
        "watermark": updates.get("markets", {}) if isinstance(updates.get("markets"), dict) else {},
        "trade_watermark": (
            updates.get("market_trades", {}) if isinstance(updates.get("market_trades"), dict) else {}
        ),
        "analysis_watermark": (
            analysis.get("market_scanner", {}) if isinstance(analysis.get("market_scanner"), dict) else {}
        ),
        "market_list_path": str(market_list_path(ROOT)),
    }


def listed_market_ids(data_dir: str, reports_dir: str, limit: int = 1000) -> List[str]:
    payload = load_markets(data_dir, reports_dir, limit=limit)
    ids: List[str] = []
    seen = set()
    for row in payload.get("rows", []):
        condition_id = normalize_condition_id(row.get("condition_id"))
        if not condition_id or condition_id in seen:
            continue
        seen.add(condition_id)
        ids.append(condition_id)
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


def load_user_trade_history(
    data_dir: str,
    user_key: str,
    *,
    query: str = "",
    limit: int = 1000,
) -> Dict[str, Any]:
    key = safe_user_key(user_key)
    if not key or key != user_key:
        return {"error": "invalid user key", "user_key": user_key, "total": 0, "rows": []}

    trades_root = Path(data_dir) / "user" / key / "trades"
    if not trades_root.exists():
        return {"error": "user trade cache not found", "user_key": key, "total": 0, "rows": []}

    rows: List[Dict[str, Any]] = []
    total = 0
    for csv_file in sorted(trades_root.rglob("trade_*.csv")):
        for raw in read_csv_rows(csv_file, limit=10_000):
            total += 1
            if query and not row_matches(raw, query):
                continue

            size = safe_float(raw.get("size"))
            price = safe_float(raw.get("price"))
            notional = size * price if size is not None and price is not None else None
            ts = safe_float(raw.get("timestamp"))
            rows.append(
                {
                    "timestamp": ts,
                    "timestamp_utc": to_iso_utc(ts),
                    "side": raw.get("side", ""),
                    "outcome": raw.get("outcome", ""),
                    "outcome_index": safe_int(raw.get("outcomeIndex")),
                    "price": price,
                    "size": size,
                    "notional": notional,
                    "title": raw.get("title") or raw.get("market_title") or raw.get("slug") or "",
                    "slug": raw.get("slug", ""),
                    "event_slug": raw.get("eventSlug", ""),
                    "condition_id": raw.get("conditionId", ""),
                    "asset": raw.get("asset", ""),
                    "transaction_hash": raw.get("transactionHash", ""),
                }
            )

    rows.sort(key=lambda row: safe_float(row.get("timestamp")) or -1.0, reverse=True)
    limited_rows = rows[: max(1, min(limit, 5000))]
    market_statuses = load_market_statuses(
        data_dir,
        {str(row.get("condition_id") or "").strip().lower() for row in rows if row.get("condition_id")},
    )
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


def static_file_payload(path: Path) -> bytes:
    resolved = path.resolve()
    root = ROOT.resolve()
    if not str(resolved).lower().startswith(str(root).lower()):
        raise OSError("refusing path outside workspace")
    return path.read_bytes()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Polymarket Information Edge Dashboard</title>
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
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 20px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 750;
      letter-spacing: 0;
    }
    main {
      max-width: 1600px;
      margin: 0 auto;
      padding: 16px 20px 28px;
      display: grid;
      gap: 16px;
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
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      min-height: 54px;
    }
    .bar h2 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .muted { color: var(--muted); }
    .tabs {
      display: flex;
      gap: 6px;
      padding: 10px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      overflow-x: auto;
    }
    .tab {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 7px 11px;
      font-weight: 650;
      cursor: pointer;
    }
    .tab.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
    }
    .view { display: none; }
    .view.active { display: grid; gap: 16px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 1px;
      background: var(--line);
    }
    .metric {
      background: var(--panel);
      padding: 14px;
      min-height: 82px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .metric strong {
      display: block;
      font-size: 23px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(7, minmax(130px, 1fr));
      gap: 8px;
      padding: 12px;
    }
    button, a.button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      padding: 7px 10px;
      text-decoration: none;
      white-space: normal;
      text-align: center;
    }
    button.primary, a.button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    button:hover, a.button:hover { border-color: var(--accent); }
    button.primary:hover, a.button.primary:hover { background: var(--accent-strong); }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    .tools {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    input[type="search"], input[type="text"] {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      font: inherit;
      min-width: 260px;
      background: #ffffff;
    }
    button.mini {
      min-height: 28px;
      padding: 4px 8px;
      font-size: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fafbfc;
    }
    th.sortable {
      cursor: pointer;
      user-select: none;
    }
    th.sortable::after {
      content: " <> ";
      color: #98a2b3;
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
    tbody tr:hover td { background: #fbfcfe; }
    tr.clickable { cursor: pointer; }
    tr.clickable.selected td {
      background: #edf7f6;
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .status-success, .good { color: var(--good); font-weight: 700; }
    .status-running, .status-queued, .status-cancelled, .warn { color: var(--warn); font-weight: 700; }
    .status-failed, .bad { color: var(--bad); font-weight: 700; }
    .expand {
      width: 30px;
      min-height: 28px;
      padding: 0;
      font-weight: 800;
    }
    .detail-row td {
      background: #fbfcfe;
      padding: 0;
    }
    .nested {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      border-top: 1px solid var(--line);
    }
    .nested th, .nested td {
      padding: 7px 10px;
      font-size: 12px;
    }
    .market-row-open td { background: #f0fdf4; }
    .market-row-closed td { background: #f8fafc; }
    .market-row-ended td { background: #fffbeb; }
    .market-row-unknown td { background: #f9fafb; }
    .market-status-pill {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      margin-right: 7px;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid var(--line);
    }
    .market-status-open {
      background: #dcfce7;
      color: #166534;
      border-color: #86efac;
    }
    .market-status-closed {
      background: #e5e7eb;
      color: #374151;
      border-color: #cbd5e1;
    }
    .market-status-ended {
      background: #fef3c7;
      color: #92400e;
      border-color: #fcd34d;
    }
    .market-status-unknown {
      background: #eef2f6;
      color: #475467;
    }
    .score {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 54px;
      border-radius: 6px;
      padding: 3px 8px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      background: #eef2f6;
      color: var(--ink);
    }
    .score-strong {
      background: #fee4e2;
      color: #912018;
    }
    .score-elevated {
      background: #fffaeb;
      color: #93370d;
    }
    .score-watch {
      background: #eff8ff;
      color: #175cd3;
    }
    .score-baseline {
      background: #f2f4f7;
      color: #475467;
    }
    .evidence {
      line-height: 1.35;
      white-space: normal;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
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
      gap: 16px;
    }
    .workspace-grid {
      display: grid;
      grid-template-columns: minmax(460px, 0.9fr) minmax(0, 1.1fr);
      gap: 16px;
      align-items: start;
    }
    .user-workspace {
      display: grid;
      grid-template-columns: minmax(440px, 0.85fr) minmax(0, 1.15fr);
      gap: 16px;
      align-items: start;
    }
    .table-wrap {
      overflow-x: auto;
    }
    .log {
      margin: 0;
      padding: 14px;
      background: #101828;
      color: #eff4ff;
      min-height: 300px;
      max-height: 520px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
    }
    .empty {
      padding: 22px;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 12px; }
      .metrics, .actions, .grid-2, .workspace-grid, .user-workspace { grid-template-columns: 1fr; }
      input[type="search"] { min-width: 0; width: 100%; }
      input[type="text"] { min-width: 0; width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Polymarket Information Edge</h1>
    <div class="tools">
      <span class="muted" id="generated">Loading</span>
      <button id="refresh">Refresh</button>
      <a class="button" href="/static/user-dashboard" target="_blank">User HTML</a>
      <a class="button" href="/static/market-dashboard" target="_blank">Legacy Market HTML</a>
    </div>
  </header>
  <main>
    <nav class="tabs">
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="markets">Markets</button>
      <button class="tab" data-tab="users">Users</button>
      <button class="tab" data-tab="jobs">Jobs</button>
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
          <div class="metric"><span>User Trade Files</span><strong id="m-user-trades">0</strong></div>
          <div class="metric"><span>Market Files</span><strong id="m-markets">0</strong></div>
          <div class="metric"><span>Price Files</span><strong id="m-prices">0</strong></div>
          <div class="metric"><span>Trade Caches</span><strong id="m-trades">0</strong></div>
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
          <button data-action="build_dashboards">Build HTML</button>
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
          <div class="metric"><span>Cached</span><strong id="mk-cached">0</strong></div>
          <div class="metric"><span>Signals</span><strong id="mk-events">0</strong></div>
          <div class="metric"><span>Market Update</span><strong id="mk-watermark">None</strong></div>
          <div class="metric"><span>Trade Update</span><strong id="mk-trade-watermark">None</strong></div>
          <div class="metric"><span>Analyzed</span><strong id="mk-analysis-watermark">None</strong></div>
          <div class="metric"><span>List File</span><strong id="mk-path">-</strong></div>
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
              <button data-action="update_users_quick">Update All Users</button>
            </div>
          </div>
          <div class="metrics">
            <div class="metric"><span>Analyzed Users</span><strong id="user-total-count">0</strong></div>
            <div class="metric"><span>Shown</span><strong id="user-shown-count">0</strong></div>
            <div class="metric"><span>Cached User Folders</span><strong id="user-cached-count">0</strong></div>
            <div class="metric"><span>Scanner Candidates</span><strong id="user-candidate-count">0</strong></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th class="sortable" data-sort-table="users" data-sort-key="user_key" data-sort-type="text">User</th>
                  <th class="num sortable" data-sort-table="users" data-sort-key="info_edge_score" data-sort-type="number">Edge</th>
                  <th class="sortable" data-sort-table="users" data-sort-key="info_edge_tier" data-sort-type="text">Tier</th>
                  <th class="sortable" data-sort-table="users" data-sort-key="evidence_summary" data-sort-type="text">Evidence</th>
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
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th class="sortable" data-sort-table="trades" data-sort-key="title" data-sort-type="text">Market</th>
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
      <div class="grid-2">
        <section>
          <div class="bar"><h2>Latest Runs</h2></div>
          <table>
            <thead><tr><th class="sortable" data-sort-table="runs" data-sort-key="name" data-sort-type="text">Name</th><th class="sortable" data-sort-table="runs" data-sort-key="status" data-sort-type="text">Status</th><th class="sortable" data-sort-table="runs" data-sort-key="started_at" data-sort-type="text">Started</th><th class="num sortable" data-sort-table="runs" data-sort-key="duration_seconds" data-sort-type="number">Duration</th></tr></thead>
            <tbody id="runs"></tbody>
          </table>
        </section>
        <section>
          <div class="bar"><h2>Jobs</h2></div>
          <table>
            <thead><tr><th class="sortable" data-sort-table="jobs" data-sort-key="label" data-sort-type="text">Job</th><th class="sortable" data-sort-table="jobs" data-sort-key="status" data-sort-type="text">Status</th><th class="sortable" data-sort-table="jobs" data-sort-key="created_at" data-sort-type="text">Created</th><th></th></tr></thead>
            <tbody id="jobs"></tbody>
          </table>
        </section>
      </div>
      <section>
        <div class="bar">
          <h2>Log</h2>
          <span class="muted" id="log-title">No job selected</span>
        </div>
        <pre class="log" id="log"></pre>
      </section>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let selectedJob = null;
    let selectedUser = null;
    let currentTab = "overview";
    let activeJobCount = 0;
    const expandedMarkets = new Set();
    const state = { status: null, users: null, trades: null, markets: null };
    const sortState = {
      overviewMarkets: { key: "signal_score", dir: "desc", type: "number" },
      overviewUsers: { key: "info_edge_score", dir: "desc", type: "number" },
      markets: { key: "signal_score", dir: "desc", type: "number" },
      users: { key: "info_edge_score", dir: "desc", type: "number" },
      runs: { key: "started_at", dir: "desc", type: "text" },
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
    function scoreCell(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "";
      return `<span class="${scoreClass(n)}">${num(n, 1)}</span>`;
    }
    function shortWallet(value) {
      const text = String(value || "");
      return text.length > 18 ? text.slice(0, 10) + "..." + text.slice(-6) : text;
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
      return key ? "https://polymarket.com/profile/" + encodeURIComponent(key) : "";
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
    function userLink(userKey, short = false) {
      return externalLink(polymarketUserUrl(userKey), short ? shortWallet(userKey) : userKey, userKey);
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
      if (["runs", "jobs"].includes(tableName) && state.status) renderStatus(state.status);
      if (tableName === "trades" && state.trades) renderTrades(state.trades);
      updateSortIndicators();
    }
    function setTab(name) {
      currentTab = name;
      document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === name));
      document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === "view-" + name));
    }
    function renderStatus(data) {
      const inv = data.inventory || {};
      $("generated").textContent = "Updated " + (inv.generated_at || data.now || "");
      $("m-users").textContent = intNum(inv.user_count || 0);
      $("m-user-trades").textContent = intNum(inv.user_trade_files || 0);
      $("m-markets").textContent = intNum(inv.market_metadata_files || 0);
      $("m-prices").textContent = intNum(inv.price_history_files || 0);
      $("m-trades").textContent = intNum(inv.market_trade_cache_files || 0);

      const running = (data.jobs || []).filter(j => ["queued", "running"].includes(j.status));
      $("job-state").textContent = running.length ? running.length + " active job" + (running.length === 1 ? "" : "s") : "Idle";
      $("run-note").textContent = running.length ? "Background work is running" : "";

      $("runs").innerHTML = sortedRows(data.runs || [], "runs").map(run => {
        const dur = typeof run.duration_seconds === "number" ? num(run.duration_seconds, 1) + "s" : "";
        return `<tr><td>${esc(run.name)}</td><td class="${cls(run.status)}">${esc(run.status)}</td><td>${esc(run.started_at)}</td><td class="num">${dur}</td></tr>`;
      }).join("") || emptyRow(4, "No runs recorded");

      $("jobs").innerHTML = sortedRows(data.jobs || [], "jobs").map(job => {
        const cancellable = ["queued", "running"].includes(job.status);
        const cancel = cancellable ? ` <button data-cancel-job="${esc(job.id)}">Cancel</button>` : "";
        return `<tr><td>${esc(job.label)}</td><td class="${cls(job.status)}">${esc(job.status)}</td><td>${esc(job.created_at)}</td><td><button data-job="${esc(job.id)}">Log</button>${cancel}</td></tr>`;
      }).join("") || emptyRow(4, "No jobs recorded");

      document.querySelectorAll("[data-job]").forEach(button => {
        button.onclick = () => selectJob(button.getAttribute("data-job"));
      });
      document.querySelectorAll("[data-cancel-job]").forEach(button => {
        button.onclick = async () => {
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
      $("users-count").textContent = intNum(data.count || 0) + " users";
      $("user-total-count").textContent = intNum(data.total_count || 0);
      $("user-shown-count").textContent = intNum(data.count || 0);
      $("user-cached-count").textContent = intNum(data.cached_user_count || 0);
      $("user-candidate-count").textContent = intNum(data.candidate_count || 0);
      $("signal-high-users").textContent = intNum(data.high_signal_count || 0);
      $("signal-candidates").textContent = intNum(data.candidate_count || 0);
      $("signal-analyzed").textContent = intNum(data.total_count || 0);
      const overviewRows = sortedRows(data.top_info_edge || data.rows || [], "overviewUsers");
      $("overview-users").innerHTML = overviewRows.slice(0, 8).map(row => {
        const selected = selectedUser === row.user_key ? " selected" : "";
        return `<tr class="clickable${selected}" data-user-key="${esc(row.user_key)}">
          <td class="num">${scoreCell(row.info_edge_score)}</td>
          <td title="${esc(row.user_key)}">${userLink(row.user_key, true)}</td>
          <td><span class="pill">${esc(row.info_edge_tier || "Unscored")}</span></td>
          <td class="evidence" title="${esc(row.evidence_summary)}">${esc(row.evidence_summary)}</td>
        </tr>`;
      }).join("") || emptyRow(4, "No user signals yet");
      $("users-table").innerHTML = sortedRows(data.rows || data.top_info_edge || [], "users").map(row => {
        const selected = selectedUser === row.user_key ? " selected" : "";
        const scanner = row.candidate_confidence ? `<div class="muted">Scanner: ${esc(row.candidate_confidence)} | z ${num(row.candidate_timing_z, 2)} | ${intNum(row.candidate_events)} events</div>` : "";
        const evidence = `<div class="evidence" title="${esc(row.evidence_summary)}">${esc(row.evidence_summary || "")}</div>${scanner}<div class="muted">${esc(row.source_label || "User")}</div>`;
        return `<tr class="clickable${selected}" data-user-key="${esc(row.user_key)}">
          <td title="${esc(row.user_key)}">${userLink(row.user_key, true)}</td>
          <td class="num">${scoreCell(row.info_edge_score)}</td>
          <td><span class="pill">${esc(row.info_edge_tier || "Unscored")}</span></td>
          <td>${evidence}</td>
          <td class="num">${num(row.skill_index)}</td>
          <td class="num">${num(row.raw_pnl_usdc)}</td>
          <td class="num">${intNum(row.total_trades)}</td>
          <td title="${esc(row.last_updated_at)}">${esc(shortDate(row.last_updated_at))}</td>
          <td><button class="mini" data-update-user="${esc(row.user_key)}">Update</button></td>
        </tr>`;
      }).join("") || emptyRow(9, "No users yet");
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
      updateSortIndicators();
    }
    function renderTrades(data) {
      const groups = sortedRows(data && data.groups ? data.groups : [], "trades");
      const userKey = data && data.user_key ? data.user_key : selectedUser;
      $("trade-title").textContent = userKey ? "Trading History: " + userKey : "Trading History";
      $("update-selected-user").disabled = !userKey;
      if (!userKey) {
        $("trade-summary").textContent = "Select a user";
      } else if (data && data.error) {
        $("trade-summary").textContent = data.error;
      } else if (data) {
        $("trade-summary").textContent = intNum(data.market_count || 0) + " markets, " + intNum(data.matched || 0) + " trades shown of " + intNum(data.total || 0) + " cached trades";
      }
      $("trades-table").innerHTML = groups.map(group => {
        const marketKey = group.market_key || group.condition_id || group.title;
        const expanded = expandedMarkets.has(marketKey);
        const buttonLabel = expanded ? "-" : "+";
        const detail = expanded ? renderTradeDetail(group.trades || []) : "";
        const status = group.market_status || "unknown";
        const statusLabel = group.market_status_label || "Unknown";
        return `<tr class="market-row-${esc(status)}">
          <td><button class="expand" data-market-key="${esc(marketKey)}">${buttonLabel}</button></td>
          <td title="${esc(group.title)}">${marketLink(group)}</td>
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
      }).join("") || emptyRow(12, selectedUser ? "No cached trades found" : "Select a user to load trades");
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
      return `<tr class="detail-row"><td colspan="12">
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
    function marketActionButtons(row) {
      const id = row.condition_id || "";
      const updateTitle = "Refresh this market's cached metadata and price history";
      if (!id) return "";
      const update = `<button class="mini" data-update-market="${esc(id)}" title="${esc(updateTitle)}">Update</button>`;
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
      $("market-count").textContent = intNum(rows.length || 0) + " markets";
      $("mk-listed").textContent = intNum(rows.length || 0);
      $("mk-cached").textContent = intNum(data.cached_count || 0);
      $("mk-events").textContent = intNum(data.scanner_event_count || 0);
      $("mk-watermark").textContent = shortDate((data.watermark || {}).updated_at) || "None";
      $("mk-trade-watermark").textContent = shortDate((data.trade_watermark || {}).updated_at) || "None";
      $("mk-analysis-watermark").textContent = shortDate((data.analysis_watermark || {}).updated_at) || "None";
      $("signal-events").textContent = intNum(data.scanner_event_count || 0);
      $("signal-last-analysis").textContent = shortDate((data.analysis_watermark || {}).updated_at) || "None";
      $("mk-path").textContent = data.market_list_path ? data.market_list_path.split(/[\\\\/]/).pop() : "-";
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
    async function refreshStatus(force = false) {
      const previousActiveJobs = activeJobCount;
      state.status = await api("/api/status" + (force ? "?refresh=1" : ""));
      activeJobCount = (state.status.jobs || []).filter(j => ["queued", "running"].includes(j.status)).length;
      renderStatus(state.status);
      if (selectedJob) refreshLog();
      if (previousActiveJobs > 0 && activeJobCount === 0) {
        await Promise.all([refreshUsers(), refreshMarkets()]);
        if (selectedUser) await refreshTrades();
      }
    }
    async function refreshUsers() {
      const query = $("user-search").value.trim();
      state.users = await api("/api/users?q=" + encodeURIComponent(query));
      renderUsers(state.users);
    }
    async function refreshMarkets() {
      const query = $("market-search") ? $("market-search").value.trim() : "";
      state.markets = await api("/api/markets?q=" + encodeURIComponent(query));
      renderMarkets(state.markets);
    }
    async function addMarket(marketId, source = "manual") {
      const id = String(marketId || "").trim();
      if (!id) return;
      state.markets = await api("/api/markets", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ market_id: id, source })
      });
      renderMarkets(state.markets);
    }
    async function selectUser(userKey) {
      if (!userKey) return;
      selectedUser = userKey;
      expandedMarkets.clear();
      setTab("users");
      renderUsers(state.users || {});
      $("trade-title").textContent = "Trading History: " + userKey;
      $("trade-summary").textContent = "Loading cached trades";
      $("trades-table").innerHTML = emptyRow(12, "Loading cached trades");
      await refreshTrades();
      $("update-selected-user").disabled = false;
    }
    async function refreshTrades() {
      if (!selectedUser) {
        renderTrades(null);
        return;
      }
      const query = $("trade-search").value.trim();
      state.trades = await api("/api/users/" + encodeURIComponent(selectedUser) + "/trades?limit=1000&q=" + encodeURIComponent(query));
      renderTrades(state.trades);
    }
    async function refreshAll(force = false) {
      await Promise.all([refreshStatus(force), refreshUsers(), refreshMarkets()]);
    }
    async function selectJob(id) {
      selectedJob = id;
      $("log-title").textContent = id;
      setTab("jobs");
      await refreshLog();
    }
    async function refreshLog() {
      if (!selectedJob) return;
      const data = await api("/api/jobs/" + encodeURIComponent(selectedJob) + "/log");
      $("log").textContent = data.log || "";
      $("log").scrollTop = $("log").scrollHeight;
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
    $("refresh").onclick = () => refreshAll(true);
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
    $("trade-search").oninput = () => refreshTrades();
    $("market-search").oninput = () => refreshMarkets();
    $("update-selected-user").onclick = async () => {
      if (!selectedUser) return;
      await startJob("update_user", { user_key: selectedUser });
    };
    renderTrades(null);
    refreshAll(false);
    setInterval(() => refreshStatus(false), 5000);
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


def _bytes_response(handler: BaseHTTPRequestHandler, body: bytes, content_type: str) -> None:
    _send_response(handler, body, content_type)


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
        user_key = safe_user_key(raw_user_key)
        if not user_key or user_key != raw_user_key:
            raise ValueError("user_key is required")
        return (
            f"Update and analyze user {user_key}",
            app_command("update", "users", "--user", user_key, "--skip-price-history", "--analyze-after"),
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
                "jobs": list_jobs(ROOT, limit=12),
                "watermarks": load_watermarks(self.data_dir),
            }
            _json_response(self, payload)
            return
        if parsed.path == "/api/users":
            query = str((params.get("q") or [""])[0])
            _json_response(self, load_user_analysis(self.data_dir, self.reports_dir, query=query))
            return
        if parsed.path.startswith("/api/users/") and parsed.path.endswith("/trades"):
            raw_user = parsed.path[len("/api/users/") : -len("/trades")]
            user_key = unquote(raw_user).strip("/")
            query = str((params.get("q") or [""])[0])
            limit = safe_int((params.get("limit") or ["1000"])[0]) or 1000
            payload = load_user_trade_history(
                self.data_dir,
                user_key,
                query=query,
                limit=limit,
            )
            status = 400 if payload.get("error") == "invalid user key" else 200
            _json_response(self, payload, status=status)
            return
        if parsed.path == "/api/scanner":
            query = str((params.get("q") or [""])[0])
            _json_response(self, load_scanner(self.reports_dir, query=query))
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
        if parsed.path == "/static/user-dashboard":
            path = Path(self.reports_dir) / "dashboard.html"
            if path.exists():
                _bytes_response(self, static_file_payload(path), "text/html; charset=utf-8")
                return
            _text_response(self, "<p>User dashboard has not been built.</p>")
            return
        if parsed.path == "/static/market-dashboard":
            path = Path(self.reports_dir) / "market_scanner_dashboard.html"
            if path.exists():
                _bytes_response(self, static_file_payload(path), "text/html; charset=utf-8")
                return
            _text_response(self, "<p>Scanner dashboard has not been built.</p>")
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
