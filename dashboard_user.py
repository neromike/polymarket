#!/usr/bin/env python3
"""Interactive dashboard for Polymarket luck/skill analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import parse_jsonish_list, safe_float, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the user luck/skill HTML dashboard.")
    parser.add_argument("--data-dir", default="data", help="Data directory root (default: data)")
    parser.add_argument(
        "--report-users-dir",
        default="reports/luck_skill/users",
        help="Directory containing per-user metrics (default: reports/luck_skill/users)",
    )
    parser.add_argument("--out", default="reports/dashboard.html", help="Output HTML path")
    return parser.parse_args()


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _first_nonempty(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_all_user_metrics(report_users_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-user analysis metrics from reports/luck_skill/users."""
    metrics_by_user: Dict[str, Dict[str, Any]] = {}
    if not report_users_dir.exists():
        return metrics_by_user

    for user_dir in sorted(report_users_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        metrics_file = user_dir / "metrics.csv"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_key = str(row.get("user_key") or user_dir.name)
                    metrics_by_user[user_key] = dict(row)
        except OSError as exc:
            print(f"Failed to read {metrics_file}: {exc}", file=sys.stderr)

    return metrics_by_user


def load_user_profile(data_dir: Path, user_key: str) -> Dict[str, Any]:
    profile_file = data_dir / "user" / user_key / "profile.csv"
    if not profile_file.exists():
        return {}

    try:
        with open(profile_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return dict(row)
    except OSError as exc:
        print(f"Failed to read {profile_file}: {exc}", file=sys.stderr)
    return {}


def load_user_trades(data_dir: Path, user_key: str) -> List[Dict[str, Any]]:
    trades_root = data_dir / "user" / user_key / "trades"
    if not trades_root.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for month_dir in sorted(trades_root.iterdir()):
        if not month_dir.is_dir():
            continue
        for csv_file in sorted(month_dir.glob("trade_*.csv")):
            try:
                with open(csv_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(dict(row))
            except OSError as exc:
                print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)
    return rows


def load_markets(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    markets_root = data_dir / "market" / "markets"
    result: Dict[str, Dict[str, Any]] = {}
    if not markets_root.exists():
        return result

    for month_dir in sorted(markets_root.iterdir()):
        if not month_dir.is_dir():
            continue
        for csv_file in sorted(month_dir.glob("market_*.csv")):
            try:
                with open(csv_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        condition_id = str(row.get("conditionId") or "")
                        if not condition_id:
                            continue
                        result[condition_id] = {
                            "conditionId": condition_id,
                            "question": str(row.get("question") or ""),
                            "slug": str(row.get("slug") or ""),
                            "eventSlug": str(row.get("eventSlug") or ""),
                            "closed": _is_truthy(row.get("closed")),
                            "umaResolutionStatus": str(row.get("umaResolutionStatus") or ""),
                          "endDate": _first_nonempty(row, "endDate", "end_date", "endDateIso", "endTime"),
                          "closedTime": _first_nonempty(row, "closedTime", "closed_time", "closedAt", "closeTime"),
                            "outcomes": parse_jsonish_list(row.get("outcomes")),
                            "outcomePrices": parse_jsonish_list(row.get("outcomePrices")),
                            "clobTokenIds": parse_jsonish_list(row.get("clobTokenIds")),
                        }
            except OSError as exc:
                print(f"Failed to read {csv_file}: {exc}", file=sys.stderr)

    return result


def _market_outcome_for_asset(market: Dict[str, Any], asset: str) -> Tuple[str, float | None]:
    token_ids = [str(x) for x in (market.get("clobTokenIds") or [])]
    outcomes = market.get("outcomes") or []
    prices = market.get("outcomePrices") or []
    if asset in token_ids:
        idx = token_ids.index(asset)
        label = str(outcomes[idx]) if idx < len(outcomes) else f"Outcome {idx}"
        price = safe_float(prices[idx]) if idx < len(prices) else None
        return label, price
    return "", None


def summarize_user_markets(
    trades: List[Dict[str, Any]],
    markets_by_condition: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    open_trade_count = 0
    closed_trade_count = 0

    for trade in trades:
        condition_id = str(trade.get("conditionId") or "")
        asset = str(trade.get("asset") or "")
        if not condition_id or not asset:
            continue

        market_for_trade = markets_by_condition.get(condition_id, {})
        is_closed_trade = bool(market_for_trade.get("closed")) or str(market_for_trade.get("umaResolutionStatus") or "").lower() == "resolved"
        if is_closed_trade:
            closed_trade_count += 1
        else:
            open_trade_count += 1

        side = str(trade.get("side") or "").upper()
        size = safe_float(trade.get("size"))
        price = safe_float(trade.get("price"))
        if side not in {"BUY", "SELL"} or size is None or price is None or size <= 0:
            continue

        key = (condition_id, asset)
        row = grouped.get(key)
        if row is None:
            market = markets_by_condition.get(condition_id, {})
            outcome_name, reference_price = _market_outcome_for_asset(market, asset)
            if not outcome_name:
                outcome_name = str(trade.get("outcome") or "") or "Unknown"
            row = {
                "condition_id": condition_id,
                "asset": asset,
                "market_title": str(market.get("question") or "Unknown market"),
                "market_slug": str(market.get("slug") or ""),
                "event_slug": str(market.get("eventSlug") or ""),
                "market_closed": bool(market.get("closed")) or str(market.get("umaResolutionStatus") or "").lower() == "resolved",
                "outcome": outcome_name,
                "reference_price": reference_price,
              "expected_close": str(market.get("endDate") or market.get("closedTime") or ""),
                "buy_shares": 0.0,
                "sell_shares": 0.0,
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "trade_count": 0,
              "trades": [],
            }
            grouped[key] = row

        notional = size * price
        row["trade_count"] += 1
        row["trades"].append(
          {
            "timestamp": trade.get("timestamp"),
            "side": side,
            "size": size,
            "price": price,
          }
        )
        if side == "BUY":
            row["buy_shares"] += size
            row["buy_notional"] += notional
        else:
            row["sell_shares"] += size
            row["sell_notional"] += notional

    open_rows: List[Dict[str, Any]] = []
    closed_rows: List[Dict[str, Any]] = []
    open_net_value = 0.0
    open_est_pnl = 0.0
    closed_pnl = 0.0
    closed_notional = 0.0
    closed_market_pnl: Dict[str, float] = {}

    for row in grouped.values():
        net_shares = row["buy_shares"] - row["sell_shares"]
        reference_price = row["reference_price"]
        position_value = net_shares * reference_price if reference_price is not None else 0.0
        pnl = row["sell_notional"] - row["buy_notional"] + position_value

        summary_row = {
            "market_title": row["market_title"],
            "market_slug": row["market_slug"],
            "event_slug": row["event_slug"],
            "outcome": row["outcome"],
            "condition_id": row["condition_id"],
            "trade_count": row["trade_count"],
            "net_shares": net_shares,
            "buy_notional": row["buy_notional"],
            "sell_notional": row["sell_notional"],
            "reference_price": reference_price,
            "expected_close": row["expected_close"],
            "position_value": position_value,
            "pnl": pnl,
            "trades": row["trades"],
        }

        if row["market_closed"]:
            closed_rows.append(summary_row)
            closed_pnl += pnl
            closed_notional += row["buy_notional"] + row["sell_notional"]
            cond = row["condition_id"]
            closed_market_pnl[cond] = closed_market_pnl.get(cond, 0.0) + pnl
        elif abs(net_shares) > 1e-12:
            open_rows.append(summary_row)
            open_net_value += position_value
            open_est_pnl += pnl

    open_rows.sort(key=lambda x: abs(x["position_value"]), reverse=True)
    closed_rows.sort(key=lambda x: abs(x["pnl"]), reverse=True)

    closed_win_count = sum(1 for v in closed_market_pnl.values() if v > 0)
    closed_loss_count = sum(1 for v in closed_market_pnl.values() if v < 0)
    closed_win_usdc = sum(v for v in closed_market_pnl.values() if v > 0)
    closed_loss_usdc = -sum(v for v in closed_market_pnl.values() if v < 0)

    return {
        "open": {
            "market_count": len({r["condition_id"] for r in open_rows}),
            "position_count": len(open_rows),
            "trade_count": open_trade_count,
            "net_value": open_net_value,
            "est_pnl": open_est_pnl,
            "rows": open_rows[:100],
        },
        "closed": {
            "market_count": len({r["condition_id"] for r in closed_rows}),
            "position_count": len(closed_rows),
            "trade_count": closed_trade_count,
            "pnl": closed_pnl,
            "notional": closed_notional,
            "win_count": closed_win_count,
            "win_usdc": closed_win_usdc,
            "loss_count": closed_loss_count,
            "loss_usdc": closed_loss_usdc,
            "rows": closed_rows[:100],
        },
    }


def build_user_payload(
    user_key: str,
    metrics: Dict[str, Any],
    data_dir: Path,
    markets_by_condition: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    profile = load_user_profile(data_dir, user_key)
    trades = load_user_trades(data_dir, user_key)
    market_summary = summarize_user_markets(trades, markets_by_condition)

    display_name = (
        str(profile.get("profile_name") or "").strip()
        or str(profile.get("profile_pseudonym") or "").strip()
        or user_key
    )

    total_trades = safe_int(metrics.get("total_trades")) or 0
    resolved_markets = safe_int(metrics.get("number_of_resolved_markets")) or 0
    luck_index = safe_float(metrics.get("luck_index"))
    luck_z = safe_float(metrics.get("luck_z"))
    skill_index = safe_float(metrics.get("skill_index"))
    skill_z = safe_float(metrics.get("skill_z"))
    skill_randomized_sd_usdc = safe_float(metrics.get("skill_randomized_sd_usdc")) or 0.0
    total_luck = safe_float(metrics.get("total_luck_usdc")) or 0.0
    skill_usdc = safe_float(metrics.get("skill_usdc")) or 0.0
    skill_roi = safe_float(metrics.get("skill_roi")) or 0.0
    raw_pnl = safe_float(metrics.get("raw_pnl_usdc")) or 0.0
    notional = safe_float(metrics.get("gross_trade_notional_usdc")) or 0.0
    warning_count = safe_int(metrics.get("warning_count")) or 0

    luck_pct = (total_luck / raw_pnl * 100) if raw_pnl != 0 else 0.0
    skill_pct = (skill_usdc / raw_pnl * 100) if raw_pnl != 0 else 0.0

    return {
        "user_key": user_key,
        "display_name": display_name,
        "profile": {
            "input_user": str(profile.get("input_user") or user_key),
            "profile_name": str(profile.get("profile_name") or ""),
            "profile_pseudonym": str(profile.get("profile_pseudonym") or ""),
            "wallet": str(profile.get("wallet") or ""),
        },
        "metrics": {
            "total_trades": total_trades,
            "resolved_markets": resolved_markets,
            "luck_index": luck_index,
            "luck_z": luck_z,
            "skill_index": skill_index,
            "skill_z": skill_z,
            "skill_randomized_sd_usdc": skill_randomized_sd_usdc,
            "total_luck": total_luck,
            "luck_pct": luck_pct,
            "skill_usdc": skill_usdc,
            "skill_pct": skill_pct,
            "skill_roi": skill_roi,
            "raw_pnl": raw_pnl,
            "notional": notional,
            "warning_count": warning_count,
            "downloaded_trades": len(trades),
        },
        "open_summary": market_summary["open"],
        "closed_summary": market_summary["closed"],
    }


def build_dashboard_payload(users: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "users": users,
    }


def build_dashboard_html(payload: Dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True)
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Polymarket Analysis Dashboard</title>
    <style>
      :root {
        --bg0: #f7f2e9;
        --bg1: #e8efe6;
        --panel: #ffffff;
        --ink: #16223b;
        --muted: #5b6478;
        --accent: #c95d2d;
        --accent-2: #1f8a70;
        --line: #d8dbe5;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: \"Segoe UI\", \"Trebuchet MS\", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at 12% 8%, rgba(201, 93, 45, 0.15), transparent 28%),
          radial-gradient(circle at 88% 0%, rgba(31, 138, 112, 0.12), transparent 32%),
          linear-gradient(160deg, var(--bg0), var(--bg1));
      }
      .layout {
        min-height: 100vh;
        display: grid;
        grid-template-columns: 320px 1fr;
        gap: 12px;
        padding: 12px;
      }
      .panel {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(22,34,59,0.12);
        border-radius: 14px;
        overflow: hidden;
      }
      .panel-head {
        padding: 14px;
        border-bottom: 1px solid var(--line);
      }
      .panel-head h1, .panel-head h2 { margin: 0; font-size: 1.1rem; }
      .panel-head a { color: inherit; text-decoration: none; }
      .panel-head a:hover { text-decoration: underline; text-decoration-color: var(--accent); }
      .sub { margin-top: 4px; color: var(--muted); font-size: 0.83rem; }
      .list-controls {
        display: flex;
        gap: 8px;
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        flex-wrap: wrap;
      }
      .list-btn {
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.78rem;
        cursor: pointer;
      }
      .list-btn.active {
        background: rgba(201,93,45,0.12);
        border-color: var(--accent);
      }
      .user-list { list-style: none; margin: 0; padding: 0; max-height: calc(100vh - 120px); overflow: auto; }
      .user-list li {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 8px;
        padding: 11px 12px;
        cursor: pointer;
        border-left: 3px solid transparent;
        border-bottom: 1px solid rgba(216,219,229,0.5);
      }
      .user-list li:hover { background: rgba(22,34,59,0.05); }
      .user-list li.active { border-left-color: var(--accent); background: rgba(201,93,45,0.12); }
      .uname { font-weight: 700; }
      .umetrics { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }
      .detail { padding: 14px; display: grid; gap: 12px; }
      .cards { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 8px; }
      .card {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 9px;
        background: #fff;
      }
      .k { font-size: 0.74rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
      .v { font-size: 1rem; font-weight: 800; margin-top: 4px; }
      .profile {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fff;
        padding: 10px;
        display: grid;
        grid-template-columns: repeat(2, minmax(220px, 1fr));
        gap: 8px;
      }
      .line { font-size: 0.86rem; color: var(--muted); }
      .line b { color: var(--ink); font-weight: 700; }
      .label-chip {
        display: inline-block;
        margin-top: 6px;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        border: 1px solid transparent;
      }
      .split { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
      .box {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fff;
      }
      .box h3 {
        margin: 0;
        padding: 10px;
        border-bottom: 1px solid var(--line);
        font-size: 0.96rem;
      }
      .box .meta { padding: 8px 10px; color: var(--muted); font-size: 0.82rem; display: flex; gap: 12px; flex-wrap: wrap; }
      .table-wrap { max-height: 340px; overflow: auto; border-top: 1px solid var(--line); }
      table { width: 100%; border-collapse: collapse; }
      th, td { text-align: left; padding: 7px 8px; font-size: 0.81rem; border-bottom: 1px solid rgba(216,219,229,0.6); }
      th { position: sticky; top: 0; background: #f8fafc; color: var(--muted); }
      td a { color: var(--ink); text-decoration-color: rgba(22,34,59,0.25); }
      td a:hover { text-decoration-color: var(--accent); }
      .trades-toggle {
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
        border-radius: 8px;
        padding: 2px 8px;
        cursor: pointer;
        font-size: 0.76rem;
      }
      .trades-toggle:hover { border-color: var(--accent); }
      .open-detail-row td { background: #fcfcfd; }
      .trade-subtable {
        width: 100%;
        border-collapse: collapse;
        margin-top: 2px;
      }
      .trade-subtable th, .trade-subtable td {
        text-align: left;
        padding: 5px 7px;
        font-size: 0.78rem;
        border-bottom: 1px solid rgba(216,219,229,0.5);
      }
      .trade-subtable th { color: var(--muted); background: #f8fafc; }
      .closed-win td { background: rgba(31, 138, 112, 0.11); }
      .closed-loss td { background: rgba(180, 35, 24, 0.10); }
      .pos { color: #1f8a70; }
      .neg { color: #b42318; }
      @media (max-width: 1120px) {
        .layout { grid-template-columns: 1fr; }
        .cards { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
        .profile { grid-template-columns: 1fr; }
        .split { grid-template-columns: 1fr; }
      }
    </style>
</head>
<body>
    <div class=\"layout\">
      <aside class=\"panel\">
        <div class=\"panel-head\">
          <h1>Users With Analysis Data</h1>
          <div class=\"sub\" id=\"generated\"></div>
        </div>
        <div class=\"list-controls\">
          <button class=\"list-btn\" id=\"sort-luck\" type=\"button\">Luck</button>
          <button class=\"list-btn\" id=\"sort-skill\" type=\"button\">Skill</button>
          <button class=\"list-btn\" id=\"sort-pnl\" type=\"button\">PnL</button>
          <button class=\"list-btn\" id=\"filter-open\" type=\"button\">All Users</button>
        </div>
        <ul class=\"user-list\" id=\"users\"></ul>
      </aside>
      <main class=\"panel\">
        <div class=\"panel-head\">
          <h2 id=\"title\">No user selected</h2>
          <div class=\"sub\" id=\"subtitle\"></div>
        </div>
        <div class=\"detail\">
          <section class=\"profile\" id=\"profile\"></section>
          <section class=\"cards\" id=\"cards\"></section>
          <section class=\"split\">
            <div class=\"box\">
              <h3>Open Markets Summary</h3>
              <div class=\"meta\" id=\"open-meta\"></div>
              <div class=\"table-wrap\">
                <table>
                  <thead>
                    <tr><th>Market</th><th>Net Shares</th><th>Current Px</th><th>Expected Close</th><th>Position Value</th><th>Est PnL</th><th>Trades</th></tr>
                  </thead>
                  <tbody id=\"open-rows\"></tbody>
                </table>
              </div>
            </div>
            <div class=\"box\">
              <h3>Closed Markets Summary</h3>
              <div class=\"meta\" id=\"closed-meta\"></div>
              <div class=\"meta\" id=\"closed-meta-wl\"></div>
              <div class=\"table-wrap\">
                <table>
                  <thead>
                    <tr><th>Market</th><th>Net Shares</th><th>Settle Px</th><th>PnL</th><th>Notional</th><th>Trades</th></tr>
                  </thead>
                  <tbody id=\"closed-rows\"></tbody>
                </table>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
    <script id=\"dashboard-data\" type=\"application/json\">__PAYLOAD__</script>
    <script>
      var dashboardData = JSON.parse(document.getElementById("dashboard-data").textContent || "{}");
      var users = Array.isArray(dashboardData.users) ? dashboardData.users : [];
      var selected = users[0] || null;
      var sortMode = "luck";
      var openOnly = false;
      var el = {
        generated: document.getElementById("generated"),
        sortLuck: document.getElementById("sort-luck"),
        sortSkill: document.getElementById("sort-skill"),
        sortPnl: document.getElementById("sort-pnl"),
        filterOpen: document.getElementById("filter-open"),
        users: document.getElementById("users"),
        title: document.getElementById("title"),
        subtitle: document.getElementById("subtitle"),
        profile: document.getElementById("profile"),
        cards: document.getElementById("cards"),
        openMeta: document.getElementById("open-meta"),
        closedMeta: document.getElementById("closed-meta"),
        closedMetaWl: document.getElementById("closed-meta-wl"),
        openRows: document.getElementById("open-rows"),
        closedRows: document.getElementById("closed-rows"),
      };

      function esc(s) {
        return String(s === null || s === undefined ? "" : s)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/\"/g, "&quot;");
      }
      function n(v) {
        var x = Number(v);
        return Number.isFinite(x) ? x : null;
      }
      function fmt(v, mode) {
        var x = n(v);
        if (x === null) return "NA";
        if (mode === "usd") return "$" + x.toLocaleString(undefined, { maximumFractionDigits: 2 });
        if (mode === "pct") return (x * 100).toFixed(2) + "%";
        return x.toLocaleString(undefined, { maximumFractionDigits: 4 });
      }
      function cls(v) {
        var x = n(v);
        if (x === null || x === 0) return "";
        return x > 0 ? "pos" : "neg";
      }

      function signed2(v) {
        var x = n(v);
        if (x === null) return "NA";
        return (x >= 0 ? "+" : "") + x.toFixed(2);
      }

      function luckSdLabel(v) {
        var x = n(v);
        if (x === null) return "No SD available";
        if (x >= 2.0) return "Very lucky";
        if (x >= 1.0) return "One standard deviation lucky";
        if (x <= -2.0) return "Very unlucky";
        if (x <= -1.0) return "One standard deviation unlucky";
        if (Math.abs(x) < 0.05) return "Exactly average luck";
        return "Near average luck";
      }

      function skillSdLabel(v) {
        var x = n(v);
        if (x === null) return "No SD available";
        if (x >= 2.0) return "Very skillful timing/price selection";
        if (x >= 1.0) return "Above-average timing/price selection";
        if (x <= -2.0) return "Very weak timing/price selection";
        if (x <= -1.0) return "Below-average timing/price selection";
        if (Math.abs(x) < 0.05) return "Exactly average timing/price selection";
        return "Near average timing/price selection";
      }

      function sdTone(v) {
        var x = n(v);
        if (x === null) {
          return { bg: "#f2f4f7", fg: "#344054", border: "#d0d5dd" };
        }
        var strength = Math.min(1, Math.abs(x) / 2.5);
        if (x >= 0) {
          return {
            bg: strength >= 0.66 ? "#d1fadf" : strength >= 0.33 ? "#ecfdf3" : "#f6fef9",
            fg: "#027a48",
            border: strength >= 0.66 ? "#12b76a" : "#6ce9a6",
          };
        }
        return {
          bg: strength >= 0.66 ? "#fee4e2" : strength >= 0.33 ? "#fef3f2" : "#fff7f6",
          fg: "#b42318",
          border: strength >= 0.66 ? "#f04438" : "#fda29b",
        };
      }

      function labelChip(label, z) {
        var tone = sdTone(z);
        return '<div class="label-chip" style="background:' + tone.bg + ';color:' + tone.fg + ';border-color:' + tone.border + ';">' + esc(label) + '</div>';
      }

      function marketHref(row) {
        var eventSlug = (row && row.event_slug) ? String(row.event_slug) : "";
        var marketSlug = (row && row.market_slug) ? String(row.market_slug) : "";
        var conditionId = (row && row.condition_id) ? String(row.condition_id) : "";
        if (eventSlug && marketSlug && eventSlug !== marketSlug) {
          return "https://polymarket.com/event/" + encodeURIComponent(eventSlug) + "/" + encodeURIComponent(marketSlug);
        }
        var fallbackSlug = eventSlug || marketSlug;
        if (fallbackSlug) {
          return "https://polymarket.com/event/" + encodeURIComponent(fallbackSlug);
        }
        if (conditionId) {
          return "https://polymarket.com/market/" + encodeURIComponent(conditionId);
        }
        return "";
      }

      function visibleUsers() {
        var list = users.slice();
        if (openOnly) {
          list = list.filter(function (u) {
            var open = u.open_summary || {};
            return (n(open.position_count) || 0) > 0;
          });
        }
        list.sort(function (a, b) {
          var am = a.metrics || {};
          var bm = b.metrics || {};
          var av = sortMode === "pnl"
            ? (n(am.raw_pnl) || 0)
            : sortMode === "skill"
              ? (n(am.skill_index) || 0)
              : (n(am.luck_index) || 0);
          var bv = sortMode === "pnl"
            ? (n(bm.raw_pnl) || 0)
            : sortMode === "skill"
              ? (n(bm.skill_index) || 0)
              : (n(bm.luck_index) || 0);
          return bv - av;
        });
        return list;
      }

      function syncSelected(list) {
        if (!list.length) {
          selected = null;
          return;
        }
        if (!selected) {
          selected = list[0];
          return;
        }
        var stillVisible = list.some(function (u) { return u.user_key === selected.user_key; });
        if (!stillVisible) {
          selected = list[0];
        }
      }

      function renderListControls() {
        el.sortLuck.className = "list-btn" + (sortMode === "luck" ? " active" : "");
        el.sortSkill.className = "list-btn" + (sortMode === "skill" ? " active" : "");
        el.sortPnl.className = "list-btn" + (sortMode === "pnl" ? " active" : "");
        el.filterOpen.textContent = openOnly ? "Open Positions Only" : "All Users";
        el.filterOpen.className = "list-btn" + (openOnly ? " active" : "");
      }

      function renderUserList() {
        var list = visibleUsers();
        syncSelected(list);
        renderListControls();
        el.users.innerHTML = "";
        list.forEach(function (u) {
          var li = document.createElement("li");
          li.className = selected && selected.user_key === u.user_key ? "active" : "";
          var m = u.metrics || {};
          var closed = u.closed_summary || {};
          var leftLine = "Luck " + fmt(m.luck_index) + " (SD=" + signed2(m.luck_z) + ") | Skill " + fmt(m.skill_index) + " (SD=" + signed2(m.skill_z) + ")";
          var pnlLine = 'PnL ' + esc(fmt(m.raw_pnl, "usd")) +
            ' (<span class="pos">' + esc(fmt(closed.win_usdc, "usd")) + '</span> / <span class="neg">' + esc(fmt(closed.loss_usdc, "usd")) + '</span>)';
          li.innerHTML =
            '<div><div class="uname">' + esc(u.display_name) + '</div><div class="umetrics">' + esc(leftLine) + '</div><div class="umetrics">' + pnlLine + '</div></div>';
          li.onclick = function () {
            selected = u;
            renderUserList();
            renderDetail();
          };
          el.users.appendChild(li);
        });
        if (!list.length) {
          el.users.innerHTML = '<li><div class="uname">No users match current filters</div></li>';
        }
      }

      function renderCards(metrics) {
        var luckBubble = fmt(metrics.luck_index) + " (SD=" + signed2(metrics.luck_z) + ")";
        var skillBubble = fmt(metrics.skill_index) + " (SD=" + signed2(metrics.skill_z) + ")";
        var luckLabel = luckSdLabel(metrics.luck_z);
        var skillLabel = skillSdLabel(metrics.skill_z);
        var pnlBreakdown =
          '<div class="v">' + esc(fmt(metrics.raw_pnl, "usd")) + '</div>' +
          '<div class="line"><b>Skill:</b> ' + esc(fmt(metrics.skill_usdc, "usd")) + '</div>' +
          '<div class="line"><b>Luck:</b> ' + esc(fmt(metrics.total_luck, "usd")) + '</div>';
        var cards = [
          ["Luck Index", luckBubble, labelChip(luckLabel, metrics.luck_z)],
          ["Skill Index", skillBubble, labelChip(skillLabel, metrics.skill_z)],
          ["PnL Breakdown", "", pnlBreakdown],
        ];
        el.cards.innerHTML = "";
        cards.forEach(function (pair) {
          var k = pair[0];
          var v = pair[1];
          var extra = pair.length > 2 ? pair[2] : "";
          var div = document.createElement("div");
          div.className = "card";
          div.innerHTML = '<div class="k">' + esc(k) + '</div>' + (v ? ('<div class="v">' + esc(v) + '</div>') : "") + extra;
          el.cards.appendChild(div);
        });
      }

      function renderRows(rows, target, closed) {
        target.innerHTML = "";
        rows.forEach(function (r, idx) {
          var tr = document.createElement("tr");
          var col5 = closed ? r.pnl : r.position_value;
          var col6 = closed ? (n(r.buy_notional) || 0) + (n(r.sell_notional) || 0) : r.pnl;
          var detailId = (closed ? "closed" : "open") + "-trades-" + idx;
          if (closed) {
            var rowPnl = n(r.pnl) || 0;
            if (rowPnl > 0) tr.className = "closed-win";
            if (rowPnl < 0) tr.className = "closed-loss";
          }
          var href = marketHref(r);
          var marketCell = "<td>" + esc(r.market_title) + "</td>";
          if (href) {
            marketCell = '<td><a href="' + href + '" target="_blank" rel="noopener noreferrer">' + esc(r.market_title) + "</a></td>";
          }
          if (closed) {
            tr.innerHTML =
              marketCell +
              "<td>" + fmt(r.net_shares) + "</td>" +
              "<td>" + fmt(r.reference_price) + "</td>" +
              '<td class="' + cls(col5) + '">' + fmt(col5, "usd") + "</td>" +
              '<td class="' + cls(col6) + '">' + fmt(col6, "usd") + "</td>" +
              "<td>" + fmt(r.trade_count) + "</td>";

            if (Array.isArray(r.trades) && r.trades.length) {
              var cTradesCell = tr.lastElementChild;
              var cBtn = document.createElement("button");
              cBtn.className = "trades-toggle";
              cBtn.type = "button";
              cBtn.textContent = "Show trades";
              cBtn.onclick = function (id, buttonRef) {
                return function () { toggleOpenTrades(id, buttonRef); };
              }(detailId, cBtn);
              cTradesCell.appendChild(document.createTextNode(" "));
              cTradesCell.appendChild(cBtn);
            }
          } else {
            tr.innerHTML =
              marketCell +
              "<td>" + fmt(r.net_shares) + "</td>" +
              "<td>" + fmt(r.reference_price) + "</td>" +
              "<td>" + esc(fmtTradeTs(r.expected_close)) + "</td>" +
              '<td class="' + cls(col5) + '">' + fmt(col5, "usd") + "</td>" +
              '<td class="' + cls(col6) + '">' + fmt(col6, "usd") + "</td>" +
              "<td>" + fmt(r.trade_count) + "</td>";

            if (Array.isArray(r.trades) && r.trades.length) {
              var tradesCell = tr.lastElementChild;
              var btn = document.createElement("button");
              btn.className = "trades-toggle";
              btn.type = "button";
              btn.textContent = "Show trades";
              btn.onclick = function (id, buttonRef) {
                return function () { toggleOpenTrades(id, buttonRef); };
              }(detailId, btn);
              tradesCell.appendChild(document.createTextNode(" "));
              tradesCell.appendChild(btn);
            }
          }
          target.appendChild(tr);

          if (Array.isArray(r.trades) && r.trades.length) {
            var detailRow = document.createElement("tr");
            detailRow.className = "open-detail-row";
            detailRow.id = detailId;
            detailRow.style.display = "none";
            detailRow.innerHTML = '<td colspan="' + (closed ? "6" : "7") + '">' + renderTradeSubtable(r.trades) + "</td>";
            target.appendChild(detailRow);
          }
        });
        if (!rows.length) {
          target.innerHTML = '<tr><td colspan="' + (closed ? "6" : "7") + '">No rows available.</td></tr>';
        }
      }

      function renderTradeSubtable(trades) {
        var body = trades.map(function (t) {
          return "<tr>" +
            "<td>" + esc(fmtTradeTs(t.timestamp)) + "</td>" +
            "<td>" + esc(t.side || "") + "</td>" +
            "<td>" + fmt(t.size) + "</td>" +
            "<td>" + fmt(t.price) + "</td>" +
            "</tr>";
        }).join("");
        return '<table class="trade-subtable">' +
          "<thead><tr><th>Timestamp</th><th>Side</th><th>Shares</th><th>Price</th></tr></thead>" +
          "<tbody>" + body + "</tbody>" +
          "</table>";
      }

      function toggleOpenTrades(id, btn) {
        var row = document.getElementById(id);
        if (!row) return;
        var showing = row.style.display !== "none";
        row.style.display = showing ? "none" : "table-row";
        if (btn) btn.textContent = showing ? "Show trades" : "Hide trades";
      }

      function fmtTradeTs(v) {
        if (v === null || v === undefined || v === "") return "NA";
        var x = Number(v);
        if (Number.isFinite(x)) {
          if (x > 0 && x < 1e12) x = x * 1000;
          var d = new Date(x);
          if (!isNaN(d.getTime())) return d.toLocaleString();
        }
        var d2 = new Date(String(v));
        if (!isNaN(d2.getTime())) return d2.toLocaleString();
        return String(v);
      }

      function renderDetail() {
        if (!selected) {
          el.title.textContent = "No user selected";
          el.subtitle.textContent = "";
          el.profile.innerHTML = "";
          el.cards.innerHTML = "";
          el.openMeta.innerHTML = "";
          el.closedMeta.innerHTML = "";
          el.closedMetaWl.innerHTML = "";
          el.openRows.innerHTML = "";
          el.closedRows.innerHTML = "";
          return;
        }

        var p = selected.profile || {};
        var m = selected.metrics || {};
        var open = selected.open_summary || { rows: [] };
        var closed = selected.closed_summary || { rows: [] };
        m.open_trade_count = open.trade_count || 0;
        m.closed_trade_count = closed.trade_count || 0;

        var titleName = selected.display_name || selected.user_key || "User";
        var wallet = (p.wallet || "").trim();
        if (wallet) {
          el.title.innerHTML = '<a href="https://polymarket.com/profile/' + encodeURIComponent(wallet) + '" target="_blank" rel="noopener noreferrer">' + esc(titleName) + "</a>";
        } else {
          el.title.textContent = titleName;
        }
        el.subtitle.textContent = "";
        el.profile.innerHTML =
          '<div class="line"><b>Pseudonym:</b> ' + esc(p.profile_pseudonym || "") + "</div>" +
          '<div class="line"><b>Wallet:</b> ' + esc(p.wallet || "") + "</div>";

        renderCards(m);

        el.openMeta.innerHTML =
          "<span>Markets: " + fmt(open.market_count) + "</span>" +
          "<span>Positions: " + fmt(open.position_count) + "</span>" +
          "<span>Trades: " + fmt(open.trade_count) + "</span>" +
          "<span>Net Value: " + fmt(open.net_value, "usd") + "</span>" +
          "<span>Est PnL: <span class='" + cls(open.est_pnl) + "'>" + fmt(open.est_pnl, "usd") + "</span></span>";

        el.closedMeta.innerHTML =
          "<span>Markets: " + fmt(closed.market_count) + "</span>" +
          "<span>Positions: " + fmt(closed.position_count) + "</span>" +
          "<span>Trades: " + fmt(closed.trade_count) + "</span>" +
          "<span>PnL: <span class='" + cls(closed.pnl) + "'>" + fmt(closed.pnl, "usd") + "</span></span>" +
          "<span>Notional: " + fmt(closed.notional, "usd") + "</span>";

        el.closedMetaWl.innerHTML =
          "<span>Wins: " + fmt(closed.win_count) + " (<span class='pos'>" + fmt(closed.win_usdc, "usd") + "</span>)</span>" +
          "<span>Losses: " + fmt(closed.loss_count) + " (<span class='neg'>" + fmt(closed.loss_usdc, "usd") + "</span>)</span>";

        renderRows(Array.isArray(open.rows) ? open.rows : [], el.openRows, false);
        renderRows(Array.isArray(closed.rows) ? closed.rows : [], el.closedRows, true);
      }

      el.generated.textContent = "Generated: " + (dashboardData.generated_at_utc || "unknown");
      el.sortLuck.onclick = function () {
        sortMode = "luck";
        renderUserList();
        renderDetail();
      };
      el.sortSkill.onclick = function () {
        sortMode = "skill";
        renderUserList();
        renderDetail();
      };
      el.sortPnl.onclick = function () {
        sortMode = "pnl";
        renderUserList();
        renderDetail();
      };
      el.filterOpen.onclick = function () {
        openOnly = !openOnly;
        renderUserList();
        renderDetail();
      };
      renderUserList();
      renderDetail();
    </script>
</body>
</html>
""".replace("__PAYLOAD__", data_json)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    report_users_dir = Path(args.report_users_dir)

    metrics_by_user = load_all_user_metrics(report_users_dir)
    if not metrics_by_user:
        print(f"No user metrics found in {report_users_dir}. Run luck_skill_analysis.py first.", file=sys.stderr)
        return

    markets_by_condition = load_markets(data_dir)

    users: List[Dict[str, Any]] = []
    for user_key in sorted(metrics_by_user.keys()):
        users.append(build_user_payload(user_key, metrics_by_user[user_key], data_dir, markets_by_condition))

    users.sort(
        key=lambda u: (u["metrics"].get("luck_index") is not None, u["metrics"].get("luck_index") or 0),
        reverse=True,
    )

    payload = build_dashboard_payload(users)
    html = build_dashboard_html(payload)

    output_file = Path(args.out)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard generated: {output_file}", file=sys.stderr)
    print("\n=== DASHBOARD USERS ===\n")
    for user in users:
        metrics = user["metrics"]
        print(
            f"User: {user['display_name']} | trades={metrics.get('total_trades')} "
            f"resolved={metrics.get('resolved_markets')} "
            f"open_positions={user['open_summary'].get('position_count')} "
            f"closed_positions={user['closed_summary'].get('position_count')}"
        )


if __name__ == "__main__":
    main()
