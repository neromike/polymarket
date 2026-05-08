#!/usr/bin/env python3
"""Build an interactive dashboard for market scanner outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import safe_float, safe_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML dashboard for market scanner outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="reports/market_scanner",
        help="Directory containing candidate_users.csv, jump_events.csv, user_event_scores.csv",
    )
    parser.add_argument(
        "--out",
        default="reports/market_scanner_dashboard.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--max-wallet-events",
        type=int,
        default=120,
        help="Max event rows kept per wallet in detail pane",
    )
    parser.add_argument(
        "--max-event-wallets",
        type=int,
        default=40,
        help="Max top positive and negative wallets per event",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except OSError as exc:
        print(f"Failed reading {path}: {exc}", file=sys.stderr)
        return []


def _to_unix(ts: Any) -> Optional[float]:
    value = safe_float(ts)
    if value is None:
        return None
    if value > 10_000_000_000:
        value /= 1000.0
    return value


def build_payload(
    candidates_rows: List[Dict[str, str]],
    event_rows: List[Dict[str, str]],
    user_event_rows: List[Dict[str, str]],
    *,
    max_wallet_events: int,
    max_event_wallets: int,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    candidate_wallets = set()

    for row in candidates_rows:
        wallet = str(row.get("wallet") or "").strip().lower()
        if not wallet:
            continue
        candidate_wallets.add(wallet)
        candidates.append(
            {
                "wallet": wallet,
                "total_jump_capture": safe_float(row.get("total_jump_capture")) or 0.0,
                "timing_z": safe_float(row.get("timing_z")) or 0.0,
                "timing_index": safe_float(row.get("timing_index")) or 0.0,
                "number_of_jumps_captured": safe_int(row.get("number_of_jumps_captured")) or 0,
                "number_of_independent_events": safe_int(row.get("number_of_independent_events")) or 0,
                "max_single_event_contribution": safe_float(row.get("max_single_event_contribution")) or 0.0,
                "gross_notional": safe_float(row.get("gross_notional")) or 0.0,
                "confidence_level": str(row.get("confidence_level") or "low"),
            }
        )

    events: List[Dict[str, Any]] = []
    event_meta: Dict[str, Dict[str, Any]] = {}
    for row in event_rows:
        event_id = str(row.get("event_id") or "")
        if not event_id:
            continue
        jump_ts = _to_unix(row.get("jump_time"))
        event_obj = {
            "event_id": event_id,
            "condition_id": str(row.get("condition_id") or ""),
            "question": str(row.get("question") or ""),
            "slug": str(row.get("slug") or ""),
            "event_slug": str(row.get("event_slug") or ""),
            "jump_time": int(jump_ts) if jump_ts is not None else None,
            "p_before": safe_float(row.get("p_before")),
            "p_after": safe_float(row.get("p_after")),
            "delta_p": safe_float(row.get("delta_p")),
            "jump_size_logit": safe_float(row.get("jump_size_logit")),
            "vol_sigma": safe_float(row.get("vol_sigma")),
            "volume_during_jump": safe_float(row.get("volume_during_jump")),
            "volume_before_jump": safe_float(row.get("volume_before_jump")),
            "price_window_seconds": safe_int(row.get("price_window_seconds")) or 0,
            "participants": safe_int(row.get("participants")) or 0,
        }
        events.append(event_obj)
        event_meta[event_id] = event_obj

    wallet_event_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    event_wallet_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in user_event_rows:
        wallet = str(row.get("wallet") or "").strip().lower()
        event_id = str(row.get("event_id") or "").strip()
        if not wallet or not event_id:
            continue

        rec = {
            "event_id": event_id,
            "condition_id": str(row.get("condition_id") or ""),
            "jump_capture": safe_float(row.get("jump_capture")) or 0.0,
            "signed_exposure": safe_float(row.get("signed_exposure")) or 0.0,
            "gross_notional": safe_float(row.get("gross_notional")) or 0.0,
            "chosen_lookback_seconds": safe_int(row.get("chosen_lookback_seconds")) or 0,
            "jump_time": safe_int(row.get("jump_time")),
            "delta_p": safe_float(row.get("delta_p")),
        }

        if wallet in candidate_wallets:
            wallet_event_rows[wallet].append(rec)
        event_wallet_rows[event_id].append({"wallet": wallet, **rec})

    wallet_details: Dict[str, Dict[str, Any]] = {}
    for wallet in candidate_wallets:
        rows = wallet_event_rows.get(wallet, [])
        rows.sort(key=lambda r: abs(r["jump_capture"]), reverse=True)
        top = rows[:max_wallet_events]

        pos_count = sum(1 for r in rows if r["jump_capture"] > 0)
        neg_count = sum(1 for r in rows if r["jump_capture"] < 0)
        total = sum(r["jump_capture"] for r in rows)
        total_pos = sum(r["jump_capture"] for r in rows if r["jump_capture"] > 0)
        total_neg = -sum(r["jump_capture"] for r in rows if r["jump_capture"] < 0)

        enriched_top = []
        for r in top:
            meta = event_meta.get(r["event_id"], {})
            enriched_top.append(
                {
                    **r,
                    "question": meta.get("question", ""),
                    "slug": meta.get("slug", ""),
                    "event_slug": meta.get("event_slug", ""),
                    "participants": meta.get("participants", 0),
                }
            )

        wallet_details[wallet] = {
            "wallet": wallet,
            "event_count": len(rows),
            "positive_event_count": pos_count,
            "negative_event_count": neg_count,
            "total_jump_capture_from_rows": total,
            "positive_capture": total_pos,
            "negative_capture": total_neg,
            "top_events": enriched_top,
        }

    event_details: Dict[str, Dict[str, Any]] = {}
    for event_id, rows in event_wallet_rows.items():
        rows_sorted = sorted(rows, key=lambda r: r["jump_capture"], reverse=True)
        top_positive = [r for r in rows_sorted if r["jump_capture"] > 0][:max_event_wallets]
        top_negative = [r for r in rows_sorted[::-1] if r["jump_capture"] < 0][:max_event_wallets]
        all_caps = [r["jump_capture"] for r in rows]

        avg_capture = sum(all_caps) / len(all_caps) if all_caps else 0.0
        var = (
            sum((x - avg_capture) ** 2 for x in all_caps) / max(1, len(all_caps) - 1)
            if all_caps
            else 0.0
        )

        event_details[event_id] = {
            "event_id": event_id,
            "participant_count": len(rows),
            "mean_capture": avg_capture,
            "sd_capture": math.sqrt(max(0.0, var)),
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    confidence_counts: Dict[str, int] = defaultdict(int)
    for c in candidates:
        confidence_counts[c["confidence_level"]] += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "candidate_count": len(candidates),
            "event_count": len(events),
            "user_event_row_count": len(user_event_rows),
            "confidence_counts": dict(confidence_counts),
        },
        "candidates": candidates,
        "events": events,
        "wallet_details": wallet_details,
        "event_details": event_details,
    }


def build_html(payload: Dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True)
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Market Scanner Dashboard</title>
  <style>
    :root {
      --bg: #f3efe8;
      --panel: #ffffff;
      --ink: #1c253a;
      --muted: #5c6477;
      --line: #dde2eb;
      --accent: #0a7f6f;
      --accent2: #b54b2f;
      --good: #157347;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: \"Segoe UI\", \"Trebuchet MS\", sans-serif;
      background:
        radial-gradient(circle at 12% 10%, rgba(10,127,111,0.14), transparent 28%),
        radial-gradient(circle at 90% 6%, rgba(181,75,47,0.12), transparent 32%),
        linear-gradient(160deg, #f3efe8, #ecf3f1);
    }
    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 12px;
      min-height: 100vh;
      padding: 12px;
    }
    .panel {
      background: rgba(255,255,255,0.93);
      border: 1px solid rgba(28,37,58,0.11);
      border-radius: 14px;
      overflow: hidden;
    }
    .head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .head h1, .head h2 { margin: 0; font-size: 1.05rem; }
    .sub { color: var(--muted); font-size: 0.82rem; margin-top: 4px; }
    .controls {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 0.76rem;
      background: #fff;
      cursor: pointer;
      color: var(--ink);
    }
    .btn.active {
      background: rgba(10,127,111,0.12);
      border-color: var(--accent);
    }
    .search {
      width: 100%;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 0.8rem;
      color: var(--ink);
      background: #fff;
    }
    .list {
      list-style: none;
      margin: 0;
      padding: 0;
      max-height: calc(100vh - 215px);
      overflow: auto;
    }
    .list li {
      border-left: 3px solid transparent;
      border-bottom: 1px solid rgba(221,226,235,0.65);
      padding: 10px 12px;
      cursor: pointer;
    }
    .list li.active {
      border-left-color: var(--accent);
      background: rgba(10,127,111,0.12);
    }
    .list li:hover { background: rgba(28,37,58,0.06); }
    .w { font-weight: 700; font-size: 0.83rem; word-break: break-all; }
    .m { font-size: 0.77rem; color: var(--muted); margin-top: 2px; }
    .main { padding: 12px; display: grid; gap: 10px; }
    .cards { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 8px; }
    .card { border: 1px solid var(--line); border-radius: 10px; background: #fff; padding: 8px; }
    .k { font-size: 0.73rem; color: var(--muted); text-transform: uppercase; }
    .v { font-size: 1rem; font-weight: 800; margin-top: 4px; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .box { border: 1px solid var(--line); border-radius: 10px; background: #fff; }
    .box h3 { margin: 0; padding: 10px; border-bottom: 1px solid var(--line); font-size: 0.95rem; }
    .meta { padding: 8px 10px; font-size: 0.8rem; color: var(--muted); display: flex; gap: 12px; flex-wrap: wrap; }
    .table-wrap { max-height: 340px; overflow: auto; border-top: 1px solid var(--line); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 7px 8px; font-size: 0.8rem; border-bottom: 1px solid rgba(221,226,235,0.6); }
    th { position: sticky; top: 0; background: #f8fafc; color: var(--muted); }
    tr:hover td { background: rgba(28,37,58,0.03); }
    .pos { color: var(--good); }
    .neg { color: var(--bad); }
    .muted { color: var(--muted); }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--ink);
      background: #fff;
    }
    .topbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }
    @media (max-width: 1180px) {
      .layout { grid-template-columns: 1fr; }
      .cards { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .split { grid-template-columns: 1fr; }
      .list { max-height: 340px; }
    }
  </style>
</head>
<body>
  <div class=\"layout\">
    <aside class=\"panel\">
      <div class=\"head\">
        <h1>Scanner Candidates</h1>
        <div class=\"sub\" id=\"left-sub\"></div>
      </div>
      <div class=\"controls\">
        <button class=\"btn\" id=\"sort-z\">Timing Z</button>
        <button class=\"btn\" id=\"sort-index\">Timing Index</button>
        <button class=\"btn\" id=\"sort-cap\">Jump Capture</button>
        <button class=\"btn\" id=\"sort-events\">Event Count</button>
        <button class=\"btn\" id=\"conf-all\">All Confidence</button>
        <button class=\"btn\" id=\"conf-candidate\">Candidate+</button>
        <input class=\"search\" id=\"wallet-search\" placeholder=\"Filter wallet...\" />
      </div>
      <ul class=\"list\" id=\"candidate-list\"></ul>
    </aside>

    <main class=\"panel\">
      <div class=\"head\">
        <h2 id=\"title\">No wallet selected</h2>
        <div class=\"sub\" id=\"subtitle\"></div>
      </div>
      <div class=\"main\">
        <div class=\"topbar\">
          <button class=\"btn active\" id=\"view-wallet\">Wallet View</button>
          <button class=\"btn\" id=\"view-events\">Event Explorer</button>
          <span class=\"pill\" id=\"summary-pill\"></span>
        </div>

        <section id=\"wallet-view\">
          <div class=\"cards\" id=\"wallet-cards\"></div>
          <div class=\"split\">
            <div class=\"box\">
              <h3>Top Event Captures For Wallet</h3>
              <div class=\"meta\" id=\"wallet-events-meta\"></div>
              <div class=\"table-wrap\">
                <table>
                  <thead>
                    <tr>
                      <th>Jump Time</th><th>Market</th><th>Capture</th><th>Exposure</th><th>Notional</th><th>Lookback</th>
                    </tr>
                  </thead>
                  <tbody id=\"wallet-events-rows\"></tbody>
                </table>
              </div>
            </div>
            <div class=\"box\">
              <h3>Event Detail For Selected Row</h3>
              <div class=\"meta\" id=\"event-detail-meta\"></div>
              <div class=\"table-wrap\">
                <table>
                  <thead>
                    <tr><th>Wallet</th><th>Capture</th><th>Exposure</th><th>Notional</th></tr>
                  </thead>
                  <tbody id=\"event-detail-rows\"></tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        <section id=\"events-view\" style=\"display:none\">
          <div class=\"split\">
            <div class=\"box\">
              <h3>Jump Events</h3>
              <div class=\"meta\" id=\"events-meta\"></div>
              <div class=\"table-wrap\">
                <table>
                  <thead>
                    <tr><th>Jump Time</th><th>Market</th><th>Delta P</th><th>Logit Jump</th><th>Participants</th><th>Volume During</th></tr>
                  </thead>
                  <tbody id=\"events-rows\"></tbody>
                </table>
              </div>
            </div>
            <div class=\"box\">
              <h3>Selected Event Top Wallets</h3>
              <div class=\"meta\" id=\"events-selected-meta\"></div>
              <div class=\"table-wrap\">
                <table>
                  <thead>
                    <tr><th>Wallet</th><th>Capture</th><th>Exposure</th><th>Notional</th></tr>
                  </thead>
                  <tbody id=\"events-selected-rows\"></tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script id=\"dashboard-data\" type=\"application/json\">__PAYLOAD__</script>
  <script>
    var data = JSON.parse(document.getElementById("dashboard-data").textContent || "{}");
    var candidates = Array.isArray(data.candidates) ? data.candidates : [];
    var events = Array.isArray(data.events) ? data.events : [];
    var walletDetails = data.wallet_details || {};
    var eventDetails = data.event_details || {};
    var selectedWallet = candidates.length ? candidates[0].wallet : null;
    var selectedWalletEventId = null;
    var selectedEventId = events.length ? events[0].event_id : null;
    var sortMode = "z";
    var filterMode = "all";
    var viewMode = "wallet";

    var el = {
      leftSub: document.getElementById("left-sub"),
      sortZ: document.getElementById("sort-z"),
      sortIndex: document.getElementById("sort-index"),
      sortCap: document.getElementById("sort-cap"),
      sortEvents: document.getElementById("sort-events"),
      confAll: document.getElementById("conf-all"),
      confCandidate: document.getElementById("conf-candidate"),
      walletSearch: document.getElementById("wallet-search"),
      candidateList: document.getElementById("candidate-list"),
      title: document.getElementById("title"),
      subtitle: document.getElementById("subtitle"),
      summaryPill: document.getElementById("summary-pill"),
      viewWallet: document.getElementById("view-wallet"),
      viewEvents: document.getElementById("view-events"),
      walletView: document.getElementById("wallet-view"),
      eventsView: document.getElementById("events-view"),
      walletCards: document.getElementById("wallet-cards"),
      walletEventsMeta: document.getElementById("wallet-events-meta"),
      walletEventsRows: document.getElementById("wallet-events-rows"),
      eventDetailMeta: document.getElementById("event-detail-meta"),
      eventDetailRows: document.getElementById("event-detail-rows"),
      eventsMeta: document.getElementById("events-meta"),
      eventsRows: document.getElementById("events-rows"),
      eventsSelectedMeta: document.getElementById("events-selected-meta"),
      eventsSelectedRows: document.getElementById("events-selected-rows"),
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
      if (mode === "int") return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
      return x.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    function cls(v) {
      var x = n(v);
      if (x === null || x === 0) return "muted";
      return x > 0 ? "pos" : "neg";
    }
    function fmtTs(v) {
      var x = n(v);
      if (x === null) return "NA";
      if (x > 0 && x < 1e12) x = x * 1000;
      var d = new Date(x);
      return isNaN(d.getTime()) ? String(v) : d.toLocaleString();
    }
    function fmtLookback(sec) {
      var s = n(sec);
      if (s === null) return "NA";
      if (s % 86400 === 0) return (s / 86400) + "d";
      if (s % 3600 === 0) return (s / 3600) + "h";
      if (s % 60 === 0) return (s / 60) + "m";
      return s + "s";
    }

    function confidenceRank(label) {
      var l = String(label || "").toLowerCase();
      if (l === "very_high") return 4;
      if (l === "high") return 3;
      if (l === "candidate") return 2;
      if (l === "monitor") return 1;
      return 0;
    }

    function filteredCandidates() {
      var q = String(el.walletSearch.value || "").trim().toLowerCase();
      var out = candidates.filter(function (c) {
        var passConf = filterMode === "all" || confidenceRank(c.confidence_level) >= 2;
        var passSearch = !q || String(c.wallet || "").toLowerCase().indexOf(q) >= 0;
        return passConf && passSearch;
      });
      out.sort(function (a, b) {
        if (sortMode === "index") return (n(b.timing_index) || 0) - (n(a.timing_index) || 0);
        if (sortMode === "capture") return (n(b.total_jump_capture) || 0) - (n(a.total_jump_capture) || 0);
        if (sortMode === "events") return (n(b.number_of_independent_events) || 0) - (n(a.number_of_independent_events) || 0);
        return (n(b.timing_z) || 0) - (n(a.timing_z) || 0);
      });
      return out;
    }

    function syncSelectedWallet(list) {
      if (!list.length) {
        selectedWallet = null;
        return;
      }
      var exists = list.some(function (x) { return x.wallet === selectedWallet; });
      if (!exists) selectedWallet = list[0].wallet;
    }

    function renderCandidateList() {
      var list = filteredCandidates();
      syncSelectedWallet(list);

      el.sortZ.className = "btn" + (sortMode === "z" ? " active" : "");
      el.sortIndex.className = "btn" + (sortMode === "index" ? " active" : "");
      el.sortCap.className = "btn" + (sortMode === "capture" ? " active" : "");
      el.sortEvents.className = "btn" + (sortMode === "events" ? " active" : "");
      el.confAll.className = "btn" + (filterMode === "all" ? " active" : "");
      el.confCandidate.className = "btn" + (filterMode === "candidate_plus" ? " active" : "");

      el.leftSub.textContent =
        "Candidates: " + fmt((data.summary || {}).candidate_count, "int") +
        " | Events: " + fmt((data.summary || {}).event_count, "int") +
        " | Rows: " + fmt((data.summary || {}).user_event_row_count, "int");

      el.candidateList.innerHTML = "";
      list.forEach(function (c) {
        var li = document.createElement("li");
        if (c.wallet === selectedWallet) li.className = "active";
        var conf = String(c.confidence_level || "low");
        li.innerHTML =
          '<div class="w">' + esc(c.wallet) + '</div>' +
          '<div class="m">TimingZ ' + esc(fmt(c.timing_z)) +
          ' | Index ' + esc(fmt(c.timing_index)) +
          ' | Capture ' + esc(fmt(c.total_jump_capture, "usd")) + '</div>' +
          '<div class="m">Events ' + esc(fmt(c.number_of_independent_events, "int")) +
          ' | ' + esc(conf) + '</div>';
        li.onclick = function () {
          selectedWallet = c.wallet;
          selectedWalletEventId = null;
          renderAll();
        };
        el.candidateList.appendChild(li);
      });

      if (!list.length) {
        el.candidateList.innerHTML = '<li><div class="w">No matching candidates</div></li>';
      }
    }

    function selectedCandidate() {
      if (!selectedWallet) return null;
      for (var i = 0; i < candidates.length; i++) {
        if (candidates[i].wallet === selectedWallet) return candidates[i];
      }
      return null;
    }

    function renderCards(candidate, details) {
      var cards = [
        ["Timing Z", fmt(candidate ? candidate.timing_z : null)],
        ["Timing Index", fmt(candidate ? candidate.timing_index : null)],
        ["Total Jump Capture", fmt(candidate ? candidate.total_jump_capture : null, "usd")],
        ["Independent Events", fmt(candidate ? candidate.number_of_independent_events : null, "int")],
        ["Gross Notional", fmt(candidate ? candidate.gross_notional : null, "usd")],
        ["Confidence", candidate ? String(candidate.confidence_level || "") : "NA"],
      ];
      if (details) {
        cards.push(["Positive Events", fmt(details.positive_event_count, "int")]);
        cards.push(["Negative Events", fmt(details.negative_event_count, "int")]);
        cards.push(["Positive Capture", fmt(details.positive_capture, "usd")]);
        cards.push(["Negative Capture", fmt(details.negative_capture, "usd")]);
      }

      el.walletCards.innerHTML = "";
      cards.forEach(function (pair) {
        var div = document.createElement("div");
        div.className = "card";
        div.innerHTML = '<div class="k">' + esc(pair[0]) + '</div><div class="v">' + esc(pair[1]) + '</div>';
        el.walletCards.appendChild(div);
      });
    }

    function marketHref(slug, eventSlug, conditionId) {
      if (eventSlug) return "https://polymarket.com/event/" + encodeURIComponent(eventSlug);
      if (slug) return "https://polymarket.com/market/" + encodeURIComponent(slug);
      if (conditionId) return "https://polymarket.com/market/" + encodeURIComponent(conditionId);
      return "";
    }

    function renderWalletEvents(candidate, details) {
      el.walletEventsRows.innerHTML = "";
      if (!candidate || !details) {
        el.walletEventsMeta.innerHTML = "";
        el.walletEventsRows.innerHTML = '<tr><td colspan="6">No wallet selected.</td></tr>';
        return;
      }

      var rows = Array.isArray(details.top_events) ? details.top_events : [];
      el.walletEventsMeta.innerHTML =
        '<span>Total events: ' + esc(fmt(details.event_count, "int")) + '</span>' +
        '<span>Positive: <span class="pos">' + esc(fmt(details.positive_event_count, "int")) + '</span></span>' +
        '<span>Negative: <span class="neg">' + esc(fmt(details.negative_event_count, "int")) + '</span></span>';

      rows.forEach(function (r) {
        var tr = document.createElement("tr");
        var href = marketHref(r.slug, r.event_slug, r.condition_id);
        var title = r.question || r.condition_id || "Market";
        var titleCell = esc(title);
        if (href) {
          titleCell = '<a href="' + href + '" target="_blank" rel="noopener noreferrer">' + esc(title) + '</a>';
        }
        tr.innerHTML =
          '<td>' + esc(fmtTs(r.jump_time)) + '</td>' +
          '<td>' + titleCell + '</td>' +
          '<td class="' + cls(r.jump_capture) + '">' + esc(fmt(r.jump_capture, "usd")) + '</td>' +
          '<td>' + esc(fmt(r.signed_exposure)) + '</td>' +
          '<td>' + esc(fmt(r.gross_notional, "usd")) + '</td>' +
          '<td>' + esc(fmtLookback(r.chosen_lookback_seconds)) + '</td>';
        tr.onclick = function () {
          selectedWalletEventId = r.event_id;
          renderWalletSelectedEvent();
        };
        el.walletEventsRows.appendChild(tr);
      });

      if (!rows.length) {
        el.walletEventsRows.innerHTML = '<tr><td colspan="6">No event rows available.</td></tr>';
      }
    }

    function renderWalletSelectedEvent() {
      el.eventDetailRows.innerHTML = "";
      if (!selectedWalletEventId) {
        el.eventDetailMeta.innerHTML = "<span>Select a wallet event row to inspect counterparties.</span>";
        el.eventDetailRows.innerHTML = '<tr><td colspan="4">No event selected.</td></tr>';
        return;
      }
      var detail = eventDetails[selectedWalletEventId] || {};
      var eventMeta = null;
      for (var i = 0; i < events.length; i++) {
        if (events[i].event_id === selectedWalletEventId) {
          eventMeta = events[i];
          break;
        }
      }
      var title = eventMeta ? (eventMeta.question || eventMeta.condition_id) : selectedWalletEventId;
      el.eventDetailMeta.innerHTML =
        '<span>Event: ' + esc(title) + '</span>' +
        '<span>Participants: ' + esc(fmt(detail.participant_count, "int")) + '</span>' +
        '<span>Mean capture: ' + esc(fmt(detail.mean_capture, "usd")) + '</span>';

      var rows = [];
      (detail.top_positive || []).forEach(function (r) { rows.push(r); });
      (detail.top_negative || []).forEach(function (r) { rows.push(r); });
      rows.sort(function (a, b) { return (n(b.jump_capture) || 0) - (n(a.jump_capture) || 0); });

      rows.forEach(function (r) {
        var tr = document.createElement("tr");
        tr.innerHTML =
          '<td>' + esc(r.wallet) + '</td>' +
          '<td class="' + cls(r.jump_capture) + '">' + esc(fmt(r.jump_capture, "usd")) + '</td>' +
          '<td>' + esc(fmt(r.signed_exposure)) + '</td>' +
          '<td>' + esc(fmt(r.gross_notional, "usd")) + '</td>';
        tr.onclick = function () {
          selectedWallet = r.wallet;
          viewMode = "wallet";
          renderAll();
        };
        el.eventDetailRows.appendChild(tr);
      });

      if (!rows.length) {
        el.eventDetailRows.innerHTML = '<tr><td colspan="4">No participant rows.</td></tr>';
      }
    }

    function renderEventsTable() {
      var sorted = events.slice().sort(function (a, b) {
        return Math.abs(n(b.jump_size_logit) || 0) - Math.abs(n(a.jump_size_logit) || 0);
      });

      el.eventsMeta.innerHTML =
        '<span>Total events: ' + esc(fmt(sorted.length, "int")) + '</span>' +
        '<span>Top sorted by |logit jump|</span>';

      el.eventsRows.innerHTML = "";
      sorted.forEach(function (e) {
        var tr = document.createElement("tr");
        var href = marketHref(e.slug, e.event_slug, e.condition_id);
        var title = e.question || e.condition_id || "Event";
        var titleCell = esc(title);
        if (href) {
          titleCell = '<a href="' + href + '" target="_blank" rel="noopener noreferrer">' + esc(title) + '</a>';
        }
        tr.innerHTML =
          '<td>' + esc(fmtTs(e.jump_time)) + '</td>' +
          '<td>' + titleCell + '</td>' +
          '<td class="' + cls(e.delta_p) + '">' + esc(fmt(e.delta_p)) + '</td>' +
          '<td>' + esc(fmt(e.jump_size_logit)) + '</td>' +
          '<td>' + esc(fmt(e.participants, "int")) + '</td>' +
          '<td>' + esc(fmt(e.volume_during_jump, "usd")) + '</td>';
        tr.onclick = function () {
          selectedEventId = e.event_id;
          renderSelectedEventTable();
        };
        el.eventsRows.appendChild(tr);
      });

      if (!sorted.length) {
        el.eventsRows.innerHTML = '<tr><td colspan="6">No events found.</td></tr>';
      }
    }

    function renderSelectedEventTable() {
      el.eventsSelectedRows.innerHTML = "";
      if (!selectedEventId) {
        el.eventsSelectedMeta.innerHTML = "<span>No event selected.</span>";
        el.eventsSelectedRows.innerHTML = '<tr><td colspan="4">Select an event on the left.</td></tr>';
        return;
      }
      var detail = eventDetails[selectedEventId] || {};
      var eventMeta = null;
      for (var i = 0; i < events.length; i++) {
        if (events[i].event_id === selectedEventId) {
          eventMeta = events[i];
          break;
        }
      }
      var label = eventMeta ? (eventMeta.question || eventMeta.condition_id) : selectedEventId;
      el.eventsSelectedMeta.innerHTML =
        '<span>' + esc(label) + '</span>' +
        '<span>Participants: ' + esc(fmt(detail.participant_count, "int")) + '</span>' +
        '<span>Mean capture: ' + esc(fmt(detail.mean_capture, "usd")) + '</span>' +
        '<span>SD capture: ' + esc(fmt(detail.sd_capture, "usd")) + '</span>';

      var rows = [];
      (detail.top_positive || []).forEach(function (r) { rows.push(r); });
      (detail.top_negative || []).forEach(function (r) { rows.push(r); });
      rows.sort(function (a, b) { return (n(b.jump_capture) || 0) - (n(a.jump_capture) || 0); });

      rows.forEach(function (r) {
        var tr = document.createElement("tr");
        tr.innerHTML =
          '<td>' + esc(r.wallet) + '</td>' +
          '<td class="' + cls(r.jump_capture) + '">' + esc(fmt(r.jump_capture, "usd")) + '</td>' +
          '<td>' + esc(fmt(r.signed_exposure)) + '</td>' +
          '<td>' + esc(fmt(r.gross_notional, "usd")) + '</td>';
        tr.onclick = function () {
          selectedWallet = r.wallet;
          viewMode = "wallet";
          renderAll();
        };
        el.eventsSelectedRows.appendChild(tr);
      });

      if (!rows.length) {
        el.eventsSelectedRows.innerHTML = '<tr><td colspan="4">No participant rows.</td></tr>';
      }
    }

    function renderMainHeader(candidate) {
      var generated = String((data.generated_at_utc || "")).replace("T", " ").replace("+00:00", " UTC");
      el.summaryPill.textContent = generated ? ("Generated " + generated) : "Generated time unavailable";
      if (!candidate) {
        el.title.textContent = "No wallet selected";
        el.subtitle.textContent = "";
        return;
      }
      el.title.textContent = candidate.wallet;
      el.subtitle.textContent =
        "TimingZ " + fmt(candidate.timing_z) +
        " | TimingIndex " + fmt(candidate.timing_index) +
        " | Confidence " + String(candidate.confidence_level || "low");
    }

    function renderViewToggle() {
      el.viewWallet.className = "btn" + (viewMode === "wallet" ? " active" : "");
      el.viewEvents.className = "btn" + (viewMode === "events" ? " active" : "");
      el.walletView.style.display = viewMode === "wallet" ? "" : "none";
      el.eventsView.style.display = viewMode === "events" ? "" : "none";
    }

    function renderAll() {
      renderCandidateList();
      var candidate = selectedCandidate();
      var details = candidate ? walletDetails[candidate.wallet] : null;
      renderMainHeader(candidate);
      renderViewToggle();
      renderCards(candidate, details);
      renderWalletEvents(candidate, details);
      renderWalletSelectedEvent();
      renderEventsTable();
      renderSelectedEventTable();
    }

    el.sortZ.onclick = function () { sortMode = "z"; renderAll(); };
    el.sortIndex.onclick = function () { sortMode = "index"; renderAll(); };
    el.sortCap.onclick = function () { sortMode = "capture"; renderAll(); };
    el.sortEvents.onclick = function () { sortMode = "events"; renderAll(); };
    el.confAll.onclick = function () { filterMode = "all"; renderAll(); };
    el.confCandidate.onclick = function () { filterMode = "candidate_plus"; renderAll(); };
    el.walletSearch.oninput = function () { renderAll(); };
    el.viewWallet.onclick = function () { viewMode = "wallet"; renderAll(); };
    el.viewEvents.onclick = function () { viewMode = "events"; renderAll(); };

    renderAll();
  </script>
</body>
</html>
""".replace("__PAYLOAD__", data_json)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    candidate_path = input_dir / "candidate_users.csv"
    event_path = input_dir / "jump_events.csv"
    user_event_path = input_dir / "user_event_scores.csv"

    candidate_rows = read_csv_rows(candidate_path)
    event_rows = read_csv_rows(event_path)
    user_event_rows = read_csv_rows(user_event_path)

    if not candidate_rows:
        print(f"WARNING: No candidates found at {candidate_path}", file=sys.stderr)
    if not event_rows:
        print(f"WARNING: No events found at {event_path}", file=sys.stderr)
    if not user_event_rows:
        print(f"WARNING: No user-event scores found at {user_event_path}", file=sys.stderr)

    payload = build_payload(
        candidate_rows,
        event_rows,
        user_event_rows,
        max_wallet_events=max(1, args.max_wallet_events),
        max_event_wallets=max(1, args.max_event_wallets),
    )

    html = build_html(payload)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"Dashboard generated: {out_path}")


if __name__ == "__main__":
    main()
