from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from config import CONDITION_RE
from utils import parse_jsonish_list, safe_float, safe_int, to_iso_utc


DB_FILENAME = "polymarket.sqlite3"
SCHEMA_VERSION = 1
WALLET_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def default_db_path(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / DB_FILENAME


def sqlite_database_available(data_dir: str | Path = "data", db_path: str | Path | None = None) -> bool:
    path = Path(db_path) if db_path else default_db_path(data_dir)
    if not path.exists() or not path.is_file():
        return False
    try:
        conn = connect(path)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'built_at'").fetchone()
            return bool(row and row["value"])
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.DatabaseError as exc:
        conn.close()
        if "idx_price_history_condition_time" in str(exc):
            _drop_optional_price_history_index(db_path)
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        else:
            raise
    return conn


def _drop_optional_price_history_index(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute(
            "DELETE FROM sqlite_master WHERE type = 'index' AND name = 'idx_price_history_condition_time'"
        )
        conn.execute("PRAGMA writable_schema = OFF")
        conn.commit()
    finally:
        conn.close()


def _is_malformed_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "database disk image is malformed" in text or "malformed database schema" in text


def _iter_rows_lenient(cursor: sqlite3.Cursor) -> Iterator[sqlite3.Row]:
    while True:
        try:
            row = cursor.fetchone()
        except sqlite3.DatabaseError as exc:
            if _is_malformed_error(exc):
                break
            raise
        if row is None:
            break
        yield row


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_text(path: Path | None) -> str:
    return str(path) if path else ""


def _mtime_iso(path: Path | None) -> str:
    if path is None:
        return _utc_now_iso()
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _mtime(path: Path | None) -> float:
    if path is None:
        return datetime.now(timezone.utc).timestamp()
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _raw_json(row: Dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _truthy(value: Any) -> int:
    return 1 if str(value or "").strip().lower() in {"1", "true", "yes", "y"} else 0


def _normalize_condition_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if CONDITION_RE.match(text) else ""


def _safe_user_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if WALLET_RE.match(text) else ""


def _safe_alias(value: Any) -> str:
    text = str(value or "").strip().lower().lstrip("@")
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in "._-")
    return cleaned[:140]


def _market_status_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    if _truthy(row.get("closed")) or str(row.get("umaResolutionStatus") or "").strip().lower() == "resolved":
        return "closed", "Closed"
    if _truthy(row.get("active")):
        return "open", "Open"

    end_text = str(row.get("endDate") or "").strip()
    if end_text:
        try:
            text = end_text[:-1] + "+00:00" if end_text.endswith("Z") else end_text
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return ("open", "Open") if dt.astimezone(timezone.utc) > datetime.now(timezone.utc) else ("ended", "Ended")
        except ValueError:
            pass
    return "unknown", "Unknown"


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS meta;
        DROP TABLE IF EXISTS market_current;
        DROP TABLE IF EXISTS user_profiles;
        DROP TABLE IF EXISTS user_trades;
        DROP TABLE IF EXISTS market_trades;
        DROP TABLE IF EXISTS price_history;
        DROP TABLE IF EXISTS user_metrics;
        DROP TABLE IF EXISTS scanner_candidate_users;
        DROP TABLE IF EXISTS scanner_jump_events;
        DROP TABLE IF EXISTS scanner_user_event_scores;
        DROP TABLE IF EXISTS indexed_user_trades;
        DROP TABLE IF EXISTS indexed_market_trades;
        DROP TABLE IF EXISTS indexed_price_history;
        DROP TABLE IF EXISTS user_analysis_warnings;
        DROP TABLE IF EXISTS user_aliases;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE market_current (
            condition_id TEXT PRIMARY KEY,
            question TEXT,
            title TEXT,
            slug TEXT,
            event_slug TEXT,
            closed INTEGER,
            active INTEGER,
            closed_time TEXT,
            end_date TEXT,
            winner TEXT,
            winning_outcome_index INTEGER,
            outcomes TEXT,
            outcome_prices TEXT,
            clob_token_ids TEXT,
            liquidity REAL,
            volume REAL,
            volume24hr REAL,
            uma_resolution_status TEXT,
            raw_json TEXT NOT NULL,
            cache_path TEXT,
            cache_mtime REAL,
            cache_mtime_iso TEXT
        );

        CREATE TABLE user_profiles (
            user_key TEXT PRIMARY KEY,
            name TEXT,
            pseudonym TEXT,
            bio TEXT,
            profile_image TEXT,
            raw_json TEXT NOT NULL,
            cache_path TEXT,
            cache_mtime REAL,
            cache_mtime_iso TEXT
        );

        CREATE TABLE user_trades (
            user_key TEXT NOT NULL,
            trade_record_key TEXT NOT NULL,
            transaction_hash TEXT,
            proxy_wallet TEXT,
            wallet TEXT,
            timestamp REAL,
            timestamp_utc TEXT,
            condition_id TEXT,
            side TEXT,
            outcome TEXT,
            outcome_index INTEGER,
            asset TEXT,
            size REAL,
            price REAL,
            notional REAL,
            title TEXT,
            slug TEXT,
            event_slug TEXT,
            name TEXT,
            pseudonym TEXT,
            raw_json TEXT NOT NULL,
            cache_path TEXT,
            PRIMARY KEY (user_key, trade_record_key)
        );

        CREATE TABLE market_trades (
            condition_id TEXT NOT NULL,
            trade_record_key TEXT NOT NULL,
            transaction_hash TEXT,
            proxy_wallet TEXT,
            timestamp REAL,
            timestamp_utc TEXT,
            side TEXT,
            outcome TEXT,
            outcome_index INTEGER,
            asset TEXT,
            size REAL,
            price REAL,
            notional REAL,
            title TEXT,
            slug TEXT,
            event_slug TEXT,
            name TEXT,
            pseudonym TEXT,
            raw_json TEXT NOT NULL,
            cache_path TEXT,
            PRIMARY KEY (condition_id, trade_record_key)
        );

        CREATE TABLE price_history (
            asset TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            timestamp_utc TEXT,
            price REAL,
            source TEXT,
            raw_json TEXT NOT NULL,
            cache_path TEXT,
            PRIMARY KEY (asset, timestamp, source)
        );

        CREATE TABLE user_metrics (
            user_key TEXT PRIMARY KEY,
            total_trades INTEGER,
            resolved_trade_count INTEGER,
            luck_index REAL,
            luck_z REAL,
            skill_index REAL,
            skill_z REAL,
            skill_usdc REAL,
            skill_roi REAL,
            raw_pnl_usdc REAL,
            gross_trade_notional_usdc REAL,
            pnl_per_trade_usdc REAL,
            pnl_per_resolved_trade_usdc REAL,
            skill_per_trade_usdc REAL,
            skill_per_resolved_trade_usdc REAL,
            monetized_edge_per_trade_usdc REAL,
            profitable_resolved_markets INTEGER,
            losing_resolved_markets INTEGER,
            profitable_market_ratio REAL,
            largest_market_profit_usdc REAL,
            largest_market_loss_usdc REAL,
            warning_count INTEGER,
            raw_json TEXT NOT NULL,
            cache_path TEXT,
            cache_mtime REAL,
            cache_mtime_iso TEXT
        );

        CREATE TABLE scanner_candidate_users (
            wallet TEXT PRIMARY KEY,
            total_jump_capture REAL,
            cluster_capped_total_jump_capture REAL,
            timing_z REAL,
            timing_index REAL,
            number_of_jumps_captured INTEGER,
            number_of_independent_events INTEGER,
            max_single_event_contribution REAL,
            max_single_event_share REAL,
            gross_notional REAL,
            avg_directional_ratio REAL,
            confidence_level TEXT,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE scanner_jump_events (
            event_id TEXT PRIMARY KEY,
            cluster_id TEXT,
            condition_id TEXT,
            question TEXT,
            slug TEXT,
            event_slug TEXT,
            jump_time REAL,
            jump_time_utc TEXT,
            p_before REAL,
            p_after REAL,
            delta_p REAL,
            jump_size_logit REAL,
            vol_sigma REAL,
            volume_during_jump REAL,
            volume_before_jump REAL,
            price_window_seconds INTEGER,
            participants INTEGER,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE scanner_user_event_scores (
            event_id TEXT NOT NULL,
            wallet TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            cluster_id TEXT,
            jump_capture REAL,
            raw_jump_capture REAL,
            signed_exposure REAL,
            gross_notional REAL,
            directional_ratio REAL,
            directional_weight REAL,
            chosen_lookback_seconds INTEGER,
            jump_time REAL,
            delta_p REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (event_id, wallet, condition_id)
        );

        CREATE TABLE indexed_user_trades (
            user_key TEXT PRIMARY KEY,
            indexed_at TEXT NOT NULL,
            file_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL
        );

        CREATE TABLE indexed_market_trades (
            condition_id TEXT PRIMARY KEY,
            indexed_at TEXT NOT NULL,
            cache_path TEXT NOT NULL,
            cache_mtime REAL NOT NULL,
            file_count INTEGER NOT NULL,
            row_count INTEGER NOT NULL
        );

        CREATE TABLE indexed_price_history (
            cache_path TEXT PRIMARY KEY,
            indexed_at TEXT NOT NULL,
            cache_mtime REAL NOT NULL,
            row_count INTEGER NOT NULL
        );

        CREATE TABLE user_analysis_warnings (
            user_key TEXT NOT NULL,
            warning_index INTEGER NOT NULL,
            warning TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_key, warning_index)
        );

        CREATE TABLE user_aliases (
            alias TEXT PRIMARY KEY,
            user_key TEXT NOT NULL,
            display_name TEXT,
            source TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX idx_user_trades_user_time ON user_trades(user_key, timestamp DESC);
        CREATE INDEX idx_user_trades_condition ON user_trades(condition_id);
        CREATE INDEX idx_market_trades_condition_time ON market_trades(condition_id, timestamp DESC);
        CREATE INDEX idx_scanner_scores_wallet ON scanner_user_event_scores(wallet);
        CREATE INDEX idx_scanner_scores_condition ON scanner_user_event_scores(condition_id);
        """
    )


def ensure_database(data_dir: str | Path = "data", db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path else default_db_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    first_init = not sqlite_database_available(data_dir, path)
    conn = connect(path)
    try:
        if first_init:
            initialize_schema(conn)
            with conn:
                conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))
                conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("built_at", _utc_now_iso()))
                conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", ("data_dir", str(Path(data_dir))))
        else:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_user_trades (
                    user_key TEXT PRIMARY KEY,
                    indexed_at TEXT NOT NULL,
                    file_count INTEGER NOT NULL,
                    row_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_market_trades (
                    condition_id TEXT PRIMARY KEY,
                    indexed_at TEXT NOT NULL,
                    cache_path TEXT NOT NULL,
                    cache_mtime REAL NOT NULL,
                    file_count INTEGER NOT NULL,
                    row_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_price_history (
                    cache_path TEXT PRIMARY KEY,
                    indexed_at TEXT NOT NULL,
                    cache_mtime REAL NOT NULL,
                    row_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_analysis_warnings (
                    user_key TEXT NOT NULL,
                    warning_index INTEGER NOT NULL,
                    warning TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_key, warning_index)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_aliases (
                    alias TEXT PRIMARY KEY,
                    user_key TEXT NOT NULL,
                    display_name TEXT,
                    source TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_user_trades_user_time ON user_trades(user_key, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_user_trades_condition ON user_trades(condition_id);
                CREATE INDEX IF NOT EXISTS idx_market_trades_condition_time ON market_trades(condition_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_scanner_scores_wallet ON scanner_user_event_scores(wallet);
                CREATE INDEX IF NOT EXISTS idx_scanner_scores_condition ON scanner_user_event_scores(condition_id);
                """
            )
            conn.commit()
    finally:
        conn.close()
    return path


def _insert_market(conn: sqlite3.Connection, path: Path, row: Dict[str, Any]) -> bool:
    condition_id = _normalize_condition_id(row.get("conditionId") or path.stem.removeprefix("market_"))
    if not condition_id:
        return False
    mtime = _mtime(path)
    conn.execute(
        """
        INSERT INTO market_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(condition_id) DO UPDATE SET
            question=excluded.question,
            title=excluded.title,
            slug=excluded.slug,
            event_slug=excluded.event_slug,
            closed=excluded.closed,
            active=excluded.active,
            closed_time=excluded.closed_time,
            end_date=excluded.end_date,
            winner=excluded.winner,
            winning_outcome_index=excluded.winning_outcome_index,
            outcomes=excluded.outcomes,
            outcome_prices=excluded.outcome_prices,
            clob_token_ids=excluded.clob_token_ids,
            liquidity=excluded.liquidity,
            volume=excluded.volume,
            volume24hr=excluded.volume24hr,
            uma_resolution_status=excluded.uma_resolution_status,
            raw_json=excluded.raw_json,
            cache_path=excluded.cache_path,
            cache_mtime=excluded.cache_mtime,
            cache_mtime_iso=excluded.cache_mtime_iso
        WHERE excluded.cache_mtime >= market_current.cache_mtime
        """,
        (
            condition_id,
            row.get("question") or row.get("title") or "",
            row.get("question") or row.get("title") or row.get("slug") or condition_id,
            row.get("slug") or "",
            row.get("eventSlug") or row.get("event_slug") or "",
            _truthy(row.get("closed")),
            _truthy(row.get("active")),
            row.get("closedTime") or "",
            row.get("endDate") or row.get("endDateIso") or "",
            row.get("winner") or "",
            safe_int(row.get("winningOutcomeIndex")),
            row.get("outcomes") or "",
            row.get("outcomePrices") or "",
            row.get("clobTokenIds") or "",
            safe_float(row.get("liquidityNum") or row.get("liquidity")),
            safe_float(row.get("volumeNum") or row.get("volume")),
            safe_float(row.get("volume24hr")),
            row.get("umaResolutionStatus") or "",
            _raw_json(row),
            _path_text(path),
            mtime,
            _mtime_iso(path),
        ),
    )
    return True


def _trade_key(row: Dict[str, Any], path: Path | None, index: int) -> str:
    fallback = f"{path.stem}:{index}" if path is not None else f"row:{index}"
    return str(
        row.get("trade_record_key")
        or row.get("id")
        or row.get("transactionHash")
        or "|".join(
            str(row.get(key) or "")
            for key in ("conditionId", "proxyWallet", "asset", "side", "timestamp", "price", "size")
        )
        or fallback
    )


def _insert_user_trade(conn: sqlite3.Connection, user_key: str, path: Path, row: Dict[str, Any], index: int) -> bool:
    condition_id = _normalize_condition_id(row.get("conditionId"))
    ts = safe_float(row.get("timestamp"))
    size = safe_float(row.get("size"))
    price = safe_float(row.get("price"))
    notional = size * price if size is not None and price is not None else None
    conn.execute(
        """
        INSERT OR REPLACE INTO user_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_key,
            _trade_key(row, path, index),
            row.get("transactionHash") or "",
            row.get("proxyWallet") or "",
            row.get("wallet") or row.get("proxyWallet") or "",
            ts,
            row.get("timestamp_utc") or to_iso_utc(ts),
            condition_id,
            row.get("side") or "",
            row.get("outcome") or "",
            safe_int(row.get("outcomeIndex")),
            row.get("asset") or "",
            size,
            price,
            notional,
            row.get("title") or row.get("market_title") or row.get("slug") or "",
            row.get("slug") or "",
            row.get("eventSlug") or row.get("event_slug") or "",
            row.get("name") or row.get("profile_name") or "",
            row.get("pseudonym") or row.get("profile_pseudonym") or "",
            _raw_json(row),
            _path_text(path),
        ),
    )
    return True


def _insert_market_trade(conn: sqlite3.Connection, condition_id: str, path: Path, row: Dict[str, Any], index: int) -> bool:
    ts = safe_float(row.get("timestamp"))
    size = safe_float(row.get("size"))
    price = safe_float(row.get("price"))
    notional = size * price if size is not None and price is not None else None
    conn.execute(
        """
        INSERT OR REPLACE INTO market_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            condition_id,
            _trade_key(row, path, index),
            row.get("transactionHash") or "",
            row.get("proxyWallet") or "",
            ts,
            row.get("timestamp_utc") or to_iso_utc(ts),
            row.get("side") or "",
            row.get("outcome") or "",
            safe_int(row.get("outcomeIndex")),
            row.get("asset") or "",
            size,
            price,
            notional,
            row.get("title") or row.get("market_title") or row.get("slug") or "",
            row.get("slug") or "",
            row.get("eventSlug") or row.get("event_slug") or "",
            row.get("name") or "",
            row.get("pseudonym") or "",
            _raw_json(row),
            _path_text(path),
        ),
    )
    return True


PRICE_HISTORY_INSERT_SQL = """
    INSERT OR REPLACE INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

PRICE_HISTORY_IMPORT_SQL = """
    INSERT OR IGNORE INTO price_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def _price_values(path: Path | None, row: Dict[str, Any], *, compact_raw: bool = False) -> Tuple[Any, ...] | None:
    asset = str(row.get("asset") or "").strip()
    condition_id = _normalize_condition_id(row.get("conditionId"))
    ts = safe_float(row.get("timestamp"))
    if not asset or not condition_id or ts is None:
        return None
    return (
        asset,
        condition_id,
        ts,
        row.get("timestamp_utc") or to_iso_utc(ts),
        safe_float(row.get("price")),
        row.get("source") or "",
        "{}" if compact_raw else _raw_json(row),
        _path_text(path),
    )


def _insert_price(conn: sqlite3.Connection, path: Path | None, row: Dict[str, Any]) -> bool:
    values = _price_values(path, row)
    if values is None:
        return False
    conn.execute(PRICE_HISTORY_INSERT_SQL, values)
    return True


def _insert_profile(conn: sqlite3.Connection, user_key: str, path: Path, row: Dict[str, Any]) -> bool:
    conn.execute(
        """
        INSERT OR REPLACE INTO user_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_key,
            row.get("name") or row.get("profile_name") or "",
            row.get("pseudonym") or row.get("profile_pseudonym") or "",
            row.get("bio") or "",
            row.get("profileImage") or row.get("profile_image") or "",
            _raw_json(row),
            _path_text(path),
            _mtime(path),
            _mtime_iso(path),
        ),
    )
    return True


def _insert_metric(conn: sqlite3.Connection, path: Path, row: Dict[str, Any]) -> bool:
    user_key = _safe_user_key(row.get("user_key"))
    if not user_key:
        return False
    conn.execute(
        """
        INSERT OR REPLACE INTO user_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_key,
            safe_int(row.get("total_trades")),
            safe_int(row.get("resolved_trade_count")),
            safe_float(row.get("luck_index")),
            safe_float(row.get("luck_z")),
            safe_float(row.get("skill_index")),
            safe_float(row.get("skill_z")),
            safe_float(row.get("skill_usdc")),
            safe_float(row.get("skill_roi")),
            safe_float(row.get("raw_pnl_usdc")),
            safe_float(row.get("gross_trade_notional_usdc")),
            safe_float(row.get("pnl_per_trade_usdc")),
            safe_float(row.get("pnl_per_resolved_trade_usdc")),
            safe_float(row.get("skill_per_trade_usdc")),
            safe_float(row.get("skill_per_resolved_trade_usdc")),
            safe_float(row.get("monetized_edge_per_trade_usdc")),
            safe_int(row.get("profitable_resolved_markets")),
            safe_int(row.get("losing_resolved_markets")),
            safe_float(row.get("profitable_market_ratio")),
            safe_float(row.get("largest_market_profit_usdc")),
            safe_float(row.get("largest_market_loss_usdc")),
            safe_int(row.get("warning_count")),
            _raw_json(row),
            _path_text(path),
            _mtime(path),
            _mtime_iso(path),
        ),
    )
    return True


def _insert_candidate(conn: sqlite3.Connection, row: Dict[str, Any]) -> bool:
    wallet = _safe_user_key(row.get("wallet"))
    if not wallet:
        return False
    conn.execute(
        """
        INSERT OR REPLACE INTO scanner_candidate_users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            wallet,
            safe_float(row.get("total_jump_capture")),
            safe_float(row.get("cluster_capped_total_jump_capture")),
            safe_float(row.get("timing_z")),
            safe_float(row.get("timing_index")),
            safe_int(row.get("number_of_jumps_captured")),
            safe_int(row.get("number_of_independent_events")),
            safe_float(row.get("max_single_event_contribution")),
            safe_float(row.get("max_single_event_share")),
            safe_float(row.get("gross_notional")),
            safe_float(row.get("avg_directional_ratio")),
            row.get("confidence_level") or "",
            _raw_json(row),
        ),
    )
    return True


def _insert_jump_event(conn: sqlite3.Connection, row: Dict[str, Any]) -> bool:
    event_id = str(row.get("event_id") or "").strip()
    condition_id = _normalize_condition_id(row.get("condition_id"))
    if not event_id or not condition_id:
        return False
    conn.execute(
        """
        INSERT OR REPLACE INTO scanner_jump_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            row.get("cluster_id") or "",
            condition_id,
            row.get("question") or "",
            row.get("slug") or "",
            row.get("event_slug") or "",
            safe_float(row.get("jump_time")),
            row.get("jump_time_utc") or to_iso_utc(row.get("jump_time")),
            safe_float(row.get("p_before")),
            safe_float(row.get("p_after")),
            safe_float(row.get("delta_p")),
            safe_float(row.get("jump_size_logit")),
            safe_float(row.get("vol_sigma")),
            safe_float(row.get("volume_during_jump")),
            safe_float(row.get("volume_before_jump")),
            safe_int(row.get("price_window_seconds")),
            safe_int(row.get("participants")),
            _raw_json(row),
        ),
    )
    return True


def _insert_user_event_score(conn: sqlite3.Connection, row: Dict[str, Any]) -> bool:
    event_id = str(row.get("event_id") or "").strip()
    wallet = _safe_user_key(row.get("wallet"))
    condition_id = _normalize_condition_id(row.get("condition_id"))
    if not event_id or not wallet or not condition_id:
        return False
    conn.execute(
        """
        INSERT OR REPLACE INTO scanner_user_event_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            wallet,
            condition_id,
            row.get("cluster_id") or "",
            safe_float(row.get("jump_capture")),
            safe_float(row.get("raw_jump_capture")),
            safe_float(row.get("signed_exposure")),
            safe_float(row.get("gross_notional")),
            safe_float(row.get("directional_ratio")),
            safe_float(row.get("directional_weight")),
            safe_int(row.get("chosen_lookback_seconds")),
            safe_float(row.get("jump_time")),
            safe_float(row.get("delta_p")),
            _raw_json(row),
        ),
    )
    return True


def database_summary(db_path: str | Path) -> Dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"db_path": str(path), "exists": False}
    conn = connect(path)
    try:
        tables = [
            "market_current",
            "user_profiles",
            "user_trades",
            "market_trades",
            "price_history",
            "user_metrics",
            "scanner_candidate_users",
            "scanner_jump_events",
            "scanner_user_event_scores",
            "indexed_user_trades",
            "indexed_market_trades",
            "indexed_price_history",
        ]
        counts = {table: _table_count_lenient(conn, table) for table in tables}
        meta = {row["key"]: row["value"] for row in _iter_rows_lenient(conn.execute("SELECT key, value FROM meta"))}
    finally:
        conn.close()
    return {"db_path": str(path), "exists": True, "counts": counts, "meta": meta}


def _table_count_lenient(conn: sqlite3.Connection, table: str) -> int:
    attempts = [
        f"SELECT COUNT(*) FROM {table}",
        f"SELECT COUNT(rowid) FROM {table} NOT INDEXED",
    ]
    for sql in attempts:
        try:
            return int(conn.execute(sql).fetchone()[0])
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
    count = 0
    try:
        cursor = conn.execute(f"SELECT rowid FROM {table} NOT INDEXED")
        for _ in _iter_rows_lenient(cursor):
            count += 1
    except sqlite3.DatabaseError as exc:
        if not _is_malformed_error(exc):
            raise
    return count


def upsert_user_profile(
    data_dir: str | Path,
    user_key: str,
    row: Dict[str, Any],
    *,
    db_path: str | Path | None = None,
) -> None:
    path = ensure_database(data_dir, db_path)
    key = _safe_user_key(user_key)
    if not key:
        return
    conn = connect(path)
    try:
        with conn:
            _insert_profile(conn, key, None, row)
    finally:
        conn.close()


def upsert_user_trades(
    data_dir: str | Path,
    user_key: str,
    rows: Sequence[Dict[str, Any]],
    *,
    replace_user: bool = False,
    db_path: str | Path | None = None,
) -> int:
    path = ensure_database(data_dir, db_path)
    key = _safe_user_key(user_key)
    if not key:
        return 0
    conn = connect(path)
    written = 0
    try:
        with conn:
            if replace_user:
                conn.execute("DELETE FROM user_trades WHERE user_key = ?", (key,))
            for idx, row in enumerate(rows):
                if _insert_user_trade(conn, key, None, row, idx):
                    written += 1
            indexed = conn.execute("SELECT row_count FROM indexed_user_trades WHERE user_key = ?", (key,)).fetchone()
            if indexed is not None:
                row_count = conn.execute("SELECT COUNT(*) FROM user_trades WHERE user_key = ?", (key,)).fetchone()[0]
                conn.execute(
                    "INSERT OR REPLACE INTO indexed_user_trades VALUES (?, ?, ?, ?)",
                    (key, _utc_now_iso(), 0, row_count),
                )
    finally:
        conn.close()
    return written


def upsert_user_alias(
    data_dir: str | Path,
    alias: str,
    user_key: str,
    *,
    display_name: str = "",
    source: str = "manual",
    db_path: str | Path | None = None,
) -> bool:
    path = ensure_database(data_dir, db_path)
    clean_alias = _safe_alias(alias)
    key = _safe_user_key(user_key)
    if not clean_alias or not key:
        return False
    conn = connect(path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_aliases VALUES (?, ?, ?, ?, ?)
                """,
                (clean_alias, key, str(display_name or ""), str(source or ""), _utc_now_iso()),
            )
        return True
    finally:
        conn.close()


def load_user_alias_rows(
    data_dir: str | Path,
    *,
    db_path: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    aliases: Dict[str, Dict[str, Any]] = {}
    try:
        try:
            cursor = conn.execute("SELECT alias, user_key, display_name, source, updated_at FROM user_aliases")
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
            cursor = conn.execute(
                "SELECT alias, user_key, display_name, source, updated_at FROM user_aliases NOT INDEXED"
            )
        for row in _iter_rows_lenient(cursor):
            alias = str(row["alias"] or "")
            user_key = str(row["user_key"] or "")
            if not alias or not user_key:
                continue
            aliases[alias] = {
                "alias": alias,
                "user_key": user_key,
                "display_name": str(row["display_name"] or alias),
                "source": str(row["source"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            }
        return aliases
    finally:
        conn.close()


def load_user_trade_uids(
    data_dir: str | Path,
    user_key: str,
    *,
    db_path: str | Path | None = None,
) -> set[str]:
    path = ensure_database(data_dir, db_path)
    key = _safe_user_key(user_key)
    if not key:
        return set()
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT trade_record_key, transaction_hash, raw_json
            FROM user_trades
            WHERE user_key = ?
            """,
            (key,),
        )
        uids: set[str] = set()
        for row in rows:
            for value in (row["trade_record_key"], row["transaction_hash"]):
                if value:
                    uids.add(str(value))
            try:
                raw = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                raw = {}
            for value in (raw.get("id"), raw.get("trade_record_key"), raw.get("transactionHash")):
                if value:
                    uids.add(str(value))
        return uids
    finally:
        conn.close()


def load_all_user_trades(
    data_dir: str | Path,
    user_key: str,
    *,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    path = ensure_database(data_dir, db_path)
    key = _safe_user_key(user_key)
    if not key:
        return []
    conn = connect(path)
    try:
        rows = []
        for row in conn.execute(
            "SELECT raw_json FROM user_trades WHERE user_key = ? ORDER BY timestamp",
            (key,),
        ):
            try:
                rows.append(json.loads(row["raw_json"]))
            except json.JSONDecodeError:
                continue
        return rows
    finally:
        conn.close()


def user_has_trades(data_dir: str | Path, user_key: str, *, db_path: str | Path | None = None) -> bool:
    path = ensure_database(data_dir, db_path)
    key = _safe_user_key(user_key)
    if not key:
        return False
    conn = connect(path)
    try:
        return bool(conn.execute("SELECT 1 FROM user_trades WHERE user_key = ? LIMIT 1", (key,)).fetchone())
    finally:
        conn.close()


def list_user_keys(
    data_dir: str | Path,
    *,
    include_candidates: bool = False,
    accepted_confidences: Iterable[str] | None = None,
    db_path: str | Path | None = None,
) -> List[str]:
    path = ensure_database(data_dir, db_path)
    accepted = {str(x).strip().lower() for x in (accepted_confidences or []) if str(x).strip()}
    conn = connect(path)
    try:
        keys: set[str] = set()
        try:
            rows = conn.execute(
                """
                SELECT user_key FROM user_profiles
                UNION SELECT user_key FROM user_metrics
                UNION SELECT user_key FROM user_trades
                """
            )
            keys.update(str(row["user_key"]) for row in _iter_rows_lenient(rows) if row["user_key"])
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
            for sql in (
                "SELECT user_key FROM user_profiles",
                "SELECT user_key FROM user_metrics",
                "SELECT user_key FROM indexed_user_trades",
                "SELECT user_key FROM user_aliases",
            ):
                try:
                    rows = conn.execute(sql)
                except sqlite3.DatabaseError as fallback_exc:
                    if not _is_malformed_error(fallback_exc):
                        raise
                    continue
                keys.update(str(row["user_key"]) for row in _iter_rows_lenient(rows) if row["user_key"])
        else:
            try:
                rows = conn.execute("SELECT user_key FROM user_aliases")
                keys.update(str(row["user_key"]) for row in _iter_rows_lenient(rows) if row["user_key"])
            except sqlite3.DatabaseError as exc:
                if not _is_malformed_error(exc):
                    raise
        if include_candidates:
            sql = "SELECT wallet, confidence_level FROM scanner_candidate_users"
            for row in conn.execute(sql):
                confidence = str(row["confidence_level"] or "").lower()
                if accepted and confidence not in accepted:
                    continue
                if row["wallet"]:
                    keys.add(str(row["wallet"]))
        return sorted(keys)
    finally:
        conn.close()


def candidate_wallets(
    data_dir: str | Path,
    accepted_confidences: Iterable[str],
    *,
    db_path: str | Path | None = None,
) -> List[str]:
    path = ensure_database(data_dir, db_path)
    accepted = {str(x).strip().lower() for x in accepted_confidences if str(x).strip()}
    conn = connect(path)
    try:
        rows = conn.execute(
            """
            SELECT wallet, confidence_level, timing_z
            FROM scanner_candidate_users
            ORDER BY
                CASE confidence_level
                    WHEN 'very_high' THEN 4
                    WHEN 'high' THEN 3
                    WHEN 'candidate' THEN 2
                    WHEN 'monitor' THEN 1
                    ELSE 0
                END DESC,
                timing_z DESC
            """
        )
        out = []
        seen = set()
        for row in rows:
            confidence = str(row["confidence_level"] or "").strip().lower()
            wallet = str(row["wallet"] or "").strip().lower()
            if accepted and confidence not in accepted:
                continue
            if wallet and wallet not in seen:
                seen.add(wallet)
                out.append(wallet)
        return out
    finally:
        conn.close()


def upsert_markets(
    data_dir: str | Path,
    rows: Sequence[Dict[str, Any]],
    *,
    db_path: str | Path | None = None,
) -> int:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    written = 0
    try:
        with conn:
            for row in rows:
                if _insert_market(conn, None, row):
                    written += 1
    finally:
        conn.close()
    return written


def market_condition_ids(data_dir: str | Path, *, db_path: str | Path | None = None) -> set[str]:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    try:
        return {str(row["condition_id"]) for row in conn.execute("SELECT condition_id FROM market_current")}
    finally:
        conn.close()


def count_market_rows(
    data_dir: str | Path,
    *,
    query: str = "",
    db_path: str | Path | None = None,
) -> int:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    where = ""
    params: List[Any] = []
    if query:
        like = f"%{query}%"
        where = (
            " WHERE condition_id LIKE ? OR title LIKE ? OR question LIKE ? "
            "OR slug LIKE ? OR event_slug LIKE ?"
        )
        params = [like, like, like, like, like]
    try:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM market_current{where}", params).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
            try:
                row = conn.execute(f"SELECT COUNT(rowid) FROM market_current NOT INDEXED{where}", params).fetchone()
                return int(row[0]) if row else 0
            except sqlite3.DatabaseError as fallback_exc:
                if not _is_malformed_error(fallback_exc):
                    raise
        count = 0
        try:
            cursor = conn.execute(f"SELECT rowid FROM market_current NOT INDEXED{where}", params)
            for _ in _iter_rows_lenient(cursor):
                count += 1
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
        return count
    finally:
        conn.close()


def _decode_market_payload(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        payload = json.loads(row["raw_json"])
    except json.JSONDecodeError:
        payload = {}
    condition_id = str(row["condition_id"] or "")
    payload["conditionId"] = payload.get("conditionId") or condition_id
    payload["question"] = payload.get("question") or row["question"] or ""
    payload["title"] = payload.get("title") or row["title"] or ""
    payload["slug"] = payload.get("slug") or row["slug"] or ""
    payload["eventSlug"] = payload.get("eventSlug") or payload.get("event_slug") or row["event_slug"] or ""
    payload["volume"] = payload.get("volume") if payload.get("volume") is not None else row["volume"]
    payload["volumeNum"] = payload.get("volumeNum") if payload.get("volumeNum") is not None else row["volume"]
    payload["liquidity"] = payload.get("liquidity") if payload.get("liquidity") is not None else row["liquidity"]
    payload["liquidityNum"] = (
        payload.get("liquidityNum") if payload.get("liquidityNum") is not None else row["liquidity"]
    )
    payload["_db_updated_at"] = row["cache_mtime_iso"] or ""
    return payload


def load_market_summary_rows(
    data_dir: str | Path,
    *,
    query: str = "",
    limit: int = 1000,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    where = ""
    params: List[Any] = []
    if query:
        like = f"%{query}%"
        where = (
            " WHERE condition_id LIKE ? OR title LIKE ? OR question LIKE ? "
            "OR slug LIKE ? OR event_slug LIKE ?"
        )
        params = [like, like, like, like, like]
    params.append(max(1, min(limit, 5000)))
    sql = f"""
        SELECT condition_id, question, title, slug, event_slug, volume, liquidity, raw_json, cache_mtime_iso
        FROM market_current
        {where}
        ORDER BY COALESCE(cache_mtime, 0) DESC, COALESCE(volume, 0) DESC, condition_id
        LIMIT ?
    """
    try:
        try:
            cursor = conn.execute(sql, params)
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
            try:
                cursor = conn.execute(sql.replace("FROM market_current", "FROM market_current NOT INDEXED"), params)
            except sqlite3.DatabaseError as fallback_exc:
                if not _is_malformed_error(fallback_exc):
                    raise
                cursor = conn.execute(
                    """
                    SELECT condition_id, question, title, slug, event_slug, volume, liquidity, raw_json, cache_mtime_iso
                    FROM market_current NOT INDEXED
                    """
                )
                rows = []
                needle = query.lower()
                for row in _iter_rows_lenient(cursor):
                    payload = _decode_market_payload(row)
                    if needle and not any(
                        needle in str(payload.get(key) or "").lower()
                        for key in ("conditionId", "question", "title", "slug", "eventSlug")
                    ):
                        continue
                    rows.append(payload)
                    if len(rows) >= params[-1]:
                        break
                return rows
        return [_decode_market_payload(row) for row in _iter_rows_lenient(cursor)]
    finally:
        conn.close()


def load_all_market_rows(data_dir: str | Path, *, db_path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    try:
        markets: Dict[str, Dict[str, Any]] = {}
        cursor = conn.execute("SELECT condition_id, raw_json FROM market_current NOT INDEXED")
        for row in _iter_rows_lenient(cursor):
            try:
                raw = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                continue
            condition_id = _normalize_condition_id(raw.get("conditionId") or row["condition_id"])
            if condition_id:
                raw["conditionId"] = condition_id
                markets[condition_id] = raw
        return markets
    finally:
        conn.close()


def upsert_price_history(
    data_dir: str | Path,
    rows: Sequence[Dict[str, Any]],
    *,
    db_path: str | Path | None = None,
) -> int:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    written = 0
    try:
        with conn:
            for row in rows:
                if _insert_price(conn, None, row):
                    written += 1
    finally:
        conn.close()
    return written


def price_history_cached(
    data_dir: str | Path,
    token_id: str,
    start_ts: int,
    end_ts: int,
    *,
    db_path: str | Path | None = None,
) -> bool:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    try:
        count = conn.execute(
            """
            SELECT COUNT(*) FROM price_history
            WHERE asset = ? AND timestamp BETWEEN ? AND ?
            """,
            (str(token_id), float(start_ts), float(end_ts)),
        ).fetchone()[0]
        return count > 0
    finally:
        conn.close()


def load_price_history_points(
    data_dir: str | Path,
    assets: Iterable[str] | None = None,
    *,
    db_path: str | Path | None = None,
) -> Dict[str, List[Tuple[float, float]]]:
    path = ensure_database(data_dir, db_path)
    asset_filter = {str(asset) for asset in (assets or []) if str(asset).strip()}
    conn = connect(path)
    try:
        by_asset: Dict[str, List[Tuple[float, float]]] = {}
        if asset_filter:
            placeholders = ",".join("?" for _ in asset_filter)
            sql = f"""
                SELECT asset, timestamp, price FROM price_history
                WHERE asset IN ({placeholders})
                ORDER BY asset, timestamp
            """
            iterator = conn.execute(sql, sorted(asset_filter))
        else:
            iterator = conn.execute("SELECT asset, timestamp, price FROM price_history ORDER BY asset, timestamp")
        for row in iterator:
            asset = str(row["asset"] or "")
            ts = safe_float(row["timestamp"])
            px = safe_float(row["price"])
            if asset and ts is not None and px is not None:
                by_asset.setdefault(asset, []).append((ts, px))
        return by_asset
    finally:
        conn.close()


def upsert_market_trades(
    data_dir: str | Path,
    condition_id: str,
    rows: Sequence[Dict[str, Any]],
    *,
    replace_market: bool = False,
    db_path: str | Path | None = None,
) -> int:
    path = ensure_database(data_dir, db_path)
    cid = _normalize_condition_id(condition_id)
    if not cid:
        return 0
    conn = connect(path)
    written = 0
    try:
        with conn:
            if replace_market:
                conn.execute("DELETE FROM market_trades WHERE condition_id = ?", (cid,))
            for idx, row in enumerate(rows):
                if _insert_market_trade(conn, cid, None, row, idx):
                    written += 1
    finally:
        conn.close()
    return written


def load_market_trade_rows(
    data_dir: str | Path,
    condition_id: str,
    *,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    path = ensure_database(data_dir, db_path)
    cid = _normalize_condition_id(condition_id)
    if not cid:
        return []
    conn = connect(path)
    try:
        rows = []
        for row in conn.execute(
            "SELECT raw_json FROM market_trades WHERE condition_id = ? ORDER BY timestamp",
            (cid,),
        ):
            try:
                rows.append(json.loads(row["raw_json"]))
            except json.JSONDecodeError:
                continue
        return rows
    finally:
        conn.close()


def market_trade_uids(
    data_dir: str | Path,
    condition_id: str,
    *,
    db_path: str | Path | None = None,
) -> set[str]:
    path = ensure_database(data_dir, db_path)
    cid = _normalize_condition_id(condition_id)
    if not cid:
        return set()
    conn = connect(path)
    try:
        uids = set()
        for row in conn.execute(
            "SELECT trade_record_key, transaction_hash, raw_json FROM market_trades WHERE condition_id = ?",
            (cid,),
        ):
            for value in (row["trade_record_key"], row["transaction_hash"]):
                if value:
                    uids.add(str(value))
            try:
                raw = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                raw = {}
            for value in (raw.get("id"), raw.get("trade_record_key"), raw.get("transactionHash")):
                if value:
                    uids.add(str(value))
        return uids
    finally:
        conn.close()


def upsert_user_metric(
    data_dir: str | Path,
    row: Dict[str, Any],
    *,
    warnings: Sequence[str] | None = None,
    db_path: str | Path | None = None,
) -> None:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    try:
        with conn:
            _insert_metric(conn, None, row)
            user_key = _safe_user_key(row.get("user_key"))
            if user_key:
                conn.execute("DELETE FROM user_analysis_warnings WHERE user_key = ?", (user_key,))
                for idx, warning in enumerate(warnings or []):
                    conn.execute(
                        "INSERT OR REPLACE INTO user_analysis_warnings VALUES (?, ?, ?, ?)",
                        (user_key, idx, str(warning), _utc_now_iso()),
                    )
    finally:
        conn.close()


def upsert_user_metrics(
    data_dir: str | Path,
    rows: Sequence[Dict[str, Any]],
    *,
    db_path: str | Path | None = None,
) -> int:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    written = 0
    try:
        with conn:
            for row in rows:
                if _insert_metric(conn, None, row):
                    written += 1
    finally:
        conn.close()
    return written


def upsert_scanner_results(
    data_dir: str | Path,
    *,
    jump_events: Sequence[Dict[str, Any]],
    user_event_scores: Sequence[Dict[str, Any]],
    candidate_users: Sequence[Dict[str, Any]],
    replace: bool = True,
    db_path: str | Path | None = None,
) -> Dict[str, int]:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    counts = {"jump_events": 0, "user_event_scores": 0, "candidate_users": 0}
    try:
        with conn:
            if replace:
                conn.execute("DELETE FROM scanner_jump_events")
                conn.execute("DELETE FROM scanner_user_event_scores")
                conn.execute("DELETE FROM scanner_candidate_users")
            for row in jump_events:
                if _insert_jump_event(conn, row):
                    counts["jump_events"] += 1
            for row in user_event_scores:
                if _insert_user_event_score(conn, row):
                    counts["user_event_scores"] += 1
            for row in candidate_users:
                if _insert_candidate(conn, row):
                    counts["candidate_users"] += 1
    finally:
        conn.close()
    return counts


def load_user_metric_rows(db_path: str | Path) -> List[Dict[str, Any]]:
    conn = connect(db_path)
    try:
        rows = []
        keys: List[str] = []
        try:
            cursor = conn.execute(
                "SELECT user_key FROM user_metrics INDEXED BY sqlite_autoindex_user_metrics_1 ORDER BY user_key"
            )
            keys = [str(row["user_key"]) for row in _iter_rows_lenient(cursor) if row["user_key"]]
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
        if keys:
            for user_key in keys:
                try:
                    row = conn.execute(
                        "SELECT raw_json, cache_mtime_iso FROM user_metrics WHERE user_key = ?",
                        (user_key,),
                    ).fetchone()
                except sqlite3.DatabaseError as exc:
                    if _is_malformed_error(exc):
                        continue
                    raise
                if not row:
                    continue
                try:
                    payload = json.loads(row["raw_json"])
                except json.JSONDecodeError:
                    continue
                payload["_db_updated_at"] = row["cache_mtime_iso"] or ""
                rows.append(payload)
        else:
            cursor = conn.execute("SELECT raw_json, cache_mtime_iso FROM user_metrics")
            for row in _iter_rows_lenient(cursor):
                try:
                    payload = json.loads(row["raw_json"])
                except json.JSONDecodeError:
                    continue
                payload["_db_updated_at"] = row["cache_mtime_iso"] or ""
                rows.append(payload)
        return rows
    finally:
        conn.close()


def load_scanner_candidate_rows(db_path: str | Path) -> List[Dict[str, Any]]:
    conn = connect(db_path)
    try:
        return [
            json.loads(row["raw_json"])
            for row in _iter_rows_lenient(conn.execute("SELECT raw_json FROM scanner_candidate_users"))
        ]
    finally:
        conn.close()


def load_scanner_jump_event_rows(db_path: str | Path) -> List[Dict[str, Any]]:
    conn = connect(db_path)
    try:
        return [
            json.loads(row["raw_json"])
            for row in _iter_rows_lenient(conn.execute("SELECT raw_json FROM scanner_jump_events"))
        ]
    finally:
        conn.close()


def load_market_raw_row(
    data_dir: str | Path,
    condition_id: str,
    *,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(data_dir, db_path)
    cid = _normalize_condition_id(condition_id)
    if not cid:
        return {}
    conn = connect(path)
    try:
        try:
            row = conn.execute("SELECT raw_json FROM market_current WHERE condition_id = ?", (cid,)).fetchone()
        except sqlite3.DatabaseError as exc:
            if _is_malformed_error(exc):
                try:
                    row = conn.execute(
                        "SELECT raw_json FROM market_current NOT INDEXED WHERE condition_id = ?",
                        (cid,),
                    ).fetchone()
                except sqlite3.DatabaseError as fallback_exc:
                    if _is_malformed_error(fallback_exc):
                        row = None
                    else:
                        raise
            else:
                raise
        if not row:
            return {}
        try:
            raw = json.loads(row["raw_json"])
        except json.JSONDecodeError:
            return {}
        raw["conditionId"] = raw.get("conditionId") or cid
        return raw
    finally:
        conn.close()


def load_user_profile_row(
    data_dir: str | Path,
    user_key: str,
    *,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(data_dir, db_path)
    key = _safe_user_key(user_key)
    if not key:
        return {}
    conn = connect(path)
    try:
        try:
            row = conn.execute(
                "SELECT raw_json, cache_mtime_iso FROM user_profiles WHERE user_key = ?",
                (key,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            if _is_malformed_error(exc):
                row = conn.execute(
                    "SELECT raw_json, cache_mtime_iso FROM user_profiles NOT INDEXED WHERE user_key = ?",
                    (key,),
                ).fetchone()
            else:
                raise
        if not row:
            return {}
        try:
            payload = json.loads(row["raw_json"])
        except json.JSONDecodeError:
            return {}
        payload["_db_updated_at"] = row["cache_mtime_iso"] or ""
        return payload
    finally:
        conn.close()


def load_user_profile_rows(
    data_dir: str | Path,
    *,
    db_path: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    path = ensure_database(data_dir, db_path)
    conn = connect(path)
    profiles: Dict[str, Dict[str, Any]] = {}
    try:
        try:
            cursor = conn.execute("SELECT user_key, name, pseudonym, raw_json, cache_mtime_iso FROM user_profiles")
        except sqlite3.DatabaseError as exc:
            if not _is_malformed_error(exc):
                raise
            cursor = conn.execute(
                "SELECT user_key, name, pseudonym, raw_json, cache_mtime_iso FROM user_profiles NOT INDEXED"
            )
        for row in _iter_rows_lenient(cursor):
            key = str(row["user_key"] or "")
            if not key:
                continue
            try:
                payload = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                payload = {}
            payload["user_key"] = key
            payload["name"] = payload.get("name") or row["name"] or ""
            payload["pseudonym"] = payload.get("pseudonym") or row["pseudonym"] or ""
            payload["_db_updated_at"] = row["cache_mtime_iso"] or ""
            profiles[key] = payload
        return profiles
    finally:
        conn.close()


def load_user_trade_rows(
    db_path: str | Path,
    user_key: str,
    *,
    query: str = "",
) -> Tuple[List[Dict[str, Any]], int, int]:
    key = _safe_user_key(user_key)
    if not key:
        return [], 0, 0
    conn = connect(db_path)
    try:
        try:
            total = conn.execute("SELECT COUNT(*) FROM user_trades WHERE user_key = ?", (key,)).fetchone()[0]
        except sqlite3.DatabaseError as exc:
            if _is_malformed_error(exc):
                total = 0
            else:
                raise
        where = "user_key = ?"
        params: List[Any] = [key]
        if query:
            like = f"%{query}%"
            where += (
                " AND (title LIKE ? OR slug LIKE ? OR event_slug LIKE ? OR condition_id LIKE ? "
                "OR outcome LIKE ? OR side LIKE ?)"
            )
            params.extend([like, like, like, like, like, like])
        sql = f"""
            SELECT * FROM user_trades
            WHERE {where}
            ORDER BY timestamp DESC
        """
        rows = []
        try:
            cursor = conn.execute(sql, params)
        except sqlite3.DatabaseError as exc:
            if _is_malformed_error(exc):
                cursor = conn.execute(
                    sql.replace("FROM user_trades", "FROM user_trades NOT INDEXED"),
                    params,
                )
            else:
                raise
        for row in _iter_rows_lenient(cursor):
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "timestamp_utc": row["timestamp_utc"] or to_iso_utc(row["timestamp"]),
                    "side": row["side"] or "",
                    "outcome": row["outcome"] or "",
                    "outcome_index": row["outcome_index"],
                    "price": row["price"],
                    "size": row["size"],
                    "notional": row["notional"],
                    "title": row["title"] or row["slug"] or "",
                    "slug": row["slug"] or "",
                    "event_slug": row["event_slug"] or "",
                    "condition_id": row["condition_id"] or "",
                    "asset": row["asset"] or "",
                    "transaction_hash": row["transaction_hash"] or "",
                }
            )
        if not total:
            total = len(rows)
        return rows, total, len(rows)
    finally:
        conn.close()


def load_market_statuses(db_path: str | Path, condition_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ids = sorted({_normalize_condition_id(cid) for cid in condition_ids if _normalize_condition_id(cid)})
    if not ids:
        return {}
    conn = connect(db_path)
    try:
        statuses: Dict[str, Dict[str, Any]] = {}
        for condition_id in ids:
            try:
                row = conn.execute("SELECT * FROM market_current WHERE condition_id = ?", (condition_id,)).fetchone()
            except sqlite3.DatabaseError as exc:
                if _is_malformed_error(exc):
                    try:
                        row = conn.execute(
                            "SELECT * FROM market_current NOT INDEXED WHERE condition_id = ?",
                            (condition_id,),
                        ).fetchone()
                    except sqlite3.DatabaseError as fallback_exc:
                        if _is_malformed_error(fallback_exc):
                            row = None
                        else:
                            raise
                else:
                    raise
            if not row:
                continue
            payload = json.loads(row["raw_json"])
            status, label = _market_status_from_row(payload)
            statuses[condition_id] = {
                "market_status": status,
                "market_status_label": label,
                "market_end_date": row["end_date"] or "",
                "market_active": str(row["active"] or ""),
                "market_closed": str(row["closed"] or ""),
                "market_title": row["title"] or row["slug"] or "",
                "market_slug": row["slug"] or "",
                "market_winner": row["winner"] or "",
                "market_winning_outcome_index": str(row["winning_outcome_index"] or ""),
                "market_outcomes": parse_jsonish_list(row["outcomes"]),
                "market_outcome_prices": parse_jsonish_list(row["outcome_prices"]),
            }
        return statuses
    finally:
        conn.close()


def load_user_scanner_market_signals(db_path: str | Path, user_key: str) -> Dict[str, Dict[str, Any]]:
    key = _safe_user_key(user_key)
    if not key:
        return {}
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT condition_id, event_id, jump_capture, gross_notional, directional_ratio, delta_p, jump_time
            FROM scanner_user_event_scores
            WHERE wallet = ?
            """,
            (key,),
        ).fetchall()
    finally:
        conn.close()

    signals: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        condition_id = row["condition_id"]
        rec = signals.setdefault(
            condition_id,
            {
                "scanner_user_event_count": 0,
                "scanner_user_positive_events": 0,
                "scanner_user_jump_capture": 0.0,
                "scanner_user_gross_notional": 0.0,
                "scanner_user_max_abs_capture": 0.0,
                "scanner_user_max_move": None,
                "scanner_user_last_jump_time": None,
                "scanner_user_last_jump_time_utc": "",
                "_event_ids": set(),
                "_directional_sum": 0.0,
                "_directional_count": 0,
            },
        )
        event_id = str(row["event_id"] or "")
        if event_id:
            rec["_event_ids"].add(event_id)
        capture = safe_float(row["jump_capture"]) or 0.0
        rec["scanner_user_jump_capture"] += capture
        rec["scanner_user_gross_notional"] += safe_float(row["gross_notional"]) or 0.0
        if capture > 0:
            rec["scanner_user_positive_events"] += 1
        rec["scanner_user_max_abs_capture"] = max(rec["scanner_user_max_abs_capture"], abs(capture))
        directional = safe_float(row["directional_ratio"])
        if directional is not None:
            rec["_directional_sum"] += directional
            rec["_directional_count"] += 1
        delta = safe_float(row["delta_p"])
        if delta is not None:
            existing_delta = safe_float(rec.get("scanner_user_max_move"))
            if existing_delta is None or abs(delta) > abs(existing_delta):
                rec["scanner_user_max_move"] = delta
        jump_time = safe_float(row["jump_time"])
        existing_time = safe_float(rec.get("scanner_user_last_jump_time"))
        if jump_time is not None and (existing_time is None or jump_time > existing_time):
            rec["scanner_user_last_jump_time"] = jump_time
            rec["scanner_user_last_jump_time_utc"] = to_iso_utc(jump_time)

    for rec in signals.values():
        rec["scanner_user_event_count"] = len(rec["_event_ids"])
        if rec["_directional_count"]:
            rec["scanner_user_avg_directional_ratio"] = rec["_directional_sum"] / rec["_directional_count"]
        else:
            rec["scanner_user_avg_directional_ratio"] = None
        rec.pop("_event_ids", None)
        rec.pop("_directional_sum", None)
        rec.pop("_directional_count", None)
    return signals
