from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


RUN_STATE_DIRNAME = "run_state"
JOBS_DIRNAME = "jobs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def reports_dir(root: Path | str = ".") -> Path:
    return Path(root) / "reports"


def run_state_dir(root: Path | str = ".") -> Path:
    return reports_dir(root) / RUN_STATE_DIRNAME


def jobs_dir(root: Path | str = ".") -> Path:
    return reports_dir(root) / JOBS_DIRNAME


def data_state_dir(data_dir: Path | str = "data") -> Path:
    return Path(data_dir) / ".state"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(data, encoding="utf-8")
    try:
        for attempt in range(5):
            try:
                tmp_path.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return dict(default or {})
    return data if isinstance(data, dict) else dict(default or {})


def process_is_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_query_limited_information = 0x1000
            still_active = 259
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid_int)
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False

    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True


def _job_log_tail(job: Dict[str, Any], limit: int = 200_000) -> str:
    log_path = Path(str(job.get("log_path") or ""))
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _looks_cancelled(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "keyboardinterrupt",
            "cancelled",
            "canceled",
            "cancellederror",
            "operation cancelled",
            "operation canceled",
        )
    )


def _job_age_seconds(job: Dict[str, Any], field: str) -> Optional[float]:
    value = str(job.get(field) or "")
    if not value:
        return None
    try:
        started = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - started).total_seconds()


def reconcile_job(root: Path | str, job: Dict[str, Any]) -> Dict[str, Any]:
    status = str(job.get("status") or "")
    if status not in {"queued", "running"}:
        return job

    log_tail = _job_log_tail(job)
    if _looks_cancelled(log_tail):
        return update_job_file(
            root,
            job,
            status="cancelled",
            returncode=130,
            finished_at=utc_now_iso(),
            error="Job appears to have been cancelled; log contains cancellation/KeyboardInterrupt text.",
        )

    worker_pid = job.get("worker_pid")
    if worker_pid is not None:
        if process_is_running(worker_pid):
            return job
        return update_job_file(
            root,
            job,
            status="failed",
            returncode=1,
            finished_at=utc_now_iso(),
            error=f"Worker process {worker_pid} is no longer running.",
        )

    if status == "queued":
        age = _job_age_seconds(job, "created_at")
        if age is not None and age > 120:
            return update_job_file(
                root,
                job,
                status="failed",
                returncode=1,
                finished_at=utc_now_iso(),
                error="Queued job was never started by a dashboard worker.",
            )

    return job


def _job_run_name(job: Dict[str, Any]) -> str:
    command = [str(part) for part in (job.get("command") or [])]
    lowered = [part.lower() for part in command]
    try:
        app_index = next(i for i, part in enumerate(lowered) if part.endswith("app.py"))
    except StopIteration:
        return ""
    parts = lowered[app_index + 1 :]
    if len(parts) >= 2 and parts[0] == "update":
        return "update_" + parts[1].replace("-", "_")
    if len(parts) >= 2 and parts[0] == "analyze":
        return "analyze_" + parts[1].replace("-", "_")
    if len(parts) >= 2 and parts[0] == "dashboard":
        return "dashboard_" + parts[1].replace("-", "_")
    return ""


def _timestamp_distance_seconds(left: Any, right: Any) -> Optional[float]:
    try:
        left_dt = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
        right_dt = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
    except ValueError:
        return None
    return abs((left_dt - right_dt).total_seconds())


def _raw_jobs(root: Path | str) -> List[Dict[str, Any]]:
    path = jobs_dir(root)
    if not path.exists():
        return []
    jobs: List[Dict[str, Any]] = []
    for job_path in path.glob("*.json"):
        data = read_json(job_path)
        if data:
            jobs.append(data)
    return jobs


def reconcile_run(root: Path | str, run: Dict[str, Any]) -> Dict[str, Any]:
    if str(run.get("status") or "") != "running":
        return run

    run_name = str(run.get("name") or "")
    run_started = run.get("started_at")
    candidates: List[Dict[str, Any]] = []
    for raw_job in _raw_jobs(root):
        job = reconcile_job(root, raw_job)
        if _job_run_name(job) != run_name:
            continue
        distance = _timestamp_distance_seconds(run_started, job.get("created_at"))
        if distance is None or distance > 300:
            continue
        candidates.append(job)

    terminal = [job for job in candidates if job.get("status") in {"success", "failed", "cancelled"}]
    if not terminal:
        return run

    terminal.sort(key=lambda job: str(job.get("finished_at") or job.get("created_at") or ""), reverse=True)
    job = terminal[0]
    updated = dict(run)
    updated.update(
        {
            "status": job.get("status"),
            "finished_at": job.get("finished_at") or utc_now_iso(),
            "returncode": job.get("returncode"),
            "error": job.get("error", ""),
            "metadata": {
                **dict(run.get("metadata") or {}),
                "reconciled_from_job_id": job.get("id"),
            },
        }
    )
    started_at = str(updated.get("started_at") or "")
    finished_at = str(updated.get("finished_at") or "")
    try:
        updated["duration_seconds"] = (
            datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            - datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        updated["duration_seconds"] = None

    run_id = str(updated.get("id") or "")
    if run_id:
        atomic_write_json(run_path(root, run_id), updated)
    atomic_write_json(latest_run_path(root, str(updated.get("name") or "unknown")), updated)
    return updated


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_path(root: Path | str, run_id: str) -> Path:
    return run_state_dir(root) / "runs" / f"{run_id}.json"


def latest_run_path(root: Path | str, name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return run_state_dir(root) / "latest" / f"{safe_name}.json"


def start_run(
    root: Path | str,
    *,
    name: str,
    kind: str,
    command: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_id = f"{utc_now_compact()}-{name}-{uuid.uuid4().hex[:8]}"
    payload: Dict[str, Any] = {
        "id": run_id,
        "name": name,
        "kind": kind,
        "status": "running",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "duration_seconds": None,
        "returncode": None,
        "command": [str(part) for part in command],
        "command_text": command_text(command),
        "metadata": dict(metadata or {}),
    }
    atomic_write_json(run_path(root, run_id), payload)
    atomic_write_json(latest_run_path(root, name), payload)
    return payload


def finish_run(
    root: Path | str,
    run: Dict[str, Any],
    *,
    returncode: int,
    error: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started_at = str(run.get("started_at") or "")
    duration_seconds: Optional[float] = None
    if started_at:
        try:
            started = datetime.fromisoformat(started_at)
            duration_seconds = (datetime.now(timezone.utc) - started).total_seconds()
        except ValueError:
            duration_seconds = None

    merged_metadata = dict(run.get("metadata") or {})
    merged_metadata.update(metadata or {})
    status = "success" if returncode == 0 else "failed"
    if returncode in {130, -2, -1073741510}:
        status = "cancelled"

    run.update(
        {
            "status": status,
            "finished_at": utc_now_iso(),
            "duration_seconds": duration_seconds,
            "returncode": returncode,
            "error": error,
            "metadata": merged_metadata,
        }
    )
    run_id = str(run.get("id") or "")
    if run_id:
        atomic_write_json(run_path(root, run_id), run)
    atomic_write_json(latest_run_path(root, str(run.get("name") or "unknown")), run)
    return run


@contextmanager
def recorded_run(
    root: Path | str,
    *,
    name: str,
    kind: str,
    command: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    run = start_run(root, name=name, kind=kind, command=command, metadata=metadata)
    try:
        yield run
    except Exception as exc:
        finish_run(root, run, returncode=1, error=str(exc))
        raise


def list_latest_runs(root: Path | str = ".") -> List[Dict[str, Any]]:
    latest_dir = run_state_dir(root) / "latest"
    if not latest_dir.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for path in sorted(latest_dir.glob("*.json")):
        data = read_json(path)
        if data:
            runs.append(reconcile_run(root, data))
    runs.sort(key=lambda row: str(row.get("started_at") or ""), reverse=True)
    return runs


def load_watermarks(data_dir: Path | str = "data") -> Dict[str, Any]:
    return read_json(data_state_dir(data_dir) / "watermarks.json", default={})


def save_watermarks(data_dir: Path | str, watermarks: Dict[str, Any]) -> None:
    atomic_write_json(data_state_dir(data_dir) / "watermarks.json", watermarks)


def update_watermark(
    data_dir: Path | str,
    namespace: str,
    key: str,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    watermarks = load_watermarks(data_dir)
    namespace_values = watermarks.setdefault(namespace, {})
    existing = namespace_values.get(key)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(values)
    existing["updated_at"] = utc_now_iso()
    namespace_values[key] = existing
    save_watermarks(data_dir, watermarks)
    return existing


def _file_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 2)


def build_inventory(
    data_dir: Path | str = "data",
    report_dir: Path | str = "reports",
    *,
    root: Path | str = ".",
    cache_seconds: int = 60,
) -> Dict[str, Any]:
    cache_path = run_state_dir(root) / "inventory.json"
    cached = read_json(cache_path)
    generated_at = cached.get("generated_at")
    if generated_at:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(str(generated_at))).total_seconds()
            if age < cache_seconds:
                return cached
        except ValueError:
            pass

    data_path = Path(data_dir)
    reports_path = Path(report_dir)
    sqlite_counts: Dict[str, int] = {}
    try:
        from sqlite_store import connect, count_market_rows, default_db_path, sqlite_database_available

        if sqlite_database_available(data_path):
            conn = connect(default_db_path(data_path))
            try:
                def count(sql: str) -> int:
                    try:
                        return int(conn.execute(sql).fetchone()[0])
                    except Exception:
                        return 0

                sqlite_counts = {
                    "user_profiles": count("SELECT COUNT(*) FROM user_profiles"),
                    "indexed_user_trades": count("SELECT COUNT(*) FROM indexed_user_trades"),
                    "distinct_users": count(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT user_key FROM user_profiles
                            UNION SELECT user_key FROM user_metrics
                            UNION SELECT user_key FROM user_trades
                            UNION SELECT user_key FROM user_aliases
                        )
                        """
                    ),
                    "user_trades": count("SELECT COUNT(*) FROM user_trades"),
                    "market_current": count("SELECT COUNT(*) FROM market_current"),
                    "indexed_market_trades": count("SELECT COUNT(*) FROM indexed_market_trades"),
                    "market_trades": count("SELECT COUNT(*) FROM market_trades"),
                    "indexed_price_history": count("SELECT COUNT(*) FROM indexed_price_history"),
                    "price_history": count("SELECT COUNT(*) FROM price_history"),
                    "price_history_assets": count("SELECT COUNT(DISTINCT asset) FROM price_history"),
                    "user_metrics": count("SELECT COUNT(*) FROM user_metrics"),
                    "scanner_candidate_users": count("SELECT COUNT(*) FROM scanner_candidate_users"),
                }
                if not sqlite_counts["market_current"]:
                    sqlite_counts["market_current"] = count_market_rows(data_path)
            finally:
                conn.close()
    except Exception:
        sqlite_counts = {}

    payload = {
        "generated_at": utc_now_iso(),
        "data_dir": str(data_path),
        "reports_dir": str(reports_path),
        "user_count": sqlite_counts.get("distinct_users")
        or sqlite_counts.get("user_profiles")
        or sqlite_counts.get("indexed_user_trades")
        or 0,
        "user_trade_rows": sqlite_counts.get("user_trades") or 0,
        "market_metadata_rows": sqlite_counts.get("market_current") or 0,
        "price_history_assets": sqlite_counts.get("price_history_assets") or sqlite_counts.get("indexed_price_history") or 0,
        "market_trade_sets": sqlite_counts.get("indexed_market_trades") or 0,
        "sqlite": sqlite_counts,
        "reports": {
            "user_dashboard_html_mb": _file_mb(reports_path / "dashboard.html"),
            "market_dashboard_html_mb": _file_mb(reports_path / "market_scanner_dashboard.html"),
        },
    }
    atomic_write_json(cache_path, payload)
    return payload


def create_job(
    root: Path | str,
    *,
    label: str,
    command: Sequence[str],
    cwd: Path | str = ".",
) -> Dict[str, Any]:
    job_id = f"{utc_now_compact()}-{uuid.uuid4().hex[:8]}"
    job = {
        "id": job_id,
        "label": label,
        "status": "queued",
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "command": [str(part) for part in command],
        "command_text": command_text(command),
        "cwd": str(cwd),
        "log_path": str(jobs_dir(root) / f"{job_id}.log"),
    }
    atomic_write_json(jobs_dir(root) / f"{job_id}.json", job)
    return job


def update_job(root: Path | str, job: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    job.update(updates)
    atomic_write_json(jobs_dir(root) / f"{job['id']}.json", job)
    return job


def update_job_file(root: Path | str, job: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    return update_job(root, job, **updates)


def list_jobs(root: Path | str = ".", limit: int = 20) -> List[Dict[str, Any]]:
    path = jobs_dir(root)
    if not path.exists():
        return []
    jobs: List[Dict[str, Any]] = []
    for job_path in path.glob("*.json"):
        data = read_json(job_path)
        if data:
            jobs.append(reconcile_job(root, data))
    jobs.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return jobs[:limit]


def has_running_jobs(root: Path | str = ".") -> bool:
    return any(job.get("status") in {"queued", "running"} for job in list_jobs(root, limit=100))


def default_python_command(script_name: str, *args: str) -> List[str]:
    return [sys.executable, str(Path(script_name)), *args]


def extend_command(command: List[str], options: Iterable[str]) -> List[str]:
    command.extend(str(item) for item in options if str(item))
    return command
