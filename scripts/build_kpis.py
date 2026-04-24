#!/usr/bin/env python3
"""Roll up the session-trajectory dataset into daily KPIs.

Reads the source dataset (one JSONL file per session, path
``sessions/YYYY-MM-DD/<session_id>.jsonl``) and writes one daily CSV to a
target dataset at ``daily/YYYY-MM-DD.csv``. Designed to run as a simple cron
via GH Actions — no framework, pandas-only, idempotent per day.

Metrics computed (one row per day):
    sessions, turns
    tokens_{prompt,completion,cache_read,cache_creation}
    cost_usd
    cache_hit_ratio   — cache_read / (cache_read + prompt)
    llm_calls
    tool_success_rate — tool_output success=True / total tool_output
    regenerate_rate   — turns that followed an undo / total turns
    time_to_first_action_s_p50 — from session_start to first tool_call
    failure_rate      — sessions that ended with an `error` event
    thumbs_up, thumbs_down  — counts from `feedback` events
    hf_jobs_submitted, hf_jobs_succeeded
    gpu_hours_by_flavor   — dict serialised as JSON string

Usage::

    python scripts/build_kpis.py \\
        --source akseljoonas/hf-agent-sessions \\
        --target akseljoonas/hf-agent-kpis \\
        --days 7                      # rolls up the last 7 days

Env:
    HF_TOKEN  (or HF_KPI_WRITE_TOKEN) — write access to target dataset.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("build_kpis")

# Rough gpu-hour pricing for hf_jobs flavor strings. Keep conservative; used
# only to compute gpu-hours (not dollars) — wall_time_s * flavor_gpu_count.
_FLAVOR_GPU_COUNT = {
    "cpu-basic": 0, "cpu-upgrade": 0,
    "t4-small": 1, "t4-medium": 1,
    "l4x1": 1, "l4x4": 4,
    "l40sx1": 1, "l40sx4": 4, "l40sx8": 8,
    "a10g-small": 1, "a10g-large": 1, "a10g-largex2": 2, "a10g-largex4": 4,
    "a100-large": 1, "a100x2": 2, "a100x4": 4, "a100x8": 8,
    "h100": 1, "h100x8": 8,
}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    return float(values[f] + (values[c] - values[f]) * (k - f))


def _iter_session_files(api, repo_id: str, day: date, token: str) -> Iterable[str]:
    """Yield repo-relative paths for all sessions on a given day."""
    prefix = f"sessions/{day.isoformat()}/"
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception as e:
        logger.warning("list_repo_files(%s) failed: %s", repo_id, e)
        return []
    return [f for f in files if f.startswith(prefix) and f.endswith(".jsonl")]


def _download_session(api, repo_id: str, path: str, token: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename=path, repo_type="dataset", token=token,
        )
    except Exception as e:
        logger.warning("hf_hub_download(%s) failed: %s", path, e)
        return None
    try:
        with open(local, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        row = json.loads(line)
        # Session uploader stores messages/events as JSON strings — unpack.
        if isinstance(row.get("messages"), str):
            row["messages"] = json.loads(row["messages"])
        if isinstance(row.get("events"), str):
            row["events"] = json.loads(row["events"])
        return row
    except Exception as e:
        logger.warning("parse(%s) failed: %s", path, e)
        return None


def _session_metrics(session: dict) -> dict:
    """Reduce a single session trajectory to its KPI contributions."""
    # Pre-seed every numeric key so downstream aggregation can sum without
    # having to special-case empty sessions.
    out: dict = {
        "sessions": 0, "turns": 0, "llm_calls": 0,
        "tokens_prompt": 0, "tokens_completion": 0,
        "tokens_cache_read": 0, "tokens_cache_creation": 0,
        "cost_usd": 0.0,
        "tool_calls_total": 0, "tool_calls_success": 0,
        "failures": 0, "regenerate_sessions": 0,
        "thumbs_up": 0, "thumbs_down": 0,
        "hf_jobs_submitted": 0, "hf_jobs_succeeded": 0,
        "first_tool_s": -1,
    }
    events = session.get("events") or []
    messages = session.get("messages") or []

    # Turn count: count user messages (same proxy as Session.turn_count).
    turn_count = sum(1 for m in messages if m.get("role") == "user")
    out["turns"] = turn_count
    out["sessions"] = 1

    tool_success = 0
    tool_total = 0
    had_error = False
    had_undo = False
    first_tool_ts = None
    session_start = session.get("session_start_time")
    gpu_hours_by_flavor: dict[str, float] = defaultdict(float)
    jobs_submitted = 0
    jobs_succeeded = 0
    thumbs_up = 0
    thumbs_down = 0

    def _parse_ts(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    start_dt = _parse_ts(session_start)

    for ev in events:
        et = ev.get("event_type")
        data = ev.get("data") or {}
        ts = _parse_ts(ev.get("timestamp"))

        if et == "llm_call":
            out["llm_calls"] += 1
            out["tokens_prompt"] += int(data.get("prompt_tokens") or 0)
            out["tokens_completion"] += int(data.get("completion_tokens") or 0)
            out["tokens_cache_read"] += int(data.get("cache_read_tokens") or 0)
            out["tokens_cache_creation"] += int(data.get("cache_creation_tokens") or 0)
            out["cost_usd"] += float(data.get("cost_usd") or 0.0)

        elif et == "tool_output":
            tool_total += 1
            if data.get("success"):
                tool_success += 1
            if first_tool_ts is None and ts is not None and start_dt is not None:
                first_tool_ts = (ts - start_dt).total_seconds()

        elif et == "tool_call":
            if first_tool_ts is None and ts is not None and start_dt is not None:
                first_tool_ts = (ts - start_dt).total_seconds()

        elif et == "error":
            had_error = True

        elif et == "undo_complete":
            had_undo = True

        elif et == "feedback":
            rating = data.get("rating")
            if rating == "up":
                thumbs_up += 1
            elif rating == "down":
                thumbs_down += 1

        elif et == "hf_job_submit":
            jobs_submitted += 1

        elif et == "hf_job_complete":
            flavor = data.get("flavor") or "unknown"
            status = (data.get("final_status") or "").lower()
            wall = float(data.get("wall_time_s") or 0.0)
            gpus = _FLAVOR_GPU_COUNT.get(flavor, 0)
            gpu_hours_by_flavor[flavor] += wall * gpus / 3600.0
            if status in ("completed", "succeeded", "success"):
                jobs_succeeded += 1

    out["tool_calls_total"] = tool_total
    out["tool_calls_success"] = tool_success
    out["failures"] = 1 if had_error else 0
    out["regenerate_sessions"] = 1 if had_undo else 0
    out["thumbs_up"] = thumbs_up
    out["thumbs_down"] = thumbs_down
    out["hf_jobs_submitted"] = jobs_submitted
    out["hf_jobs_succeeded"] = jobs_succeeded
    out["first_tool_s"] = first_tool_ts if first_tool_ts is not None else -1
    out["_gpu_hours_by_flavor"] = dict(gpu_hours_by_flavor)  # aggregated later
    out["_user"] = session.get("user_id") or session.get("session_id")  # fallback
    return dict(out)


def _aggregate_day(per_session: list[dict]) -> dict:
    """Collapse a day's worth of session rollups into the final KPI row."""
    ttfa_values = [s["first_tool_s"] for s in per_session if s.get("first_tool_s", -1) >= 0]
    gpu_hours = defaultdict(float)
    for s in per_session:
        for f, h in (s.get("_gpu_hours_by_flavor") or {}).items():
            gpu_hours[f] += h

    total_sessions = sum(s["sessions"] for s in per_session)
    total_turns = sum(s["turns"] for s in per_session)
    tokens_prompt = sum(s["tokens_prompt"] for s in per_session)
    tokens_cache_read = sum(s["tokens_cache_read"] for s in per_session)
    tool_total = sum(s["tool_calls_total"] for s in per_session)
    tool_success = sum(s["tool_calls_success"] for s in per_session)

    unique_users = {s.get("_user") for s in per_session if s.get("_user")}

    return {
        "sessions": total_sessions,
        "users": len(unique_users),
        "turns": total_turns,
        "llm_calls": int(sum(s["llm_calls"] for s in per_session)),
        "tokens_prompt": int(tokens_prompt),
        "tokens_completion": int(sum(s["tokens_completion"] for s in per_session)),
        "tokens_cache_read": int(tokens_cache_read),
        "tokens_cache_creation": int(sum(s["tokens_cache_creation"] for s in per_session)),
        "cost_usd": round(sum(s["cost_usd"] for s in per_session), 4),
        "cache_hit_ratio": round(
            tokens_cache_read / (tokens_cache_read + tokens_prompt), 4
        ) if (tokens_cache_read + tokens_prompt) > 0 else 0.0,
        "tool_success_rate": round(tool_success / tool_total, 4) if tool_total > 0 else 0.0,
        "failure_rate": round(
            sum(s["failures"] for s in per_session) / total_sessions, 4
        ) if total_sessions > 0 else 0.0,
        "regenerate_rate": round(
            sum(s["regenerate_sessions"] for s in per_session) / total_sessions, 4
        ) if total_sessions > 0 else 0.0,
        "time_to_first_action_s_p50": round(_percentile(ttfa_values, 0.5), 2),
        "time_to_first_action_s_p95": round(_percentile(ttfa_values, 0.95), 2),
        "thumbs_up": int(sum(s["thumbs_up"] for s in per_session)),
        "thumbs_down": int(sum(s["thumbs_down"] for s in per_session)),
        "hf_jobs_submitted": int(sum(s["hf_jobs_submitted"] for s in per_session)),
        "hf_jobs_succeeded": int(sum(s["hf_jobs_succeeded"] for s in per_session)),
        "gpu_hours_by_flavor_json": json.dumps(dict(gpu_hours), sort_keys=True),
    }


def _write_daily_csv(row: dict, day: date, target_repo: str, token: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    columns = list(row.keys())
    buf = io.StringIO()
    buf.write(",".join(["date", *columns]) + "\n")
    buf.write(",".join([day.isoformat(), *[_csv_cell(row[c]) for c in columns]]) + "\n")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write(buf.getvalue())
        tmp_path = tmp.name

    try:
        api.create_repo(
            repo_id=target_repo, repo_type="dataset", exist_ok=True, token=token,
        )
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"daily/{day.isoformat()}.csv",
            repo_id=target_repo,
            repo_type="dataset",
            token=token,
            commit_message=f"KPIs for {day.isoformat()}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _csv_cell(v: Any) -> str:
    s = str(v)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def run_for_day(api, source_repo: str, target_repo: str, day: date, token: str) -> dict:
    paths = _iter_session_files(api, source_repo, day, token)
    per_session: list[dict] = []
    for path in paths:
        sess = _download_session(api, source_repo, path, token)
        if not sess:
            continue
        per_session.append(_session_metrics(sess))

    if not per_session:
        logger.info("No sessions found for %s — skipping", day)
        return {}

    row = _aggregate_day(per_session)
    _write_daily_csv(row, day, target_repo, token)
    logger.info("Wrote KPIs for %s: %s", day, row)
    return row


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="akseljoonas/hf-agent-sessions")
    ap.add_argument("--target", default="akseljoonas/hf-agent-kpis")
    ap.add_argument(
        "--days", type=int, default=1,
        help="Number of trailing days to roll up (default: 1 = yesterday).",
    )
    ap.add_argument(
        "--date", type=str, default=None,
        help="Single YYYY-MM-DD to roll up; overrides --days.",
    )
    args = ap.parse_args(argv)

    token = os.environ.get("HF_KPI_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_KPI_WRITE_TOKEN or HF_TOKEN must be set.")
        return 1

    from huggingface_hub import HfApi
    api = HfApi()

    if args.date:
        target_days = [date.fromisoformat(args.date)]
    else:
        today = datetime.now(timezone.utc).date()
        target_days = [today - timedelta(days=i) for i in range(1, args.days + 1)]

    for day in target_days:
        run_for_day(api, args.source, args.target, day, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
