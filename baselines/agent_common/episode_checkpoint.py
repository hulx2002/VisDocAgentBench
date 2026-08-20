#!/usr/bin/env python3
"""Atomic, opt-in partial-episode checkpoints for the public agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


CHECKPOINT_SCHEMA_VERSION = "visdocagentbench_partial_episode_v2"
RECOVERABLE_EPISODE_STATUSES = frozenset(
    {
        "planner_retry_exhausted",
        "step_wall_timeout",
        "forced_submit_retry_exhausted",
        "forced_submit_timeout",
    }
)


class EpisodeCheckpointError(ValueError):
    """Raised when a checkpoint cannot be restored without changing an episode."""


@dataclass
class EpisodeProgress:
    next_step: int = 1
    model_outputs: list[dict[str, Any]] = field(default_factory=list)
    repair_used: bool = False
    pending_repair: str = ""
    transport_failures: list[dict[str, Any]] = field(default_factory=list)
    resume_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_step": self.next_step,
            "model_outputs": self.model_outputs,
            "repair_used": self.repair_used,
            "pending_repair": self.pending_repair,
            "transport_failures": self.transport_failures,
            "resume_count": self.resume_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeProgress":
        if not isinstance(value, dict):
            raise EpisodeCheckpointError("checkpoint progress must be an object")
        next_step = int(value.get("next_step", 0))
        model_outputs = value.get("model_outputs", [])
        transport_failures = value.get("transport_failures", [])
        if next_step < 1:
            raise EpisodeCheckpointError("checkpoint next_step must be positive")
        if not isinstance(model_outputs, list):
            raise EpisodeCheckpointError("checkpoint model_outputs must be a list")
        if not isinstance(transport_failures, list):
            raise EpisodeCheckpointError(
                "checkpoint transport_failures must be a list"
            )
        return cls(
            next_step=next_step,
            model_outputs=model_outputs,
            repair_used=bool(value.get("repair_used", False)),
            pending_repair=str(value.get("pending_repair", "")),
            transport_failures=transport_failures,
            resume_count=int(value.get("resume_count", 0)),
        )


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=32)
def _sha256_file(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Return a content identity suitable for strict resume fingerprints."""
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": _sha256_file(str(resolved), stat.st_size, stat.st_mtime_ns),
    }


def file_collection_identity(
    files: Iterable[tuple[str, Path]],
) -> dict[str, Any]:
    """Return one compact identity for a labeled collection of files."""
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for label, path in sorted(files, key=lambda item: item[0]):
        identity = file_identity(path)
        total_bytes += int(identity["size"])
        records.append({"label": str(label), **identity})
    return {
        "num_files": len(records),
        "total_bytes": total_bytes,
        "sha256": canonical_digest(records),
    }


def directory_identity(path: Path) -> dict[str, Any]:
    """Return a content identity for a local model or processor directory."""
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_file():
        return {"path": str(resolved), "exists": True, **file_identity(resolved)}
    files = [
        (child.relative_to(resolved).as_posix(), child)
        for child in resolved.rglob("*")
        if child.is_file()
    ]
    return {
        "path": str(resolved),
        "exists": True,
        **file_collection_identity(files),
    }


def checkpoint_path(checkpoint_dir: Path, query_id: str) -> Path:
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id).strip("._")
    if not safe_prefix:
        safe_prefix = "query"
    safe_prefix = safe_prefix[:80]
    return checkpoint_dir / f"{safe_prefix}.{canonical_digest(query_id)[:16]}.json"


def observations_from_tool_trace(
    tool_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, item in enumerate(tool_trace):
        if not isinstance(item, dict):
            raise EpisodeCheckpointError(
                f"tool_trace[{index}] must be an object"
            )
        tool_name = str(item.get("tool_name", ""))
        if not tool_name:
            raise EpisodeCheckpointError(
                f"tool_trace[{index}] has no tool_name"
            )
        observations.append(
            {
                "tool": tool_name,
                "arguments": item.get("arguments", {}),
                "observation": item.get("observation", {}),
                "artifacts": item.get("artifacts", []),
            }
        )
    return observations


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_episode_checkpoint(
    path: Path,
    *,
    fingerprint: dict[str, Any],
    progress: EpisodeProgress,
    runtime_state: dict[str, Any] | None,
    legacy_runtime_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "fingerprint_sha256": canonical_digest(fingerprint),
        "progress": progress.to_dict(),
        "runtime_state": runtime_state,
        "legacy_runtime_state": legacy_runtime_state,
    }
    _atomic_write_json(path, payload)


def load_episode_checkpoint(
    path: Path,
    *,
    expected_fingerprint: dict[str, Any],
    max_steps: int,
) -> tuple[EpisodeProgress, dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeCheckpointError(
            f"unable to read episode checkpoint {path}: {exc}"
        ) from exc
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise EpisodeCheckpointError(
            f"unsupported checkpoint schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    stored_fingerprint = payload.get("fingerprint")
    expected_digest = canonical_digest(expected_fingerprint)
    if (
        not isinstance(stored_fingerprint, dict)
        or payload.get("fingerprint_sha256") != canonical_digest(stored_fingerprint)
        or payload.get("fingerprint_sha256") != expected_digest
    ):
        raise EpisodeCheckpointError(
            f"checkpoint fingerprint does not match the current episode: {path}"
        )
    progress = EpisodeProgress.from_dict(payload.get("progress", {}))
    if progress.next_step > max_steps + 1:
        raise EpisodeCheckpointError(
            f"checkpoint next_step={progress.next_step} exceeds max_steps+1={max_steps + 1}"
        )
    runtime_state = payload.get("runtime_state")
    if runtime_state is not None and not isinstance(runtime_state, dict):
        raise EpisodeCheckpointError("checkpoint runtime_state must be an object")
    legacy_runtime_state = payload.get("legacy_runtime_state")
    if legacy_runtime_state is not None and not isinstance(
        legacy_runtime_state, dict
    ):
        raise EpisodeCheckpointError(
            "checkpoint legacy_runtime_state must be an object"
        )
    if runtime_state is None and legacy_runtime_state is None:
        raise EpisodeCheckpointError(
            "checkpoint contains neither runtime_state nor legacy_runtime_state"
        )
    return progress, runtime_state, legacy_runtime_state


def remove_episode_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def recoverable_progress_from_prediction(
    row: dict[str, Any],
    *,
    max_steps: int,
    enabled_tools: frozenset[str],
) -> tuple[EpisodeProgress, dict[str, Any]]:
    status = str(row.get("status", ""))
    if status not in RECOVERABLE_EPISODE_STATUSES:
        raise EpisodeCheckpointError(
            f"prediction status {status!r} is not partial-resume eligible"
        )
    model_outputs = row.get("model_outputs", [])
    tool_trace = row.get("tool_trace", [])
    if not isinstance(model_outputs, list) or not model_outputs:
        raise EpisodeCheckpointError("partial prediction has no model_outputs")
    if not isinstance(tool_trace, list):
        raise EpisodeCheckpointError("partial prediction tool_trace must be a list")
    failure = model_outputs[-1]
    if not isinstance(failure, dict):
        raise EpisodeCheckpointError("partial prediction failure output is invalid")
    next_step = int(failure.get("step", 0))
    if next_step < 1 or next_step > max_steps + 1:
        raise EpisodeCheckpointError(
            f"partial prediction failure step {next_step} is outside the episode budget"
        )
    completed_outputs = model_outputs[:-1]
    if len(completed_outputs) != len(tool_trace):
        raise EpisodeCheckpointError(
            "legacy partial prediction contains a repair/action pattern that cannot "
            "be restored exactly; start this query from step 1"
        )
    for index, (output, trace_item) in enumerate(
        zip(completed_outputs, tool_trace, strict=True)
    ):
        action = output.get("action", {}) if isinstance(output, dict) else {}
        action_name = str(action.get("action", "")) if isinstance(action, dict) else ""
        tool_name = (
            str(trace_item.get("tool_name", ""))
            if isinstance(trace_item, dict)
            else ""
        )
        if action_name != tool_name or action_name not in enabled_tools:
            raise EpisodeCheckpointError(
                f"legacy partial prediction step {index + 1} cannot be mapped "
                "one-to-one onto the current tool trace"
            )
        artifacts = trace_item.get("artifacts", [])
        if isinstance(artifacts, list) and any(
            str(handle).startswith("crop_") for handle in artifacts
        ):
            raise EpisodeCheckpointError(
                "legacy partial prediction contains crop artifacts without "
                "versioned artifact paths; start this query from step 1"
            )
    error = str(failure.get("error", ""))
    progress = EpisodeProgress(
        next_step=next_step,
        model_outputs=completed_outputs,
        repair_used=False,
        pending_repair="",
        transport_failures=[
            {
                "status": status,
                "step": next_step,
                "phase": str(failure.get("phase", "")),
                "attempts": int(failure.get("attempts", 0)),
                "error": error,
                "source": "legacy_prediction_bootstrap",
            }
        ],
        resume_count=0,
    )
    usage = row.get("tool_usage", {})
    if not isinstance(usage, dict):
        raise EpisodeCheckpointError("partial prediction tool_usage must be an object")
    return progress, {"tool_trace": tool_trace, "tool_usage": usage}
