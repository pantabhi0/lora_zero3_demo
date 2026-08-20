"""Phase 5 logging/metrics helpers.

Design constraints (see AGENTS.md):
- Per optimizer update: compute time (summed across micro-steps) and comm time.
- Comm timed via DDP comm hook covering the FULL hook duration.
- CUDA events accumulated, synchronized ONCE at end of run — never per-step.
- Per-step loss/step flushed live to CSV; timing rows finalized at end.
- Fixed sample budget is the comparison unit across all runs.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

RUN_COLUMNS = ["step", "loss", "samples_processed"]
TIMING_COLUMNS = ["step", "compute_ms", "comm_ms"]
SUMMARY_COLUMNS = [
    "run_tag",
    "world_size",
    "rank",
    "total_samples",
    "total_updates",
    "wall_clock_s",
    "samples_per_sec",
    "tokens_per_sec",
    "compute_ms_per_update_avg",
    "comm_ms_per_update_avg",
    "comm_pct_of_step",
    "extrapolated_epoch_s",
    "final_loss",
    "loss_criterion_met",
]


def run_csv_path(output_dir: str, run_tag: str, rank: int) -> Path:
    return Path(output_dir) / f"run_{run_tag}_rank{rank}.csv"


def timing_csv_path(output_dir: str, run_tag: str, rank: int) -> Path:
    return Path(output_dir) / f"timing_{run_tag}_rank{rank}.csv"


def summary_csv_path(output_dir: str) -> Path:
    return Path(output_dir) / "summary.csv"


class LiveCSV:
    """Per-step live rows, flushed to disk each row so a TUI can re-read."""

    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(columns))
        self._writer.writeheader()
        self._fh.flush()

    def write_row(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class TimingCollector:
    """Accumulates (start, end) CUDA-event pairs per step; syncs once at end."""

    def __init__(self) -> None:
        self._by_step: Dict[int, List[tuple]] = defaultdict(list)
        self._synced = False

    def record(self, step: int, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        if self._synced:
            raise RuntimeError("record after finalize()")
        self._by_step[step].append((start, end))

    def finalize(self) -> Dict[int, float]:
        """Sync once, then return {step: elapsed_ms}. Idempotent."""
        if not self._synced:
            torch.cuda.synchronize()
            self._synced = True
        out: Dict[int, float] = {}
        for step, pairs in self._by_step.items():
            total_ms = 0.0
            for start, end in pairs:
                total_ms += start.elapsed_time(end)
            out[step] = total_ms
        return out


def make_timing_hook(collector: TimingCollector, step_provider, world_size: int = 1):
    """Return a DDP comm hook timing the FULL all-reduce duration.

    Replicates DDP's default hook semantics (all-reduce then divide by
    world_size) so gradients stay averaged, and times the whole op.
    """

    def hook(state, bucket) -> torch.futures.Future:
        import torch.distributed as dist

        step = step_provider()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fut = dist.all_reduce(
            bucket.buffer(), op=dist.ReduceOp.SUM, async_op=True
        ).get_future()

        def _on_done(future: torch.futures.Future) -> torch.Tensor:
            bucket.buffer().div_(world_size)
            end.record()
            collector.record(step, start, end)
            return bucket.buffer()

        return fut.then(_on_done)

    # DDP._check_comm_hook compares the return annotation against the REAL
    # torch.futures.Future[torch.Tensor]; with `from __future__ import
    # annotations` the def-line annotation stays a string and the check fails.
    # Force the real type object onto the hook.
    hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]

    return hook


def new_cuda_events() -> tuple:
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def samples_per_sec(total_samples: int, wall_clock_s: float) -> float:
    return total_samples / wall_clock_s if wall_clock_s > 0 else 0.0


def tokens_per_sec(total_samples: int, seq_length: int, wall_clock_s: float) -> float:
    return (total_samples * seq_length) / wall_clock_s if wall_clock_s > 0 else 0.0


def comm_pct(compute_ms: float, comm_ms: float) -> float:
    total = compute_ms + comm_ms
    return (comm_ms / total * 100.0) if total > 0 else 0.0


def extrapolated_epoch_s(wall_clock_s: float, fraction_done: float) -> float:
    if fraction_done <= 0:
        return 0.0
    return wall_clock_s / fraction_done


def loss_criterion_met(
    losses: Sequence[float], threshold: float = 0.10, window_fraction: float = 0.10
) -> tuple:
    """P3 acceptance: mean(last window_fraction of updates) >= threshold below
    mean(first window_fraction). Window scales with update count."""
    n = len(losses)
    if n < 2:
        return False, 0.0, 0.0, 0
    window = max(1, int(round(n * window_fraction)))
    first_mean = sum(losses[:window]) / window
    last_mean = sum(losses[n - window :]) / window
    met = first_mean > 0 and (first_mean - last_mean) / first_mean >= threshold
    return met, first_mean, last_mean, window


def curve_matches(
    a: Sequence[float], b: Sequence[float], tol: float = 0.02
) -> tuple:
    """P4 acceptance: per-step relative agreement within tol (1-2% default)."""
    n = min(len(a), len(b))
    if n == 0:
        return False, float("inf")
    max_rel = 0.0
    for i in range(n):
        ref = abs(a[i])
        if ref > 1e-12:
            max_rel = max(max_rel, abs(a[i] - b[i]) / ref)
        elif abs(a[i] - b[i]) > 1e-12:
            max_rel = float("inf")
    return max_rel <= tol, max_rel


def write_timing_csv(
    output_dir: str, run_tag: str, rank: int, compute: Dict[int, float], comm: Dict[int, float]
) -> Path:
    path = timing_csv_path(output_dir, run_tag, rank)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMING_COLUMNS)
        writer.writeheader()
        for step in sorted(set(compute) | set(comm)):
            writer.writerow(
                {
                    "step": step,
                    "compute_ms": round(compute.get(step, 0.0), 3),
                    "comm_ms": round(comm.get(step, 0.0), 3),
                }
            )
    return path


def write_summary_csv(
    output_dir: str,
    row: Dict[str, object],
    summary_path: Optional[Path] = None,
) -> Path:
    path = summary_path or summary_csv_path(output_dir)
    header = SUMMARY_COLUMNS
    file_exists = path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})
    return path


def monotonic_now() -> float:
    return time.monotonic()


def load_env_file(path: str = ".env") -> None:
    """Fill unset env vars from a .env file (KEY=VALUE lines).

    Existing vars are never overridden, so docker --env-file (which exports
    all vars) takes precedence over the repo-local .env.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(
            f"Missing required env var {name}. Fill .env per Phase 0 "
            "('Read this first' in gigabit-lora-demo-plan.md). No auto-detection."
        )
    return val


def require_dist_env() -> Dict[str, str]:
    """Phase 1/4 gate: NCCL_SOCKET_IFNAME + static-IP config, no defaults."""
    load_env_file()
    env = {
        "MASTER_ADDR": _require_env("MASTER_ADDR"),
        "MASTER_PORT": _require_env("MASTER_PORT"),
        "WORLD_SIZE": _require_env("WORLD_SIZE"),
        "RANK": _require_env("RANK"),
        "NCCL_SOCKET_IFNAME": _require_env("NCCL_SOCKET_IFNAME"),
    }
    return env


def load_losses(csv_path: Path) -> List[float]:
    losses: List[float] = []
    if not csv_path.exists():
        return losses
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                losses.append(float(row["loss"]))
            except (ValueError, KeyError):
                continue
    return losses