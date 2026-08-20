"""Phase 6 — live TUI reading Phase 5 CSVs only (no separate logging).

Fixed 1s re-read of the local rank's run CSV (+ timing CSV when present).
Laptop A shows rank 0, laptop B shows rank 1, side-by-side.

Usage:
    uv run python -m src.tui --csv logs/run_baseline_rank0.csv
    uv run python -m src.tui --csv logs/run_2node_rank0.csv --refresh 1.0

Plain-stdout fallback stays available (TUI_RENDER=plain).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="run CSV (e.g. logs/run_2node_rank0.csv)")
    p.add_argument("--refresh", type=float, default=1.0)
    return p.parse_args()


def read_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def timing_path_for(run_csv: Path) -> Path:
    return run_csv.parent / run_csv.name.replace("run_", "timing_")


def build_table(rows: List[dict], timing: List[dict], title: str):
    from rich.table import Table

    table = Table(title=title)
    for col in ("step", "loss", "samples_processed", "compute_ms", "comm_ms"):
        table.add_column(col)
    # Timing CSV steps are 0-based, run CSV steps are 1-based: merge by POSITION.
    for i, r in enumerate(rows[-40:]):
        t = timing[i] if i < len(timing) else {}
        table.add_row(
            str(r.get("step", "")),
            r.get("loss", ""),
            r.get("samples_processed", ""),
            t.get("compute_ms", ""),
            t.get("comm_ms", ""),
        )
    return table


def run_rich(run_csv: Path, timing_csv: Path, refresh: float) -> None:
    from rich.console import Console

    console = Console()
    title = f"TUI — {run_csv.name}"
    try:
        while True:
            rows = read_rows(run_csv)
            timing = read_rows(timing_csv)
            console.clear()
            console.print(build_table(rows, timing, title))
            time.sleep(refresh)
    except KeyboardInterrupt:
        console.print("\nstopped.")


def main() -> None:
    args = parse_args()
    run_csv = Path(args.csv)
    timing_csv = timing_path_for(run_csv)

    if args.refresh <= 0:
        for row in read_rows(run_csv):
            print(row)
        return

    if os.environ.get("TUI_RENDER", "") == "plain":
        while True:
            for row in read_rows(run_csv):
                print(
                    f"{row.get('step','')}\t{row.get('loss','')}"
                    f"\t{row.get('samples_processed','')}",
                    flush=True,
                )
            time.sleep(args.refresh)
        return

    try:
        run_rich(run_csv, timing_csv, args.refresh)
    except ImportError:
        print("rich not available; run with TUI_RENDER=plain", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()