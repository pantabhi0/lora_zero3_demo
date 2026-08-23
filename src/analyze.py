"""Generate static academic figures from completed campaign artifacts.

Usage:
    uv run python -m src.analyze --scale 1k
    uv run python -m src.analyze --scale long
    uv run python -m src.analyze --scale all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CAMPAIGNS = {
    "1k": ("baseline_1k", "2node_1k"),
    "long": ("baseline_long", "2node_long"),
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def require_complete(scale: str) -> tuple[str, str]:
    baseline, distributed = CAMPAIGNS[scale]
    missing = []
    for tag in (baseline, distributed):
        required = [
            Path("logs") / f"run_{tag}_rank0.csv",
            Path("logs") / f"timing_{tag}_rank0.csv",
            Path("logs") / f"validation_{tag}_rank0.csv",
            Path("logs") / f"gpu_{tag}_rank0.csv",
            Path("logs") / tag / "metadata.json",
        ]
        if tag.startswith("2node"):
            required.append(Path("logs") / f"gpu_{tag}_rank1.csv")
        missing.extend(str(path) for path in required if not path.exists())
    if missing:
        raise SystemExit(f"incomplete {scale} campaign; missing: {', '.join(missing)}")
    return baseline, distributed


def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def plot_loss(scale: str, tags: tuple[str, str], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tag, label in zip(tags, ("1-node baseline", "2-node DDP")):
        train = rows(Path("logs") / f"run_{tag}_rank0.csv")
        validation = rows(Path("logs") / f"validation_{tag}_rank0.csv")
        ax.plot([f(r, "tokens_seen") for r in train],
                [f(r, "loss") for r in train], label=f"{label} train")
        ax.plot([f(r, "tokens_seen") for r in validation],
                [f(r, "val_loss") for r in validation], "o--", label=f"{label} validation")
    ax.set(title=f"Training and held-out validation loss: {scale}",
           xlabel="Tokens seen", ylabel="Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"loss_tokens_{scale}.png", dpi=160)
    plt.close(fig)


def validation_by_tokens(scale: str, tags: tuple[str, str], out: Path) -> None:
    """Write explicit token-aligned validation comparison, never step-aligned."""
    validation = {tag: rows(Path("logs") / f"validation_{tag}_rank0.csv") for tag in tags}
    points = sorted({int(f(row, "tokens_seen")) for data in validation.values() for row in data})
    lines = [
        f"# Token-aligned validation comparison: {scale}", "",
        "Validation curves are compared on `tokens_seen`, never raw optimizer step.",
        "", "| tokens_seen | baseline_val_loss | 2node_val_loss |", "| ---: | ---: | ---: |",
    ]
    for token_count in points:
        values = []
        for tag in tags:
            exact = next((f(row, "val_loss") for row in validation[tag]
                          if int(f(row, "tokens_seen")) == token_count), "")
            values.append(exact)
        lines.append(f"| {token_count} | {values[0]} | {values[1]} |")
    (out / f"validation_tokens_{scale}.md").write_text("\n".join(lines) + "\n")


def plot_throughput(scale: str, tags: tuple[str, str], out: Path) -> None:
    import matplotlib.pyplot as plt

    summary = rows(Path("logs") / "academic_summary.csv")
    values = []
    for tag in tags:
        match = next((r for r in summary if r.get("run_tag") == tag), None)
        if match is None:
            raise SystemExit(f"missing academic summary row for {tag}")
        values.append(f(match, "tokens_per_sec"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["1-node", "2-node"], values, color=["#4c78a8", "#f58518"])
    ax.set(title=f"Throughput: {scale}", ylabel="Tokens/sec")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / f"throughput_{scale}.png", dpi=160)
    plt.close(fig)


def plot_gpu(tag: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for rank in (0, 1):
        data = rows(Path("logs") / f"gpu_{tag}_rank{rank}.csv")
        if not data:
            continue
        ax.plot([f(r, "elapsed_s") for r in data],
                [f(r, "utilization_gpu_pct") for r in data], label=f"rank {rank}")
    ax.set(title=f"GPU utilization: {tag}", xlabel="Elapsed seconds", ylabel="GPU utilization (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"gpu_utilization_{tag}.png", dpi=160)
    plt.close(fig)


def plot_breakdown(tag: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    summary = rows(Path("logs") / "academic_summary.csv")
    match = next((r for r in summary if r.get("run_tag") == tag), None)
    if match is None:
        raise SystemExit(f"missing academic summary row for {tag}")
    keys = ["compute_ms_per_update_avg", "comm_ms_per_update_avg",
            "data_ms_per_update_avg", "validation_wall_clock_s", "other_ms_per_update_avg"]
    values = [f(match, key) for key in keys]
    values[3] = values[3] * 1000.0 / max(1.0, f(match, "total_updates"))
    labels = ["compute", "comm", "data", "validation", "other"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = 0.0
    for label, value in zip(labels, values):
        ax.bar([tag], [value], bottom=[bottom], label=label)
        bottom += value
    ax.set(title=f"Time breakdown: {tag}", ylabel="Milliseconds / update")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"breakdown_{tag}.png", dpi=160)
    plt.close(fig)


def plot_speedup(scale: str, tags: tuple[str, str], out: Path) -> None:
    import matplotlib.pyplot as plt

    summary = rows(Path("logs") / "academic_summary.csv")
    selected = {r.get("run_tag"): r for r in summary if r.get("run_tag") in tags}
    speedup = f(selected[tags[0]], "wall_clock_s") / f(selected[tags[1]], "wall_clock_s")
    efficiency = speedup / 2.0
    (out / f"speedup_efficiency_{scale}.md").write_text(
        f"# Speedup and parallel efficiency: {scale}\n\n"
        f"| Metric | Value |\n| --- | ---: |\n"
        f"| Speedup | {speedup:.4f}x |\n| Parallel efficiency | {efficiency:.4f} |\n"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["speedup", "efficiency"], [speedup, efficiency], color=["#54a24b", "#e45756"])
    ax.set(title=f"Speedup and efficiency: {scale}")
    fig.tight_layout()
    fig.savefig(out / f"speedup_efficiency_{scale}.png", dpi=160)
    plt.close(fig)


def plot_quality(scale: str, tags: tuple[str, str], out: Path) -> None:
    import matplotlib.pyplot as plt

    summary = rows(Path("logs") / "academic_summary.csv")
    selected = [r for r in summary if r.get("run_tag") in tags]
    labels = [r.get("run_tag", "") for r in selected]
    values = [f(r, "final_val_loss", math.nan) for r in selected]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#4c78a8", "#f58518"])
    ax.set(title=f"Final held-out validation loss: {scale}", ylabel="Validation loss")
    fig.tight_layout()
    fig.savefig(out / f"final_quality_{scale}.png", dpi=160)
    plt.close(fig)


def plot_comm_stability(tag: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    data = rows(Path("logs") / f"timing_{tag}_rank0.csv")
    values = [f(r, "comm_ms") for r in data]
    window = max(1, min(10, len(values) // 10 or 1))
    rolling = [sum(values[max(0, i - window + 1):i + 1]) /
               len(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(values)), values, alpha=0.25, label="per-update comm")
    ax.plot(range(len(rolling)), rolling, label=f"{window}-step rolling mean")
    ax.set(title=f"Communication stability: {tag}", xlabel="Optimizer update", ylabel="Comm ms")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"comm_stability_{tag}.png", dpi=160)
    plt.close(fig)


def plot_vram(tag: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    summary = rows(Path("logs") / "academic_summary.csv")
    match = next((r for r in summary if r.get("run_tag") == tag), None)
    if match is None:
        return
    values = [f(match, "peak_vram_allocated_mib"), f(match, "peak_vram_reserved_mib")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["allocated", "reserved"], values, color=["#72b7b2", "#eeca3b"])
    ax.set(title=f"Peak VRAM: {tag}", ylabel="MiB")
    fig.tight_layout()
    fig.savefig(out / f"peak_vram_{tag}.png", dpi=160)
    plt.close(fig)


def plot_bandwidth(tag: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    summary = rows(Path("logs") / "academic_summary.csv")
    match = next((r for r in summary if r.get("run_tag") == tag), None)
    if match is None:
        return
    effective = f(match, "effective_bandwidth_mb_s")
    ceiling = 125.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["effective", "1 Gbps ceiling"], [effective, ceiling], color=["#f58518", "#bab0ac"])
    ax.set(title=f"Communication bandwidth estimate: {tag}", ylabel="MB/s")
    fig.tight_layout()
    fig.savefig(out / f"bandwidth_{tag}.png", dpi=160)
    plt.close(fig)


def plot_scale_consistency(out: Path) -> None:
    import matplotlib.pyplot as plt
    summary = rows(Path("logs") / "academic_summary.csv")
    values = []
    for scale, tags in CAMPAIGNS.items():
        selected = {r.get("run_tag"): r for r in summary if r.get("run_tag") in tags}
        if len(selected) == 2:
            values.append((scale, f(selected[tags[0]], "wall_clock_s") / f(selected[tags[1]], "wall_clock_s")))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([x[0] for x in values], [x[1] for x in values], color="#54a24b")
    ax.set(title="Speedup consistency across scales", ylabel="1-node / 2-node speedup")
    ax.axhline(2.0, color="#555", linestyle="--", label="ideal 2x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "speedup_consistency.png", dpi=160)
    plt.close(fig)


def write_tables(scale: str, tags: tuple[str, str], out: Path) -> None:
    summary = rows(Path("logs") / "academic_summary.csv")
    selected = [r for r in summary if r.get("run_tag") in tags]
    speedup = f(selected[0], "wall_clock_s") / f(selected[1], "wall_clock_s")
    table = [f"# Analysis summary: {scale}", "", "| Run | World | Samples | Wall s | Tokens/s | Train loss | Val loss | Val perplexity |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in selected:
        table.append("| {run_tag} | {world_size} | {total_samples} | {wall_clock_s} | {tokens_per_sec} | {final_loss} | {final_val_loss} | {final_val_perplexity} |".format(**{key: row.get(key, "") for key in ("run_tag", "world_size", "total_samples", "wall_clock_s", "tokens_per_sec", "final_loss", "final_val_loss", "final_val_perplexity")}))
    table += ["", f"Speedup: {speedup:.4f}x", f"Parallel efficiency: {speedup / 2:.4f}", ""]
    (out / f"summary_{scale}.md").write_text("\n".join(table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["1k", "long", "all"], required=True)
    args = parser.parse_args()
    scales = CAMPAIGNS if args.scale == "all" else {args.scale: CAMPAIGNS[args.scale]}
    for scale in scales:
        tags = require_complete(scale)
        out = Path("logs") / "analysis" / scale
        out.mkdir(parents=True, exist_ok=True)
        plot_loss(scale, tags, out)
        validation_by_tokens(scale, tags, out)
        plot_throughput(scale, tags, out)
        plot_speedup(scale, tags, out)
        plot_quality(scale, tags, out)
        for tag in tags:
            plot_gpu(tag, out)
            plot_breakdown(tag, out)
            plot_vram(tag, out)
            plot_bandwidth(tag, out)
            if tag.startswith("2node"):
                plot_comm_stability(tag, out)
        write_tables(scale, tags, out)
    if args.scale == "all":
        plot_scale_consistency(Path("logs") / "analysis")


if __name__ == "__main__":
    main()
