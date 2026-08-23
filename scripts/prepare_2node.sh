#!/usr/bin/env bash
# Select completed rank-0 baseline and create exact paired DDP config.
set -euo pipefail

uv run python - <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

rows = []
with open("logs/academic_summary.csv", newline="") as fh:
    rows = list(csv.DictReader(fh))
candidates = []
for row in rows:
    if row.get("world_size") != "1":
        continue
    tag = row.get("run_tag", "")
    config = Path("logs") / tag / "config.json"
    metadata = Path("logs") / tag / "metadata.json"
    if config.exists() and metadata.exists() and row.get("total_samples"):
        candidates.append((tag, row, config, metadata))
if not candidates:
    raise SystemExit("No completed academic single-node baseline candidates found")
for index, (tag, row, _, _) in enumerate(candidates, 1):
    print(f"{index}. {tag} samples={row.get('total_samples')} wall_s={row.get('wall_clock_s')}")
choice = os.environ.get("PREPARE_CHOICE")
if choice is None:
    with open("/dev/tty") as tty:
        print("Select baseline [1]: ", end="", flush=True)
        choice = tty.readline().strip()
choice = choice or "1"
try:
    tag, row, config_path, metadata_path = candidates[int(choice) - 1]
except (ValueError, IndexError):
    raise SystemExit("Invalid baseline selection")
config = json.loads(config_path.read_text())
expected = {"baseline_1k": "2node_1k", "baseline_long": "2node_long"}
paired = expected.get(tag)
if paired is None:
    raise SystemExit(f"Baseline tag {tag} is not an approved academic campaign tag")
config["logging"]["run_tag"] = paired
config["dataset"]["subset_size"] = int(config["dataset"]["subset_size"])
out = Path("configs/resolved")
out.mkdir(parents=True, exist_ok=True)
path = out / f"{paired}.yaml"
import yaml
path.write_text(yaml.safe_dump(config, sort_keys=False))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"resolved config: {path}")
print(f"sha256: {digest}")
print("sync to B:")
print("scp -r configs/resolved/ james@192.168.50.2:~/lora_zero3_demo/configs/")
print(f"A: scripts/launch_rank0.sh --config configs/resolved/{paired}.yaml")
print(f"B: scripts/launch_rank1.sh --config configs/resolved/{paired}.yaml")
PY
