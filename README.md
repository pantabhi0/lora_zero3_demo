# HeteroTrain

Two-node fp16 LoRA fine-tuning of `Qwen/Qwen2.5-0.5B` over direct gigabit
Ethernet. One GTX 1650 is compared with two GTX 1650 laptops using plain
PyTorch DDP.

The repository contains the complete four-run academic campaign, static
figures, WebUI, inference page, Docker image workflow, and measured report.

## Academic Results

Academic conclusions use exactly these four runs. Historical `baseline` and
`default` artifacts are excluded.

| Run | Samples | Updates | Wall clock | Tokens/sec | Final val loss | Final val PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_1k` | 1000 | 250 | 3286.125 s | 26.807 | 0.606460 | 1.833927 |
| `2node_1k` | 1000 | 125 | 1668.567 s | 52.795 | 0.572788 | 1.773204 |
| `baseline_long` | 3288 | 822 | 8841.677 s | 31.929 | 0.567638 | 1.764096 |
| `2node_long` | 3288 | 411 | 4475.444 s | 63.080 | 0.565360 | 1.760081 |

| Scale | Speedup | Parallel efficiency | DDP comm share |
| --- | ---: | ---: | ---: |
| 1k | 1.969x | 98.47% | 2.87% |
| Long | 1.976x | 98.78% | 2.93% |

Quality uses held-out response-only validation loss/perplexity. Prompt tokens
are masked with `-100`; historical loss values are not comparable.

Full evaluation: [`report.md`](report.md).

## What This Is

The experiment measures whether two inexpensive, identical GPUs connected by
1 Gbps Ethernet can process a fixed sample budget faster than one GPU.

- Model: `Qwen/Qwen2.5-0.5B` base, not `-Instruct`.
- Training: fp16 LoRA, no quantization, no DeepSpeed, no ZeRO, no FSDP.
- Attention: PyTorch SDPA; FlashAttention-2 is not used because GTX 1650 is
  Turing capability 7.5.
- Network: direct point-to-point Ethernet.
- Launcher: one manually started process per machine.
- Comparison unit: fixed total sample budget, not raw optimizer step count.
- Timing: CUDA events and DDP comm hook are synchronized/finalized only at the
  end of training. New telemetry is independent of this timing path.

The two-node run has twice the global samples per optimizer update:

```text
1-node: 1 GPU × batch 1 × accumulation 4 = 4 samples/update
2-node: 2 GPUs × batch 1 × accumulation 4 = 8 samples/update
```

Therefore a 1000-sample run has 250 single-GPU updates and 125 DDP updates.
Comparisons of loss/validation curves use `tokens_seen`, never raw step index.

## Repository Layout

```text
Dockerfile
pyproject.toml                  uv-managed dependencies
uv.lock                         locked dependency graph
.env.example                    distributed environment template
configs/lora_config.yaml        academic default config
configs/resolved/               paired campaign configs
scripts/build_image.sh          build/save/load Docker image
scripts/launch_rank0.sh         A/rank-0 launcher
scripts/launch_rank1.sh         B/rank-1 launcher
scripts/prepare_2node.sh        select baseline and create paired config
src/hello_world_dist.py         NCCL/network smoke test
src/train_lora.py               training, validation, telemetry, metadata
src/metrics.py                  timing and metric helpers
src/analyze.py                  static PNG analysis generator
src/webui.py                    WebUI entrypoint
src/web/                        WebUI inference/backend/static assets
src/tui.py                      terminal fallback
src/infer.py                    standalone inference REPL
logs/                           run artifacts and generated analysis
hf_cache/                       local Hugging Face cache
report.md                       final academic evaluation
documentation.md                technical deep dive
```

## Requirements

- Two Ubuntu machines, each with one GTX 1650 4 GB GPU.
- NVIDIA driver working on both machines: `nvidia-smi` succeeds.
- Native Docker Engine, not Docker Desktop.
- NVIDIA Container Toolkit registered with Docker.
- `uv` installed on both machines.
- Direct gigabit-capable Ethernet connection.
- Same repository code and same Docker image on both machines.

The measured machines use:

```text
A / rank 0: 192.168.50.1/24, interface eno1
B / rank 1: 192.168.50.2/24, interface enp44s0
```

## First-Time Network Setup

Find Ethernet interfaces:

```bash
ip a
```

Verify link speed on both machines:

```bash
sudo ethtool <interface> | grep Speed
```

Required result:

```text
Speed: 1000Mb/s
```

If a cable or adapter negotiates at 10/100:

```bash
sudo ethtool -s <interface> speed 1000 duplex full autoneg on
```

Assign static addresses. Example netplan for A:

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eno1:
      addresses: [192.168.50.1/24]
```

Example for B:

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp44s0:
      addresses: [192.168.50.2/24]
```

Apply and test both directions:

```bash
sudo netplan apply
ping -c 3 192.168.50.2   # from A
ping -c 3 192.168.50.1   # from B
```

No gateway is required for this direct link.

## Host and Docker Setup

Install `uv` on both machines:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install and configure native Docker plus NVIDIA Container Toolkit according
to the current Ubuntu/NVIDIA instructions. Verify:

```bash
docker info | grep -i runtime
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

The Docker runtime must expose `nvidia`. Docker Desktop is not supported:
its VM networking and GPU passthrough break this experiment.

## Environment

On each machine:

```bash
cp .env.example .env
$EDITOR .env
```

A and B share:

```dotenv
MASTER_ADDR=192.168.50.1
MASTER_PORT=29500
WORLD_SIZE=2
```

A uses:

```dotenv
RANK=0
NCCL_SOCKET_IFNAME=eno1
```

B uses:

```dotenv
RANK=1
NCCL_SOCKET_IFNAME=enp44s0
```

The code does not auto-detect interfaces or addresses. Bare-metal execution
auto-loads unset values from the repository `.env`; Docker receives values
through `--env-file .env`. Existing environment variables win.

## Validate NCCL Before Training

Run both commands at approximately the same time:

```bash
# A
RANK=0 uv run python -m src.hello_world_dist

# B
RANK=1 uv run python -m src.hello_world_dist
```

Both must print:

```text
all_reduce ok: tensor[0]=3.0
clean exit
```

If initialization hangs, fix `NCCL_SOCKET_IFNAME`, the static addresses,
firewall, cable, or rendezvous port. Do not add retry logic.

## Build and Distribute the Image

Build once on A:

```bash
uv sync
scripts/build_image.sh --save
```

Transfer to B:

```bash
scp hetero-demo.tar james@192.168.50.2:~/lora_zero3_demo/
```

Load on B:

```bash
cd ~/lora_zero3_demo
scripts/build_image.sh --load hetero-demo.tar
```

The image contains source, scripts, and Python dependencies. It does not
contain the model or dataset. Both launchers mount `hf_cache/`, so model and
dataset files download once per machine and survive `--rm` containers.

Every source or dependency change requires rebuilding and redistributing the
image. Configs are also host-mounted at `/app/configs`, so resolved configs do
not require an image rebuild.

## Academic Campaign Commands

The four completed academic tags are:

```text
baseline_1k
2node_1k
baseline_long
2node_long
```

Do not use historical `baseline` or `default` for academic conclusions.

### Single-GPU Run

The launcher now supports a short single-GPU command. It uses the same Docker
flags and mounts but omits `--distributed`:

```bash
scripts/launch_rank0.sh \
  --single-gpu \
  --config configs/resolved/baseline_long.yaml
```

The config selects the tag and budget. The completed long config contains:

```yaml
dataset:
  subset_size: 3288
  validation_size: 100
  seed: 42
logging:
  run_tag: baseline_long
```

### Paired DDP Run

On A, select a completed baseline:

```bash
PREPARE_CHOICE=1 scripts/prepare_2node.sh
```

For normal interactive use, omit `PREPARE_CHOICE`; the script reads selection
from `/dev/tty`. It writes `configs/resolved/2node_1k.yaml` or
`configs/resolved/2node_long.yaml`, prints the raw YAML hash, and prints the
required sync/launch commands.

Transfer the resolved config separately so its directory is preserved:

```bash
scp configs/resolved/2node_long.yaml \
  james@192.168.50.2:~/lora_zero3_demo/configs/resolved/
```

Launch B first:

```bash
scripts/launch_rank1.sh --config configs/resolved/2node_long.yaml
```

Then launch A:

```bash
scripts/launch_rank0.sh --config configs/resolved/2node_long.yaml
```

Do not add `--distributed`; the launch scripts add it by default. The scripts
print a canonical runtime config hash. Both ranks must print the same hash.

### Config Override Rules

`src/train_lora.py` accepts explicit CLI overrides for subset size, epochs,
batch size, sequence length, grad accumulation, learning rate, LoRA rank,
alpha, dropout, and run tag (`--subset-size`, `--epochs`, `--batch-size`,
`--seq-length`, `--grad-accumulation-steps`, `--learning-rate`, `--lora-r`,
`--lora-alpha`, `--lora-dropout`, `--run-tag`). An explicit `--config` path
is fully noninteractive; a bare single-GPU invocation with no arguments
prompts for tunable values with defaults shown.

Academic run tags are restricted to the four campaign tags. This prevents an
accidental run from entering academic artifacts under an unrelated name.

## Training Artifacts

For each tag, rank 0 writes:

```text
logs/run_<tag>_rank0.csv
logs/timing_<tag>_rank0.csv
logs/validation_<tag>_rank0.csv
logs/gpu_<tag>_rank0.csv
logs/<tag>/config.json
logs/<tag>/metadata.json
logs/<tag>/metadata_rank0.json
logs/<tag>/adapter/
```

Rank 1 additionally writes its run, timing, GPU, and metadata files. Rank 0
validation is authoritative; rank 1 does not write a validation CSV.

After Docker runs:

```bash
sudo chown -R "$USER:$USER" logs/
```

The active marker is `logs/.training_active`. If a run is interrupted, partial
files are preserved and future runs refuse to start until the marker is
inspected and cleared.

## Static Analysis

After both runs in a scale pair are complete and rank-1 files have been copied
to A:

```bash
uv run python -m src.analyze --scale 1k
uv run python -m src.analyze --scale long
uv run python -m src.analyze --scale all
```

Outputs:

```text
logs/analysis/1k/
logs/analysis/long/
logs/analysis/speedup_consistency.png
```

Figures include loss/validation loss versus `tokens_seen`, throughput bars,
speedup/efficiency, GPU utilization over time, compute/comm/data/validation/
other time breakdown, effective bandwidth versus the 125 MB/s ceiling, peak
VRAM, communication-stability rolling average, final validation quality, and
cross-scale speedup consistency. Markdown summaries and token-aligned
validation comparison tables are generated alongside the PNGs.

The analysis command refuses incomplete pair artifacts. It uses only the four
academic tags and measured values.

## WebGUI

### Existing Single-Run Viewer

This mode displays one run CSV. It preserves the original viewer workflow.

On A for rank 0:

```bash
uv run python -m src.webui \
  --csv logs/run_2node_1k_rank0.csv \
  --port 8000
```

On B for rank 1:

```bash
uv run python -m src.webui \
  --csv logs/run_2node_1k_rank1.csv \
  --port 8000
```

Open this on each machine:

```text
http://localhost:8000
```

The Metrics tab polls `/api/rows` every second and shows all available run
rows. During an active run, loss/samples are live because the run CSV flushes
each update. Compute/comm timing remains blank until the timing CSV is
finalized at run end; this is intentional because CUDA event timing must not
be synchronized per step.

If port 8000 is occupied:

```bash
uv run python -m src.webui \
  --csv logs/run_2node_1k_rank0.csv \
  --port 8001
```

Open `http://localhost:8001`.

To expose a server beyond localhost:

```bash
uv run python -m src.webui \
  --csv logs/run_2node_1k_rank0.csv \
  --host 0.0.0.0 \
  --port 8000
```

Then open `http://<machine-ip>:8000` from the other machine. Use this only
on a trusted local network; the server has no authentication.

### Academic Pair Dashboard

Generate figures first:

```bash
uv run python -m src.analyze --scale all
```

Run the 1k pair dashboard on A:

```bash
uv run python -m src.webui \
  --pair 1k \
  --analysis-dir logs/analysis \
  --port 8000
```

Run the long pair dashboard:

```bash
uv run python -m src.webui \
  --pair long \
  --analysis-dir logs/analysis \
  --port 8000
```

The scale selector changes between `1k` and `long`. The dashboard displays
generated PNGs and pair summary values from `academic_summary.csv`. The
Download all link creates a ZIP containing locally available CSVs, configs,
metadata, and analysis figures.

Rank-1 files physically live on B. Download All is local to the machine where
the server runs; it does not fetch files from the other laptop. Transfer rank-1
artifacts to A first when creating a complete A-side bundle.

### Inference Tab

Inference is A-only and must be explicitly enabled. It loads the base model
and one selected campaign adapter at a time:

```bash
uv run python -m src.webui \
  --pair 1k \
  --analysis-dir logs/analysis \
  --inference \
  --port 8000
```

Open `http://localhost:8000` and select **Inference**.

The page lists available campaign adapters:

```text
baseline_1k
2node_1k
baseline_long
2node_long
```

Enter a coding instruction and choose generation settings. The page produces
side-by-side BASE and FINETUNED answers. BASE disables the selected adapter;
FINETUNED enables it. The same seed is reset for both generations. Prompts
are visually shown as chat messages but each prompt is evaluated independently;
previous chat turns are not fed back into the model.

Inference controls:

- adapter tag
- maximum new tokens
- temperature
- top-p
- repetition penalty
- seed

Answers are safely escaped and rendered with lightweight local Markdown-like
formatting for headings, inline code, fenced code, and line breaks. Raw HTML
is not executed. Inference is blocked while `logs/.training_active` exists.
Prompts and answers are not persisted by default.

Do not run inference concurrently with training on A; both need the same GPU.

### TUI Fallback

```bash
uv run python -m src.tui --csv logs/run_2node_1k_rank0.csv
```

The Rich TUI shows the last 40 terminal rows. Plain output:

```bash
TUI_RENDER=plain uv run python -m src.tui \
  --csv logs/run_2node_1k_rank0.csv
```

The browser shows all rows; the terminal cap does not apply to WebUI.

## Standalone Inference REPL

The original CLI inference tool remains available on A:

```bash
uv run python -m src.infer \
  --adapter logs/2node_long/adapter \
  --max-new-tokens 512
```

It loads base plus the selected adapter and prints BASE/FINETUNED outputs.
The WebUI inference page is the more convenient four-adapter selector.

## Common Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| NCCL hangs at init | Wrong interface, firewall, or IP | Recheck `ip a`, `NCCL_SOCKET_IFNAME`, ping, and `MASTER_PORT` |
| `invalid device ordinal` on B | Global rank used as CUDA device | Every node uses local `cuda:0` |
| `nvidia-smi telemetry unavailable` | Container cannot execute nvidia-smi | Fix Toolkit/GPU runtime before training |
| Validation OOM | Validation batch too large | Current code uses validation batch size 1; rebuild image if B is stale |
| `logs/.training_active` exists | Previous run interrupted | Inspect partial files, move/preserve them, then remove marker deliberately |
| `run_*_rank1.csv` missing | B did not finish or files not transferred | Check B container/log, then scp rank-1 artifacts to A |
| Rank-1 validation CSV missing | Expected design | Rank 0 writes authoritative validation only |
| Config hashes differ | Different resolved config/source | Stop both ranks, sync exact resolved YAML and image |
| WebUI port busy | Existing listener | Use `--port 8001` or stop the listener |
| WebUI page has no figures | Analysis not generated or wrong directory | Run `uv run python -m src.analyze --scale all`; use `--analysis-dir logs/analysis` |
| Inference tab disabled | Server not started with `--inference` or no adapters | Start A with `--inference`; verify four adapter directories |
| Model appears to download every run | Cache mount missing | Check `-v "$PWD/hf_cache:/root/.cache/huggingface"` and `du -sh hf_cache` |
| Python not found in container | `--user` was added | Remove `--user`; launchers intentionally run as container root |
| Logs are root-owned | Container writes as root | `sudo chown -R "$USER:$USER" logs/` |
| B runs stale code | Image was not rebuilt/loaded | Build once on A, transfer tar, load on B |

## Current Artifact Sources

Academic analysis reads:

```text
logs/academic_summary.csv
logs/run_<academic-tag>_rank<N>.csv
logs/timing_<academic-tag>_rank<N>.csv
logs/validation_<academic-tag>_rank0.csv
logs/gpu_<academic-tag>_rank<N>.csv
logs/<academic-tag>/config.json
logs/<academic-tag>/metadata*.json
```

Historical `logs/summary.csv`, `run_baseline_rank0.csv`, and
`run_default_rank0.csv` are retained for debugging but excluded from the
academic report and analysis.

## Development Checks

```bash
uv run python -m py_compile src/*.py src/web/*.py
bash -n scripts/*.sh
git diff --check
uv run python -m src.analyze --scale all
```

No training or Docker build is needed to regenerate static figures once all
artifacts exist.

## Documentation

- [`report.md`](report.md): measured academic report.
- [`documentation.md`](documentation.md): source-level technical explanation.
- [`gigabit-lora-demo-plan.md`](gigabit-lora-demo-plan.md): phased plan and
  completed status.
- [`AGENTS.md`](AGENTS.md): operational rules and current campaign state.
