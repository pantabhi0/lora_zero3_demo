# HeteroTrain — 2-Node LoRA Fine-Tuning over Gigabit

Fine-tune a small language model (**Qwen/Qwen2.5-0.5B**, base, *not* -Instruct) with **fp16 LoRA** in two configurations and compare them:

| Config | Hardware | Method |
| --- | --- | --- |
| **Baseline** | 1 × GTX 1650 (4 GB, Turing CC 7.5) | plain PyTorch, single GPU |
| **2-node DDP** | 2 × GTX 1650 across two laptops | PyTorch DDP over a 1 Gbps Ethernet link |

The goal is to **quantify the compute/communication tradeoff**: two cheap laptops on one cable vs one laptop. The result is *measured*, not assumed — the GTX 1650 has **no Tensor Cores**, so FP16 ≈ FP32 speed and the 2-node run may be faster *or* slower. It turned out **1.96× faster** (see [Results](#results)).

Everything is Dockerized for identical deployment on both machines, built through a phased plan with a hard acceptance check at each step.

---

## Results

Real runs on this hardware — 1,000 CodeAlpaca samples, 1 epoch, batch 1, seq 256, grad-accum 4, LoRA r=16:

| Metric | Baseline (1 GPU) | 2-node DDP | Δ |
| --- | ---: | ---: | --- |
| Optimizer updates | 250 | 125 | 2× |
| Wall clock | 2410.95 s (≈40.2 min) | 1230.38 s (≈20.5 min) | **1.96× faster** |
| Samples/sec | 0.415 | 0.813 | 1.96× |
| Tokens/sec | 106.2 | 208.1 | 1.96× |
| Compute per update | 9629.4 ms | 9396.4 ms | −2.4% |
| Comm per update (all-reduce) | — | 381.3 ms | — |
| Comm share of step | — | **3.9%** | — |
| Final loss | 0.521 | 0.343 | — |
| Loss criterion (P3/P4) | MET | MET | — |
| Curve match vs baseline | — | OK (max_rel = 0.0017) | — |

**Interpretation.** Splitting the sample budget across two GPUs halves the number of optimizer updates (each update now covers 8 samples instead of 4), and the per-update all-reduce costs only 3.9% of step time — so the 2-node run is ~2× faster despite the communication. On GPU/CPU-poorer hardware or bigger batch sizes the comm ratio would shrink further; on slower links (100 Mbit) it would erode the speedup.

Readout points: laptop **A = rank 0** is the authoritative measurement machine for the headline comparison (both runs' rank-0 logs were produced there — no cross-machine clock skew).

---

## Repo layout

```
hetero-demo/
├── Dockerfile                  # pinned CUDA 13.0 base + uv-provisioned deps
├── pyproject.toml / uv.lock    # all Python deps (torch 2.13.0+cu130, peft, …)
├── .env.example                # per-machine env template (fill in, copy to .env)
├── configs/lora_config.yaml    # model/dataset/training/LoRA/logging settings
├── scripts/
│   ├── build_image.sh          # build once, distribute identical image
│   ├── launch_rank0.sh         # run training on laptop A
│   └── launch_rank1.sh         # run training on laptop B
├── src/
│   ├── hello_world_dist.py     # Phase 1/2 — NCCL+network smoke test
│   ├── train_lora.py           # Phase 3/4 — LoRA fine-tune, single & DDP
│   ├── metrics.py              # timing, live CSV, env gate, acceptance checks
│   ├── infer.py                # Phase 6 — base vs finetuned prompt UI
│   ├── tui.py                  # Phase 6 — live terminal table
│   └── webui.py                # Phase 6 — live browser GUI
├── logs/                       # per-run CSVs, summary, saved adapters
├── hf_cache/                   # HuggingFace cache (volume-mounted, one-time DL)
└── gigabit-lora-demo-plan.md   # the plan this repo implements
```

---

## Requirements

* **Two laptops**, each with one NVIDIA GTX 1650 (4 GB VRAM). Any single-GPU Turing-or-newer NVIDIA card works, but the timing numbers are only meaningful on *identical* GPUs.
* **Ubuntu** on both (built on 26.04 LTS; anything with a recent NVIDIA driver + Docker works).
* A **gigabit-capable Ethernet link** between them (built-in RJ45 or a USB-Ethernet adapter that actually negotiates 1000 Mb/s — verify with `ethtool`, cheap adapters silently cap at 100).
* `uv`, Docker Engine, NVIDIA Container Toolkit (all covered below).

---

## Zero-to-running setup

The order below is exactly the phase order from the plan. **Do not skip ahead** — each step has a check; everything downstream depends on the network step working.

### Step 0 — Physical + static networking (Phase 0)

1. **Cable the laptops directly** (no switch needed; modern NICs auto-negotiate crossover).
2. **Find the interface name** on each machine:

   ```bash
   ip a   # look for the Ethernet interface, e.g. eno1 / enp44s0
   ```

3. **Confirm it runs at gigabit** (both sides):

   ```bash
   ethtool <iface> | grep Speed   # want: Speed: 1000Mb/s
   ```

   If it reports 10/100Mb/s, try forcing it (a bad/old cable or a 100 Mbit-only adapter is the usual culprit):

   ```bash
   sudo ethtool -s <iface> speed 1000 duplex full autoneg on
   ```

4. **Assign static IPs** — one machine becomes **A (rank 0)**, the other **B (rank 1)**. Example with netplan (`/etc/netplan/01-netcfg.yaml`):

   ```yaml
   # Laptop A (rank 0) — adjust interface name
   network:
     version: 2
     renderer: networkd
     ethernets:
       eno1:
         addresses: [192.168.50.1/24]
   ```

   ```yaml
   # Laptop B (rank 1)
   network:
     version: 2
     renderer: networkd
     ethernets:
       enp44s0:
         addresses: [192.168.50.2/24]
   ```

   ```bash
   sudo netplan apply
   ```

   No gateway/DNS needed — this is a point-to-point link.

5. **Bidirectional ping check** (must pass before anything else):

   ```bash
   # on A:
   ping -c 3 192.168.50.2
   # on B:
   ping -c 3 192.168.50.1
   ```

   Expect low, stable latency (≈0.6 ms here). A hang at a later phase is *almost always* this step being wrong — fix it now, not later.

### Step 1 — Host prerequisites (Phase −1)

```bash
# both laptops:
sudo prime-select nvidia && sudo reboot      # hybrid-graphics laptops only
# set power mode: AC + performance (BIOS/OS setting) — battery throttles the GPU
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify after reboot:

```bash
nvidia-smi        # GTX 1650 visible and the only active GPU
uv --version
```

### Step 2 — Docker + NVIDIA Container Toolkit (Phase −1)

Identical on both machines (script it once, run twice):

```bash
# Docker Engine (Ubuntu):
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker        # or leave socket-activated, see note
# add your user to the docker group:
sudo usermod -aG docker "$USER"

# NVIDIA Container Toolkit (registers the `nvidia` runtime with Docker):
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Important.** This repo uses the **native Docker Engine**, not Docker Desktop. Docker Desktop runs a QEMU VM with no GPU passthrough (`--gpus all` fails with "no known GPU vendor found") and its `--network host` is the VM's network, which breaks NCCL. If you only have Desktop, uninstall it and use the native engine.

Verify:

```bash
docker info | grep -i runtime      # must show: nvidia
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

### Step 3 — Copy repo, fill `.env`

Copy the repo to both machines (they must be identical). On each machine:

```bash
cp .env.example .env
$EDITOR .env
```

Values — **same on both nodes**:

```bash
MASTER_ADDR=192.168.50.1      # laptop A's static IP
MASTER_PORT=29500
WORLD_SIZE=2
```

**Per machine** (note `RANK` and the interface):

```bash
# on A:
RANK=0
NCCL_SOCKET_IFNAME=eno1
# on B:
RANK=1
NCCL_SOCKET_IFNAME=enp44s0
```

There are **no defaults in code** — a missing var fails with a clear error (never auto-detected). Docker overrides `RANK` per launch script anyway.

### Step 4 — Bare-metal hello world (Phase 1)

Validates GPU + NCCL + the network before any ML complexity:

```bash
# laptop A:
RANK=0 uv run python -m src.hello_world_dist
# laptop B (same moment):
RANK=1 uv run python -m src.hello_world_dist
```

**Check:** both print `all_reduce ok: tensor[0]=3.0 (expected 3.0) in ~0.3–0.5 ms` and exit cleanly. If it *hangs* at init: `NCCL_SOCKET_IFNAME` is wrong, or a firewall blocks the rendezvous port — fix the root cause, do not add retries.

### Step 5 — Build + distribute the image (Phase 2)

Build **once**, ship the identical image to both machines:

```bash
# on A:
uv sync                    # bare-metal env, also used by the tools below
scripts/build_image.sh --save
scp hetero-demo.tar james@192.168.50.2:~/lora_zero3_demo/
```

```bash
# on B:
scripts/build_image.sh --load hetero-demo.tar
```

**Check (both machines):**

```bash
docker run --rm --gpus all hetero-demo:latest python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# → 2.13.0+cu130 13.0 True
```

Then re-run the hello world inside containers:

```bash
# A:
docker run --rm --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file .env --env RANK=0 \
  -v "$PWD/logs:/app/logs" -v "$PWD/hf_cache:/root/.cache/huggingface" \
  hetero-demo:latest python -m src.hello_world_dist
# B: same with --env RANK=1
```

**Check:** same `3.0` result, inside Docker. If Phase 1 passed but this fails: `--network host` missing or an env var not passed through.

> **HF cache.** `hf_cache/` is volume-mounted to `/root/.cache/huggingface` so the model + dataset download **once per machine** and are reused on every `--rm` container run.

### Step 6 — Single-GPU baseline (Phase 3)

On laptop A only:

```bash
scripts/launch_rank0.sh --run-tag baseline
# plain docker equivalent (same flags as Phase 2) — launch_rank0.sh adds --distributed,
# so for a single-GPU baseline run train_lora WITHOUT --distributed instead:
uv run python -m src.train_lora --config configs/lora_config.yaml --run-tag baseline
```

*(The 2-node script is `--distributed`; the bare-metal single-GPU command above is the literal Phase 3 baseline.)*

Takes ≈40 min for the default budget. **Check:** `logs/summary.csv` gains a `baseline` row with `loss_criterion_met=1`, and `logs/baseline/adapter/` exists (the saved LoRA adapter).

### Step 7 — 2-node DDP run (Phase 4)

Same config, both machines:

```bash
# A:
scripts/launch_rank0.sh
# B:
scripts/launch_rank1.sh
```

Takes ≈20 min. **Check:** a `default` row in `logs/summary.csv`, and the loss curve matches the baseline within ~1–2% per step (the script prints this automatically).

> **run_tag quirk.** `configs/lora_config.yaml` sets `logging.run_tag: default`, so the 2-node run is tagged `default` and its adapter lands in `logs/default/adapter`. The script's auto "2node"/"baseline" fallback only fires when the config has no `run_tag`. Use `--run-tag` to override.

### Step 8 — Watch it live (Phase 6)

After (or during) a run, on each machine:

```bash
# A:
uv run python -m src.webui --csv logs/run_default_rank0.csv
# B:
uv run python -m src.webui --csv logs/run_default_rank1.csv
```

Open `http://localhost:8000` on each laptop. The page polls every second and shows every step: loss, samples processed, compute_ms, comm_ms. Terminal alternative: `uv run python -m src.tui --csv logs/run_default_rank0.csv` (or `TUI_RENDER=plain` for plain stdout). See [Tool reference](#tool-reference).

### Step 9 — Compare base vs finetuned (Phase 6 add-on, laptop A only)

```bash
uv run python -m src.infer                 # default adapter: logs/default/adapter
# or, for the baseline adapter:
uv run python -m src.infer --adapter logs/baseline/adapter
```

Type prompts; it prints the **BASE** and **FINETUNED** answers side by side so the fine-tuning effect is visible. `quit` / `exit` / Ctrl-D to leave.

---

## Tool reference

| Tool | Command | Purpose |
| --- | --- | --- |
| Hello world | `uv run python -m src.hello_world_dist` | NCCL + network smoke test (needs `RANK`/`MASTER_ADDR`/… in env) |
| Train | `uv run python -m src.train_lora --config configs/lora_config.yaml [--distributed] [--run-tag TAG] [--subset-size N] [--epochs N]` | LoRA fine-tune; `--distributed` for the 2-node DDP run |
| Validate timings | `… --validate-timings` | short `torch.profiler` run cross-checking comm-hook timing, then exits |
| TUI | `uv run python -m src.tui --csv logs/run_default_rank0.csv [--refresh 1.0]` | live terminal table (last 40 rows); `TUI_RENDER=plain` for stdout |
| Web GUI | `uv run python -m src.webui --csv logs/run_default_rank0.csv [--port 8000] [--host 127.0.0.1] [--refresh 1.0]` | live browser table, all rows; `--host 0.0.0.0` to view the other machine's |
| Infer | `uv run python -m src.infer [--adapter logs/baseline/adapter] [--max-new-tokens 512]` | base vs finetuned prompt REPL (machine A) |
| Build image | `scripts/build_image.sh [--save \| --load FILE.tar]` | build once / export tarball / import tarball |
| Launch rank 0 | `scripts/launch_rank0.sh` | docker run training on A |
| Launch rank 1 | `scripts/launch_rank1.sh` | docker run training on B |

Environment variables (all required, no defaults):

| Var | Value |
| --- | --- |
| `MASTER_ADDR` | laptop A static IP (same both nodes) |
| `MASTER_PORT` | e.g. `29500` |
| `WORLD_SIZE` | `2` |
| `RANK` | `0` on A, `1` on B |
| `NCCL_SOCKET_IFNAME` | per-machine Ethernet iface (A: `eno1`, B: `enp44s0` here) |

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `uv run …` fails with "Missing required env var …" | env not set | fill `.env` and re-run (repo `.env` is auto-loaded), or export the vars |
| Hello world / training hangs at init | wrong `NCCL_SOCKET_IFNAME`, or firewall blocking `MASTER_PORT` | fix iface name (`ip a`), allow the port, no retry-workarounds |
| `ping` fails between laptops | static IP / cable problem | recheck Step 0 (interface, netplan, cable) |
| `ethtool` shows 10/100Mb/s | old cable or 100 Mbit adapter | try `sudo ethtool -s <iface> speed 1000 duplex full autoneg on`; otherwise swap adapter |
| `docker run --gpus all` → "no known GPU vendor found" | Docker Desktop in use | use the native Docker Engine |
| `--gpus all` fails after Toolkit install | `nvidia` runtime not registered | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Container: `exec: "python": not found` | `--user` flag added | remove `--user`; the venv python lives under root-only `/root` |
| NCCL "invalid device ordinal" on B | script used global `RANK` as device id | must use local `cuda:0` (`torch.cuda.set_device(0)`) — already fixed in this repo |
| "Unable to cast Future to Tensor" from comm hook | hook returned the Future, not the buffer | return `bucket.buffer()` — see `make_timing_hook` |
| `ValueError: unsupported format character '}'` from webui | `%`-formatting the HTML template which contains `%` in CSS | template uses `.replace("__REFRESH_MS__", …)` instead |
| Logs owned by root | container runs as root, `--rm` | `sudo chown -R $USER: logs/` after each run |
| B runs old code | image bakes `src/`, B never reloaded it | rebuild + `docker save`/`--load` again (any `src/` change) |
| Port 8000 already in use | leftover listener | `sudo fuser -k 8000/tcp`, or `--port 8001` |
| Model re-downloads every run | `hf_cache/` mount missing | add `-v "$PWD/hf_cache:/root/.cache/huggingface"` |
| Loss diverges between baseline and 2-node | bit-level all-reduce non-associativity (expected) | compare with `curve_matches` (tol 2%); >tol = real bug |
| Web table starts at step 86 | old TUI-era 40-row cap | pull latest `src/webui.py` (now renders all rows) |

---

## Gotchas & design constraints (why it is the way it is)

* **No auto-detection, ever.** `MASTER_ADDR`, `NCCL_SOCKET_IFNAME`, static IPs are required config from `.env`; missing → loud failure. This is deliberate — guessing the NIC is how distributed runs silently hang.
* **One GPU per node** → every process uses local `cuda:0`, *never* the global `RANK` (an easy one-line bug that only appears on node B).
* **One all-reduce per optimizer update.** Non-final gradient-accumulation micro-steps run under `model.no_sync()`; the comm hook fires exactly once per update, which is what makes the timing numbers clean.
* **Fixed sample budget is the comparison unit.** All runs consume the same total samples (default 1000) regardless of batch/accum/world size — so 2-node does fewer, bigger updates, which is precisely the point.
* **CUDA events are synced once, at end of run** — per-step syncs would distort the very wall-clock being measured.
* **Comm timing covers the FULL hook** (all-reduce + bucket flatten/copy + divide), and compute time subtracts the overlapped comm so the two don't double-count.
* **Frozen base params** → DDP all-reduces only the LoRA adapter gradients (tiny, ~3 MB), keeping comm at 3.9% of step time.
* **Same model on both machines, identical Docker image** — no cross-architecture or version drift.
* `.env` (per-machine values), `hf_cache/`, `logs/`, `.venv/`, and `hetero-demo.tar` should be `.gitignore`d before pushing to GitHub. No git repo is initialized in this project.

---

## License

Not specified — check with the repo owner.