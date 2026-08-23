# HeteroTrain Demo: 2-Node LoRA Fine-Tuning Speedup over Gigabit (2× GTX 1650 Laptops)

## Goal

Fine-tune the same small model with LoRA in two configurations — (a) a
single GTX 1650, (b) two GTX 1650s across two laptops via PyTorch DDP over
a gigabit Ethernet link — and compare **training time** and **communication
overhead** between them. Both configurations are expected to complete
successfully; this is a speed/overhead comparison, not a feasibility
(OOM) test. Package everything in Docker for consistent deployment across
both machines. Wrap the demo in a simple web GUI (stdlib-only), with a TUI
kept as a plain-terminal fallback.

**Status: COMPLETE.** All phases −1→6 passed. Measured result: **1.96× faster
with 2 nodes**, communication = **3.9% of step time**. Full details in the
per-phase status notes below and in `README.md` / `documentation.md`.

---

## Status and measured results (as-built)

Academic campaign is complete. Use only these four tags for final conclusions:
`baseline_1k`, `2node_1k`, `baseline_long`, and `2node_long`. Historical
`baseline` and `default` artifacts are retained only for reference. Academic
loss uses correct response-only `-100` masking and held-out validation; curves
are aligned by `tokens_seen`, never raw optimizer step.

| Phase | Result |
| --- | --- |
| −1 Host prerequisites | PASS (driver, prime-select, AC/perf, uv, Docker+Toolkit both) |
| 0 Network | PASS — A `192.168.50.1/24` (`eno1`), B `192.168.50.2/24` (`enp44s0`); gigabit forced with `ethtool` (link partner originally advertised 10/100 — bad cable); ping ≈0.58 ms |
| 1 Bare-metal hello | PASS — `all_reduce ok: tensor[0]=3.0` both ranks, clean exit |
| 2 Dockerized hello | PASS — same `3.0` inside containers (`--network host`, `--env-file .env`) |
| 3/4 Academic 1k pair | PASS — `baseline_1k` 3286.125 s, `2node_1k` 1668.567 s, speedup 1.969x; final validation loss 0.606460/0.572788 |
| 3/4 Academic long pair | PASS — `baseline_long` 8841.677 s, `2node_long` 4475.444 s, speedup 1.976x; final validation loss 0.567638/0.565360 |
| 5 Metrics | PASS — torch.profiler cross-check: hook 3784.9 ms vs profiler NCCL kernels 3777.3 ms over same window (**0.2% agreement**) → hook methodology trusted |
| 6 Web GUI | PASS — legacy viewer, academic pair dashboard, PNG serving, Academic Summary, ZIP export, A-only inference tab; `src/tui.py` fallback |

Headline academic numbers are maintained in `logs/academic_summary.csv` and
fully interpreted in `report.md`. Compare curves by `tokens_seen`, never raw
optimizer step. Historical training loss uses different label semantics and is
excluded from academic conclusions.

---

## Read this first — steps only you can do (not the coding agent)

A coding agent can write and run code, but it cannot touch physical
hardware or your network. These steps are yours, done before or alongside
the agent's work:

1. **Install Ubuntu 26.04 LTS on both laptops.** This is a very recent LTS
   release — before relying on it, run `ubuntu-drivers devices` on each
   laptop and confirm it offers a driver that recognizes the GTX 1650. If
   it doesn't yet, you'll need NVIDIA's own CUDA/driver repo for Ubuntu
   26.04 directly from NVIDIA rather than waiting on Ubuntu's default
   repos — check `developer.download.nvidia.com` for the current 26.04
   path when you get here, since availability may lag the OS release.
   *(Done: Ubuntu 26.04 + driver verified with `nvidia-smi` on both.)*
2. **Physically connect the two laptops via Ethernet** (built-in RJ45 on
   each, or a confirmed gigabit-rated USB-Ethernet adapter — cheap
   adapters sometimes silently cap at 100Mbps, verify with `ethtool
   <iface>` showing `Speed: 1000Mb/s`). *(Done; a bad cable had silently
   dropped the link to 10/100 — fixed by forcing `sudo ethtool -s <iface>
   speed 1000 duplex full autoneg on` on A and swapping the cable.)*
3. **Determine the network interface name on each laptop and assign static
   IPs.** This is intentionally not decided or hardcoded in this plan —
   you'll do this once your physical setup is in front of you (`ip a` to
   find the interface name, e.g. `enp3s0`). Fill the interface name and IPs
   into the config/env values described in Phase 0 once known. The code
   the agent writes should read this as required config with no default —
   not guess or auto-detect it. *(Done: A=`eno1`/192.168.50.1, B=`enp44s0`/
   192.168.50.2.)*
4. Everything else below (Docker, uv, NVIDIA Container Toolkit install,
   the training script, metrics, web GUI/TUI) can be built and automated by
   your coding agent.

---

## Hard constraints (do not violate)

- Two laptops, each with one GTX 1650 GPU (Turing, CC 7.5, 4GB VRAM), OS
  Ubuntu 26.04 LTS on both. Same GPU model on both machines, so no
  cross-architecture complexity — do not add TORCH_CUDA_ARCH_LIST juggling.
  FlashAttention-2 is not supported on Turing regardless (sm80+ only) —
  use PyTorch's default/SDPA attention backend, do not attempt to install
  or enable FA2.
- Use plain fp16 LoRA (no quantization). The model is chosen small enough
  (see Model choice) to fit comfortably on a single GTX 1650 standalone —
  quantization isn't needed for feasibility here, only for going bigger,
  which is out of scope for this demo.
- Use **PyTorch DDP** for the 2-node run, not DeepSpeed/ZeRO. Both GPUs
  hold a full copy of the model; only LoRA adapter gradients are
  synchronized via all-reduce each step. This keeps the communication
  pattern simple (one all-reduce per step) and easy to log cleanly.
- Use **`uv`** for all Python dependency management, both bare-metal and
  inside Docker — not `pip` directly. Install: `curl -LsSf
  https://astral.sh/uv/install.sh | sh`. Prefer a `pyproject.toml` +
  `uv.lock` committed to the repo (via `uv add <package>` /  `uv sync`)
  over a loose `requirements.txt`, so both laptops resolve to byte-identical
  dependency versions. `uv.lock` is the single source of truth for ALL
  Python packages incl. torch (2.13.0+cu130).
- **`NCCL_SOCKET_IFNAME` (and the static IPs) have no default value in
  code.** Read them from an env file / config the user fills in per the
  "Read this first" section above. If unset, fail with a clear error
  telling the user to set it — do not attempt to auto-detect the interface.
  *(As-built: `src/metrics.py` `require_dist_env()` gates Phases 1/4; a
  `load_env_file()` auto-loader fills UNSET vars from the repo-local `.env`
  so bare-metal `uv run` needs no manual `source`, while docker
  `--env-file` still wins because existing vars are never overridden.)*
- **One GPU per node → local device `cuda:0`, NEVER the global `RANK`.** Both
  `hello_world_dist.py` and `train_lora.py` use `torch.cuda.set_device(0)` +
  `device_ids=[0]`. *(Real bug hit on B: `set_device(rank)` with RANK=1 →
  "invalid device ordinal".)*
- Network is a single gigabit link between the two machines only. Do not
  implement or assume multi-node beyond 2 nodes.
- No DeepSpeed/pdsh/SSH-based multi-node launcher needed since this is
  plain `torch.distributed` DDP — launch each node's process manually with
  explicit env vars (MASTER_ADDR, MASTER_PORT, WORLD_SIZE=2, RANK).
- Batch size, sequence length, and grad-accumulation steps must be
  configurable (config file or CLI args), not hardcoded, with a sensible
  default. Do not build a batch/seq-length sweep or auto-varying logic —
  just make it a one-line config edit to change any of them and rerun, for
  optional manual variation later (see Phase 5, metric 4).
- **Comparison unit = fixed SAMPLE BUDGET** (default 1000), identical across
  all runs regardless of batch/accum/rank split. Atomic logged unit =
  optimizer update; per-step throughput is derived from logs afterward, not
  hand-matched to configs.
- **Exactly ONE all-reduce per optimizer update.** Wrap non-final
  grad-accum micro-steps in `model.no_sync()`.
- **Comm is timed via a DDP comm hook** covering the FULL hook duration
  (bucketed all-reduce + flatten/copy + divide). Compute CUDA events are
  accumulated and synchronized ONCE at end of run — never per-step (per-step
  syncs would distort the wall-clock being measured).
- **Freeze base params** (`requires_grad=False`) so DDP all-reduces only the
  LoRA adapter grads (~3 MB) — the whole reason comm is 3.9%, not 50%.
- Build and validate in the phase order below. Do not start the web GUI
  (Phase 6) until Phases 0–4 pass. A working demo with plain log output is
  an acceptable final fallback if time runs out.
- Every phase has an explicit acceptance check. Do not proceed to the next
  phase until the current one's check passes.

## Model choice

Pick a small model that fits comfortably (with real headroom, not at the
OOM edge) on a single GTX 1650 with plain fp16 LoRA. **As-built:
`Qwen/Qwen2.5-0.5B` — the BASE model, NOT -Instruct.** *(Changed from the
original suggestion of Qwen2.5-0.5B-Instruct: the -Instruct model already
answers like an assistant, so its loss starts close to convergence and the
visible loss drop that makes the demo legible would be small. The base model
starts confused and drops visibly.)* Rough budget: fp16 weights for a ~0.5B
model are ~1GB; with LoRA adapters, optimizer state, and activations (batch
size 1, gradient checkpointing on) this stays well under 4GB, leaving margin
for the Phase 5 batch/seq-length variation runs without hitting OOM.---

## Phase -1 — Host prerequisites (Ubuntu 26.04, driver, uv, Docker)

**Status: PASS.**

1. Confirm NVIDIA driver is correctly installed and active per the "Read
   this first" section — verify with `nvidia-smi` on both laptops.
2. On both laptops, run `sudo prime-select nvidia` (not `on-demand`) if
   the laptop has hybrid/Optimus graphics, to force the discrete GTX 1650
   as active rather than an integrated GPU. Reboot and re-verify with
   `nvidia-smi`.
3. Set both laptops to plugged-in, performance/high-power mode. Battery or
   balanced power profiles throttle the GPU/CPU and will skew the
   training-time comparison this whole demo is built around.
4. Install `uv` on both laptops (see Hard Constraints for the install
   command).
5. Install Docker Engine + NVIDIA Container Toolkit on both laptops (same
   steps on each, script it once and run identically on both rather than
   doing it by hand twice, to avoid drift).
6. Close unnecessary GUI applications/browser tabs before running the
   actual demo — at 4GB total VRAM, desktop compositor and browser GPU
   usage is a meaningful fraction of the budget.

**Acceptance check:** `nvidia-smi` on both machines shows the GTX 1650 as
the only visible/active GPU, both are on AC power in performance mode, `uv
--version` and `docker run --rm --gpus all nvidia/cuda:*-base-ubuntu*
nvidia-smi` both succeed on both laptops. *(As-built: the Docker daemon is
the NATIVE engine — systemd `docker.service`, socket-activated, starts on
first CLI use; NVIDIA Container Toolkit runtime `nvidia` registered in
daemon.json. Docker Desktop must NOT be used: it runs a QEMU VM with no GPU
passthrough — `--gpus all` fails with "no known GPU vendor found" and its
`--network host` is the VM's network, which breaks NCCL.)*

---

## Phase 0 — Network validation (no code yet)

**Status: PASS.**

1. Connect the two laptops directly via Ethernet cable, using whichever
   path was confirmed gigabit-capable in the "Read this first" section.
   Crossover cable is not required — modern NICs auto-negotiate regardless
   of cable type.
2. Confirm the static IPs and interface names you assigned per "Read this
   first":
   - Laptop A: `192.168.50.1/24`, interface `eno1` (r8169)
   - Laptop B: `192.168.50.2/24`, interface `enp44s0`
   *(Gigabit was forced on A with `sudo ethtool -s eno1 speed 1000 duplex
   full autoneg on` — the link partner originally advertised 10/100, the
   cable was the suspect. Verify `Speed: 1000Mb/s` with ethtool on both
   sides.)*
3. From each machine, `ping` the other machine's static IP. *(Measured
   ≈0.58 ms one-way.)*
4. Put `NCCL_SOCKET_IFNAME` (per-machine, they may differ: A=`eno1`,
   B=`enp44s0`) into the env file the training scripts will read — this is
   the value from "Read this first," not something the agent should invent
   or guess.

**Acceptance check:** bidirectional ping succeeds with low, stable latency,
and both laptops have their `NCCL_SOCKET_IFNAME` value recorded in config.
Do not proceed until this works — nothing downstream will function if this
fails.

---

## Phase 1 — Distributed "hello world" (no ML, no Docker yet)

**Status: PASS.** Both ranks printed `all_reduce ok: tensor[0]=3.0 (expected
3.0)` and exited cleanly.

Write a minimal `torch.distributed` script (`src/hello_world_dist.py`):
- Initializes a process group (`backend="nccl"`) using env vars
  `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE=2`, `RANK`.
- Each rank creates a small tensor on its GPU, does one `all_reduce`, and
  prints the result plus elapsed time.
- Run manually: on laptop A (`RANK=0`), on laptop B (`RANK=1`), pointing
  `MASTER_ADDR` at laptop A's static IP.
- Reads `NCCL_SOCKET_IFNAME` from the Phase 0 config/env file (per-machine
  value) before launching — do not hardcode it in the script.
- **Pins local device 0** (`torch.cuda.set_device(0)`), never global RANK.

**Acceptance check:** both processes connect, complete the all_reduce, and
exit cleanly with matching results. This confirms GPU + NCCL + network are
all correctly wired together before any ML complexity is added.

**Common failure mode to check for:** if the processes hang at
initialization rather than erroring, it is almost always
`NCCL_SOCKET_IFNAME` pointing at the wrong (or no) interface, or a firewall
blocking the rendezvous port. Do not add retry/timeout logic to paper over
this — fix the actual interface/firewall issue, since it will resurface in
every later phase otherwise. *(One real hiccup: B's sshd was down once —
connection refused — restarted by the user.)*

---

## Phase 2 — Dockerize the hello-world

**Status: PASS.** Same `3.0` result inside containers on both machines.

1. Base image: a pinned CUDA/cuDNN image matching the project's locked
   CUDA/PyTorch versions — `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04`,
   not an NGC PyTorch image (which bundles its own torch that would fight
   uv.lock). Use `uv` inside the Dockerfile for Python dependencies on top
   of the base image (`COPY pyproject.toml uv.lock ./` then `RUN uv sync
   --frozen --python 3.12`) — not `pip install` directly. *(No
   cudnn9-suffixed tag exists for CUDA 13.0.0; `13.0.0-cudnn-devel` ships
   cuDNN 9.x. devel not runtime image — needed for toolkit-level pieces.
   Host driver provided by host via NVIDIA Container Toolkit; never install
   a driver inside the container.)*
2. Build the same image once, load/distribute it identically to both
   machines (do not rebuild separately on each machine — that risks the
   exact version drift Docker and uv's lockfile are both meant to prevent).
   `scripts/build_image.sh --save` writes a tarball, `--load FILE.tar`
   loads it. *(Image ~6.8 GB. Images are per-daemon: a Docker-Desktop build
   does NOT appear in the native engine.)*
3. Run both containers with:
   - `--gpus all`
   - `--network host` (required — do not use default bridge networking,
     it breaks NCCL's socket rendezvous)
   - `--ipc=host`
   - `--ulimit memlock=-1 --ulimit stack=67108864`
4. Pass `NCCL_SOCKET_IFNAME`, `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`,
   `RANK` in via an env file per node (`--env-file .env`, plus `--env
   RANK=0/1` forced per launch script) — the same per-machine values
   recorded in Phase 0, not new ones.
5. Re-run the Phase 1 hello-world script inside these containers on both
   machines.
6. **Volume mounts added:** `hf_cache/` → `/root/.cache/huggingface` (model
   + dataset download ONCE per machine, reused across `--rm` runs) and
   `logs/` → `/app/logs`. *(Container is `--rm`, so without the mount every
   run re-downloads.)*

**Acceptance check:** identical result to Phase 1, now running inside
Docker on both machines. If this fails but Phase 1 passed bare-metal, the
issue is almost always `--network host` missing or a container-side env
var not passed through — isolate before moving on. *(Real bug: B's
container ran the OLD `set_device(rank)` code because the image bakes
`src/` — re-exporting the rebuilt image fixed it.)*

---

## Phase 3 — LoRA fine-tune script, single-GPU baseline

**Status: PASS.**

1. Use HuggingFace Transformers + PEFT. Apply LoRA (r=16, alpha=32,
   dropout=0.05) to Qwen2.5-0.5B (base), fp16, no quantization, no
   DeepSpeed — plain PyTorch here.
2. Make batch size, seq length, grad-accum config-driven
   (`configs/lora_config.yaml`). Do not hardcode these.
3. Get a normal single-GPU training loop working and producing a
   decreasing loss on a small dataset, on one laptop only, no distribution
   yet. This run should complete successfully — it's your baseline, not a
   feasibility test. *(Run on laptop A only, `--run-tag baseline`, in
   Docker.)*
4. Instrument the training loop to log, per step, forward+backward compute
   time. (Communication time isn't applicable yet — added in Phase 4.)
   Also log total wall-clock time for a fixed sample budget.

**Acceptance check:** loss decreases over a short run on a single GPU
inside the Docker container, and per-step compute time plus total
wall-clock time for the run are logged to a file. This is your baseline
for the Phase 5 comparison.

Historical baseline artifacts remain for debugging only; academic baseline is
`baseline_1k` and `baseline_long`.---

## Phase 4 — Two-node distributed LoRA run via DDP

**Status: PASS.**

1. Wrap the Phase 3 script with the same `torch.distributed` init pattern
   validated in Phase 1/2 (`RANK`, `WORLD_SIZE=2`, `MASTER_ADDR`, the
   per-machine `NCCL_SOCKET_IFNAME` from Phase 0), wrapping the model in
   `torch.nn.parallel.DistributedDataParallel` (`device_ids=[0]`,
   `gradient_as_bucket_view=True` — required for the buffer-based comm
   hook).
2. Only the LoRA adapter parameters need gradients — freeze base params
   (`requires_grad=False`) so DDP syncs just the adapter grads. Add
   `enable_input_require_grads()` for gradient checkpointing under frozen
   base params.
3. Launch rank 0 on laptop A and rank 1 on laptop B (two separate `docker
   run` commands via `scripts/launch_rank0.sh` / `launch_rank1.sh`, one per
   machine).
4. Confirm both GPUs are training the same run (matching loss curves, step
   counts advancing together), using the identical config (model, batch
   size, seq length, sample budget) as the Phase 3 baseline, so the two runs
   are a fair comparison.
5. Instrument this run the same way as Phase 3, plus: per-step
   communication time (the all-reduce call), separate from compute time.
   *(As-built: comm timed via `ddp.register_comm_hook(None,
   make_timing_hook(...))` covering the FULL hook duration; the final
   micro-step's compute span contains the all-reduce, so compute reported =
   raw − comm (no double-count). One all-reduce per update via
   `model.no_sync()` on non-final micro-steps.)*

**Acceptance check:** 2-node run completes without hangs or NCCL errors,
produces a valid loss curve, and logs per-step compute time, per-step
comm time, and total wall-clock time — using the same config as the Phase
3 baseline run.

Historical DDP artifacts remain for debugging only; academic DDP artifacts are
`2node_1k` and `2node_long`. The historical run_tag quirk (`run_tag: default`)
no longer applies: academic configs resolve tags explicitly and training
rejects non-campaign tags.

---

## Phase 5 — Metrics: compare training time and communication overhead

**Status: PASS.**

1. **Training time comparison.** Report total wall-clock time for the same
   sample budget: single-GPU (Phase 3) vs. 2-node DDP (Phase 4). This is
   the core comparison the demo exists to produce. *(As-built: 2410.95 s
   vs 1230.38 s → 1.96×.)*
2. **Communication overhead.** From Phase 4's per-step compute time vs.
   comm time, report the ratio and the % of step time spent on the
   all-reduce. *(As-built: 381.3 ms on a 9396 ms compute step = 3.9%.)
   Compute is reported de-overlapped: `compute_clean = compute − comm`
   per step.*
3. **Absolute practicality.** Log actual tokens/sec for both runs, and
   extrapolate to a full-epoch time estimate for each (e.g. "at this rate,
   one epoch would take ~X minutes"). *(As-built: 106.2 vs 208.1 tok/s;
   `extrapolated_epoch_s` in academic_summary.csv.)*
4. **Batch size / sequence length capability (build now, run later if time
   allows).** Since these are already config-driven per Phase 3, no extra
   code is needed — this is just "manually change the config value and
   rerun both configurations, log the result." Do not build automated
   sweep/parameter-search logic. **OPTIONAL per descope order — not run.**

**Acceptance check:** a saved log/CSV with, for both the single-GPU and
2-node runs: total wall-clock time, per-step compute time, per-step comm
time (2-node only), tokens/sec, and extrapolated epoch time. Metric 4's
extra runs are logged the same way if and when they're done.

**Timing-methodology validation (required before trusting Phase 5 logs):**
run `--validate-timings` on the comm-hook numbers under `torch.profiler`.
*(Measured: comm-hook wall 3784.9 ms vs profiler raw NCCL kernel time
3777.3 ms over the same window — 0.2% agreement → hook methodology
trusted.)* Validation-mode bug found+fixed: the single-pass
`range(accum*3)` loop with `micro == accum-1` fired once, not 3× — fixed to
mirror the real loop with `no_sync()`. Also: torch 2.13 renamed profiler
`cuda_time_total` → `device_time_total`/`self_device_time_total`, and only
`ncclDevKernel`-prefixed events are counted (other names triple-count the
same op).

---

## Phase 6 — Web GUI + TUI fallback (build last, only after Phase 5 passes)

**Status: PASS.** `src/webui.py` (stdlib-only) and `src/tui.py` (rich,
plain-stdout fallback) both tested against the Phase 4 logs: 125 rows, 0
empty-timing.

1. *(Changed: a web GUI replaces the original rich/textual TUI as the
   primary viewer — same data source, zero new dependencies.)* `src/webui.py`
   uses `http.server` + inline HTML/JS: browser polls `/api/rows` every
   `--refresh` s (default 1.0); table shows ALL rows (TUI's last-40 was a
   terminal-height cap; browsers scroll). `--host 0.0.0.0` to view the
   other machine's table from one laptop.
2. Read from the same log/metrics source produced in Phase 5 — do not
   build separate logging logic for the viewer; it visualizes the existing
   instrumentation (`LiveCSV` rows flushed every step).
3. Keep a plain-stdout/log fallback path working at all times — `src/tui.py`
   stays as a plain-stdout fallback (rich table, or `TUI_RENDER=plain`).

**Acceptance check:** the viewer displays live, correct data during an
actual run, sourced from the real metrics, not mocked/simulated values.
*(As-built: one server per machine, A=rank0 CSV, B=rank1 CSV, side-by-side
in browsers. Merging rule: timing CSV steps are 0-based, run CSV 1-based —
merge by POSITION, never by step key. Real bug: the first webui rendered
row 125 with empty timings — the positional merge fixed it. Also the HTML
template must use `.replace("__REFRESH_MS__", …)` not `%`-formatting,
because the CSS contains `width:100%`.)*

**Phase 6 add-on — inference UI (machine A only):** `src/infer.py` loads the
base Qwen2.5-0.5B AND the trained adapter (two ~0.5B fp16 models, fits the
4 GB 1650), REPL that prints BASE and FINETUNED answers side by side.
Academic inference lists the four campaign adapters and loads one selected
adapter at a time; fails with a clear error if missing. Generation is SAMPLING
(do_sample,
temp 0.7, top_p 0.9, repetition_penalty 1.2) — greedy decode loops/repeats
on a 0.5B and leaks CodeAlpaca "### " sections; output is cut at the first
`\n### `. Default `--max-new-tokens 512` (practical ceiling ~8000 on the
1650 due to KV cache ~98 KB/token; 20–40 tok/s).

---

## Suggested file/repo structure (as-built)

```
hetero-demo/
  Dockerfile
  pyproject.toml            # uv-managed dependencies
  uv.lock                   # single source of truth (torch 2.13.0+cu130)
  .env.example              # MASTER_ADDR, NCCL_SOCKET_IFNAME etc. — user fills real values
  .env                      # per-machine, gitignored before publishing
  configs/
    lora_config.yaml        # model, dataset, batch/seq/accum, LoRA, run_tag
  scripts/
    build_image.sh          # build once / --save tarball / --load tarball
    launch_rank0.sh         # run on laptop A (RANK=0, hf_cache+logs mounts)
    launch_rank1.sh         # run on laptop B (RANK=1)
  src/
    hello_world_dist.py     # Phase 1/2
    train_lora.py           # Phases 3–4 + --validate-timings (Phase 5)
    metrics.py              # Phase 5 logging/timing/env-gate/acceptance core
    infer.py                # Phase 6 add-on (base vs finetuned, machine A)
    tui.py                  # Phase 6 fallback (rich or plain stdout)
    webui.py                # Phase 6 primary viewer (stdlib web GUI)
  logs/                     # academic CSVs, analysis PNGs, adapters
  hf_cache/                 # volume-mounted HF cache (download once per machine)
```

## Environment variables reference (both nodes — values filled in by user per Phase 0)

```
MASTER_ADDR=<laptop A's static IP>      # same value on both nodes
MASTER_PORT=29500
WORLD_SIZE=2
RANK=<0 on laptop A, 1 on laptop B>     # launch scripts force RANK anyway
NCCL_SOCKET_IFNAME=<per-machine, no default>   # A=eno1, B=enp44s0 here
```

Repo-local `.env` is auto-loaded by `metrics.load_env_file()` for UNSET
vars only (existing vars win, so docker `--env-file` takes precedence);
bare-metal `uv run` needs no manual `source`.

---

## If time runs short: descoping order

Cut in this order, stopping as soon as you're back on schedule:
1. Web GUI — fall back to the TUI, then plain logs. *(As-built: web GUI
   done; TUI retained as fallback regardless.)*
2. Phase 5 metric 4 (manual batch/seq-length variation runs) — the config
   support stays in the code either way since it's free to build, only the
   extra logged runs are optional. *(Not run — optional.)*
3. Docker packaging — fall back to bare-metal on both machines (the
   scientific claim doesn't depend on Docker, only the "consistent
   deployment" nicety does; `uv` alone still gives you dependency
   consistency without Docker if needed). *(As-built: Docker used and
   validated, Phases 2–4.)*
4. Do not cut Phase 3's baseline run or Phase 4's 2-node run — that
   comparison is the actual demo. *(Both done.)*
5. Do not cut Phase 5 metrics 1–3 (training time comparison, communication
   overhead, tokens/sec) — these are the actual deliverable. *(All done.)*
6. Do not cut Phase 0/1 network validation — skipping it just moves the
   same debugging time later, with more layers stacked on top to unwind.
   *(Done.)*
