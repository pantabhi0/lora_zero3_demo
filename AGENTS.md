# AGENTS.md — HeteroTrain 2-Node LoRA Demo

## Session rules
- Before starting any long-running command (training, big transfer, build):
  tell the user, ask for confirmation, and give progress updates during it —
  never leave the user hanging on a silent long process.

Goal: fine-tune Qwen/Qwen2.5-0.5B (BASE, not -Instruct) with fp16 LoRA:
single GTX 1650 vs 2 laptops via PyTorch DDP over 1Gbps. Quantify the
compute/comm tradeoff — result may be faster OR slower; do not frame as a
guaranteed speedup (no Tensor Cores on GTX 1650; FP16 ≈ FP32 rate).

## Read first
- `gigabit-lora-demo-plan.md` = source of truth. Follow phases -1→6 IN ORDER.
  Each phase has an acceptance check; never combine/skip phases.
- Web GUI (Phase 6) only after Phases 0–4 pass. Descope order: web GUI →
  Phase 5 metric 4 extra runs → Docker. Never cut: Phase 3 baseline, Phase 4
  2-node run, Phase 5 metrics 1–3, Phase 0/1 network validation.

## Current state (update as phases pass)
- Phase 0 (network/IPs) PASS: A=192.168.50.1/24 (`eno1`, r8169), B=192.168.50.2/24
  (`enp44s0`). Gigabit was forced on A with `sudo ethtool -s eno1 speed 1000
  duplex full autoneg on` (link partner originally advertised 10/100 — cable
  was the suspect). Verify with ethtool Speed = 1000Mb/s both sides; A's
  prompt username is `abhi`, B is `james@pixelpilot-Nitro-AN515-56`.
- Phase 1 (bare-metal hello) PASS: both ranks print `all_reduce ok:
  tensor[0]=3.0` + clean exit. Phase 2 (dockerized hello) PASS: same result
  inside containers (`--network host`, `--env-file .env`).
- Historical Phase 3/4 artifacts (`baseline`/`default`) are retained for
  debugging only and have no academic meaning. Academic results use the four
  tags documented under `## Academic campaign` below.
- Phase 5 validation PASS: torch.profiler cross-check run — comm-hook wall
  time 3785ms vs profiler raw NCCL kernel time 3777ms over the same window
  (0.2% agreement) → hook methodology trusted. (Validate-loop bug found: the
  single-pass `range(accum*3)` with `micro == accum-1` fired once not 3×;
  fixed to mirror the real loop with no_sync — next image build has it; the
  current image b20dbdcf3028 still has the old validate code, not re-run.)
  Phase 5 metric 4 (extra batch/accum runs) still OPTIONAL per descope order.
- Phase 6 WebUI PASS: legacy CSV viewer, academic pair dashboard, static PNG
  serving, ZIP export, Academic Summary table, and A-only inference tab tested.
  `src/tui.py` remains terminal fallback.
- Logs from container runs are root-owned — `sudo chown -R $USER: logs/` after
  each run before reading/infer.

## Academic campaign (new work)
- Historical `baseline` and `default` artifacts have no academic meaning.
  Academic conclusions use exactly four new tags only:
  `baseline_1k`, `2node_1k`, `baseline_long`, `2node_long`.
- Model remains `Qwen/Qwen2.5-0.5B` base. New campaign uses correct causal-LM
  prompt masking (`labels=-100` before response tokens), a fixed disjoint
  held-out split of 100 examples, seed 42, and rank-disjoint training data.
- Reserve validation examples after one fixed seed shuffle, before selecting
  training subset. Validation runs every 10% of updates plus final update;
  both ranks evaluate, synchronize with barrier, and rank 0 writes authoritative
  validation CSV. Token-weighted loss; safe perplexity.
- New metrics are independent of validated CUDA event timing: nvidia-smi at 1Hz,
  peak VRAM, loader+H2D data time, explicit other residual, validation time,
  metadata/config snapshots, and effective LoRA payload bandwidth estimate.
  Telemetry preflight happens before NCCL init; unavailable nvidia-smi fails run.
- Existing CUDA TimingCollector/make_timing_hook behavior remains unchanged:
  events synchronize once at end-of-run, never per-step; one all-reduce/update.
- `academic_summary.csv` is source for new analysis. Historical `summary.csv`
  is excluded. Interrupted runs leave partial artifacts and `.training_active`;
  next run refuses until partial state is inspected.
- Run order: baseline_1k acceptance first, 2node_1k acceptance second, derive
  long subset from measured baseline_1k end-to-end rate, then baseline_long and
  2node_long with identical sample count. No fifth calibration run.
- Academic pair comparisons align validation/training curves by `tokens_seen`,
  never raw optimizer step: baseline_1k has 250 updates while 2node_1k has
  125 updates for same sample budget.
- During long training runs, after confirming startup, poll artifacts/logs no
  more often than every 15 minutes unless drastic change or an error requires
  immediate inspection. Do not repeatedly poll active run state.
- Resolved DDP configs live in `configs/resolved/`, are host-mounted into
  containers, and must be hash-checked on both ranks. `scripts/prepare_2node.sh`
  selects a completed baseline and prints the required scp/launch commands.
- Academic campaign complete:
  - `baseline_1k`: 1000 samples, 3286.125s, 26.807 tok/s, final val loss
    0.606460, final val PPL 1.833927.
  - `2node_1k`: 1000 samples, 1668.567s, 52.795 tok/s, comm=283.488ms/update
    (2.87%), final val loss 0.572788, final val PPL 1.773204. Speedup=1.969x.
  - `baseline_long`: 3288 samples, 8841.677s, 31.929 tok/s, final val loss
    0.567638, final val PPL 1.764096.
  - `2node_long`: 3288 samples, 4475.444s, 63.080 tok/s, comm=287.709ms/update
    (2.93%), final val loss 0.565360, final val PPL 1.760081. Speedup=1.976x.
  - Academic conclusions use only these four tags. Static figures live under
    `logs/analysis/`; final measured report is `report.md`.

## Already done (user) — do NOT duplicate
- Ubuntu 26.04 + NVIDIA driver installed+verified on both (nvidia-smi).
  No driver-install steps. Physical gigabit link wired; static IPs +
  interface names assigned (Phase 0).
- User-only: `prime-select nvidia`, AC/perf power mode, Docker + NVIDIA
  Container Toolkit install, bidirectional ping validation, filling .env.
  Agent does NOT write driver/network-setup steps or auto-detect anything.
- Docker daemon is the NATIVE engine, systemd `docker.service` (disabled,
  socket-activated — starts on first CLI use; no Desktop). Toolkit runtime
  `nvidia` already registered in daemon.json. Do NOT reinstall/configure.

## Hard rules
- No git commit/push unless user explicitly requests it. Read-only git checks
  are allowed when needed to protect worktree state.
- `uv` only (uv add / uv sync). pyproject.toml + uv.lock. Never pip, never
  requirements.txt — bare-metal AND in Dockerfile.
- NCCL_SOCKET_IFNAME, MASTER_ADDR, static IPs: required config, NO default.
  Read from .env (user fills). If unset → fail with clear error. Never
  auto-detect. Env: MASTER_ADDR=<laptop A IP>, MASTER_PORT, WORLD_SIZE=2,
  RANK(0=A,1=B), NCCL_SOCKET_IFNAME (per-machine, may differ).
- `.env` auto-loader in `metrics.require_dist_env()` (`load_env_file()`):
  fills UNSET vars from repo-local `.env`; existing vars never overridden,
  so docker `--env-file` wins. Bare-metal `uv run` needs no manual `source`.
- One GPU per node → CUDA device is `cuda:0`, NEVER the global RANK. Both
  `hello_world_dist.py` and `train_lora.py` use `torch.cuda.set_device(0)` +
  `device_ids=[0]`. (Bug hit on B: `set_device(rank)` with RANK=1 → "invalid
  device ordinal".) B's repo/src syncs from A via `scp -r src/
  james@192.168.50.2:~/lora_zero3_demo/`.
- Model: Qwen/Qwen2.5-0.5B base. NOT -Instruct (instruct starts too close to
  convergence; want visible loss drop).
- Dataset: sahil2801/CodeAlpaca-20k via load_dataset. Fixed seed shuffle.
  Default subset = 1000 samples, 1 epoch; configurable count.
- fp16 LoRA only. No quantization, no DeepSpeed/ZeRO/FSDP, no FA2 (Turing
  sm80+ only; use default/SDPA). Plain DDP, 2 nodes, one all-reduce/step.
- Freeze base params (no requires_grad) so DDP all-reduces ONLY LoRA adapter
  grads. Wrap the PEFT model in DDP.
- batch_size, seq_length, grad_accumulation_steps all config-driven
  (configs/lora_config.yaml), one-line edit to vary. No sweep/auto-vary.
- DDP + grad accum: wrap non-final micro-steps in model.no_sync() — exactly
  ONE all-reduce per optimizer update.
- Comparison unit: fixed SAMPLE BUDGET (default 1000), identical across all
  runs regardless of batch/accum/rank split. Atomic logged unit = optimizer
  update. Derive per-step from logs afterward; do not hand-match configs.
- Launch: plain torch.distributed, manual docker run per node. No pdsh/SSH.

## Docker
- Base: `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` (pinned exact tag). No
  cudnn9-suffixed tag exists for CUDA 13.0.0 — `13.0.0-cudnn-devel` ships
  cuDNN 9.x. NO NGC pytorch base (bundles its own torch that fights uv's).
  devel (not runtime) image: needed for toolkit-level pieces.
- uv.lock = single source of truth for ALL Python pkgs incl. torch
  (2.13.0+cu130), peft, transformers, accelerate. Base image provides only
  CUDA userspace/cuDNN/NCCL + system libs. `uv sync --frozen --python 3.12`
  (Python 3.12 matches verified host env). No strip workaround — torch comes
  from PyPI wheel like bare-metal.
- Host NVIDIA driver is provided by host via NVIDIA Container Toolkit; never
  install a driver inside the container. Host driver must support CUDA 13.
- Build ONCE, distribute identical image to both machines. Never rebuild
  per-machine. `scripts/build_image.sh` builds/tags; `--save` writes a
  tarball, `--load FILE.tar` loads it — transfer tarball to machine B,
  `docker load`, no rebuild.
- Container verification (single-GPU, run on target): `docker run --rm
  --gpus all <image> python -c "import torch; print(torch.__version__,
  torch.version.cuda, torch.cuda.is_available())"` must print `2.13.0+cu130
  13.0 True`; then `get_device_name(0)`/`get_device_capability(0)` →
  `NVIDIA GeForce GTX 1650` / `(7, 5)`; then
  `torch.distributed.is_nccl_available()` → True; then import
  torch/transformers/peft/accelerate.
- Run flags (required): `--gpus all --network host --ipc=host
  --ulimit memlock=-1 --ulimit stack=67108864`. --network host mandatory
  (NCCL socket rendezvous breaks on bridge). Env via -e / env file, same
  values as Phase 0. Logs from container runs are root-owned — `sudo chown -R
  $USER: logs/` after each run before reading/infer. (A `--user` flag breaks
  the container: the venv's python lives under root-only /root; `--user` →
  "exec: python: not found". Only fix is baking a uv python install dir into
  the image — deferred.)
- HF cache is VOLUME-MOUNTED at repo `hf_cache/` → `/root/.cache/huggingface`
  in both launch scripts: model+dataset download ONCE per machine, reused on
  every run (container is `--rm`, so without the mount it re-downloads each
  time). Cache is root-owned — fine, only the container reads it. Image stays
  lean; no pre-baked model/dataset.
- Docker engine choice: use NATIVE Docker Engine (context `default`,
  /var/run/docker.sock), NOT Docker Desktop. Desktop runs a QEMU VM with no
  GPU passthrough (`--gpus all` fails: "no known GPU vendor found") and its
  --network host is the VM's, breaking NCCL. Native engine needs NVIDIA
  Container Toolkit configured (`nvidia-ctk runtime configure --runtime=docker`
  + docker restart) or `--gpus all` fails the same way. Toolkit install is a
  user-only step (needs sudo). Verify with `docker info | grep -i runtime`
  showing `nvidia`, then `docker run --rm --gpus all <image> nvidia-smi`.
- Images are per-daemon: a build inside Desktop's VM does NOT appear in the
  native engine and is lost when Desktop stops. Rebuild via scripts/build_image.sh.
- The image BAKES in `src/` — whenever `src/` changes, rebuild the image,
  `docker save`/`--load` it to B again, or B runs stale code. (Phase 2 hit
  this: B's container had the old `set_device(rank)` bug.) Quick local image
  rebuild uses the cached layers (~8s), then re-export `hetero-demo.tar`.

## Metrics / timing (Phase 3–5)
- Per optimizer update, log: compute time (summed across micro-steps) and
  comm time. Comm via `ddp.register_comm_hook` timing the FULL hook duration
  (bucketed all-reduce + flatten/copy — real distributed cost; count it).
  CUDA events accumulated, synchronized ONCE at end-of-run — never per-step
  (would distort the wall-clock being measured).
- One torch.profiler-tagged validation run cross-checks hook numbers before
  Phase 5 logs are trusted.
- CSV in logs/: total wall-clock, samples/sec, per-update compute + comm,
  tokens/sec, extrapolated epoch time, comm% of step, final loss. Timings
  finalized at run end; per-step loss/step flushed live.
- Laptop A (rank 0) = authoritative measurement point for the headline
  comparison: baseline and 2-node rank 0 both run there → no cross-machine
  clock skew / hardware variance. Comm also logged on rank 1 for reference.

## Phase acceptance (sharpened)
- P3: last-10% of updates mean ≥10% below first-10% mean (window scales with
  step count; threshold adjustable if budget grows). Single-GPU baseline on
  laptop A only.
- P4: loss curves must agree within ~1–2% relative per step. Bit-level
  divergence is EXPECTED (all-reduce summation-order non-associativity);
  beyond-tolerance divergence = bug. Same sample budget + seed as baseline.
- P1 hang at init = NCCL_SOCKET_IFNAME wrong or firewall; fix root cause,
  no retry/timeout paper-over.

## Phase 6 UI (web GUI)
- Change of plan (replaces the TUI): simple stdlib-only web GUI, `src/webui.py`
  (http.server + inline HTML/JS, NO new deps). Feature-equivalent to the TUI:
  per-step loss / samples_processed / compute_ms / comm_ms table, browser
  polls `/api/rows` every `--refresh` s (default 1.0), all rows shown.
- One server per machine, same split as the TUI: A=rank0, B=rank1, side-by-side
  in browsers. `src/tui.py` still exists as a plain-stdout fallback.
- Table shows ALL rows (TUI's last-40 was a terminal-height cap; browsers
  scroll). Timing CSV steps are 0-based, run CSV 1-based — merge by POSITION,
  never by step key (webui.py and tui.py both do this).
- Demo: generate figures with `uv run python -m src.analyze --scale all`, then
  run `uv run python -m src.webui --pair 1k --analysis-dir logs/analysis` on A.
  Legacy `--csv` mode remains available for one rank's table.
- Phase 6 PASS: legacy viewer, pair dashboard, PNG serving, Academic Summary,
  ZIP export, and A-only inference tab tested. TUI remains fallback.
- Academic extension: `src/analyze.py` generates measured-only static PNGs;
  `src/webui.py --pair 1k --inference` serves pair figures and A-only inference
  tab. Inference loads one selected adapter at a time, toggles adapter layers
  for BASE/FINETUNED output, uses same seed, safe local rendering, and blocks
  while `.training_active` exists.

## Structure (from plan)
Dockerfile, pyproject.toml, uv.lock, .env.example, configs/{lora_config.yaml,
resolved/}, scripts/{build_image.sh,prepare_2node.sh,launch_rank0.sh,
launch_rank1.sh}, src/{hello_world_dist.py,train_lora.py,metrics.py,analyze.py,
infer.py,tui.py,webui.py,web/}, logs/.

## Inference UI (machine A only) — user does the prompting
- After a training run, the rank-0 adapter is saved to
  `logs/<run_tag>/adapter` (`model.save_pretrained` + tokenizer). This is the
  ONLY checkpoint; nothing else persists.
- `uv run python -m src.infer` loads base Qwen2.5-0.5B AND the trained
  adapter (two models, ~2×0.5B fp16 — fits the 4GB 1650), REPL, prints the
  BASE and FINETUNED answers for each prompt. `--adapter` overrides the path,
  `--max-new-tokens` sets length. Runs on A only; comparison is the user's
  manual step — the agent only provides the interface.
- WebUI inference lists available academic adapters and loads one selected
  adapter at a time; standalone inference accepts an explicit adapter path.
- Generation is SAMPLING (do_sample, temp 0.7, top_p 0.9, repetition_penalty
  1.2) — greedy decode on the 0.5B loops/repeats and leaks CodeAlpaca "### "
  sections; output is cut at the first "\n### " drift. Default max-new-tokens=512.

## You still run manually
- Install Docker + NVIDIA Container Toolkit both machines (script it once,
  run identically both).
- Fill .env per-machine (MASTER_ADDR, MASTER_PORT, NCCL_SOCKET_IFNAME, RANK).
- `sudo prime-select nvidia` + reboot + AC/perf power mode both.
- Phase 0 pings. Then: docker run per node (scripts/launch_rank*.sh).
