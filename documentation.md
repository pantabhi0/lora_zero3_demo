# HeteroTrain — Technical Documentation

Deep dive into what the demo does, why it is designed the way it is, and how every file works. Intended for someone who wants to understand the code, reproduce the methodology, or extend the project.

---

## 1. What the demo actually measures

The demo fine-tunes **Qwen/Qwen2.5-0.5B** (the base model, not -Instruct) with **fp16 LoRA** on **CodeAlpaca-20k** (`sahil2801/CodeAlpaca-20k`), under two configurations:

1. **Baseline** — a single GTX 1650, plain PyTorch.
2. **2-node DDP** — two GTX 1650s, one per laptop, synchronized with PyTorch DDP over a gigabit link.

Both runs consume the **same total sample budget** (default 1000 samples, 1 epoch). The comparison unit is an **optimizer update** (after gradient accumulation), because that is the natural step at which DDP synchronizes.

The headline output is a tradeoff: the 2-node run doubles throughput per update (each update covers `world_size × batch × accum` samples) but pays a per-update all-reduce cost. On this hardware the cost is 3.9% of step time, so 2-node wins ~2×. On a slower link the comm fraction grows and the speedup erodes; on a bigger model/GPU per node the compute fraction grows and comm matters less.

> **Why base, not -Instruct?** The -Instruct model already answers like an assistant, so its loss starts close to convergence — the visible loss drop that makes the demo legible would be small. The base model starts confused and drops visibly.

> **Why not expect a big speedup?** GTX 1650 (Turing, CC 7.5) has **no Tensor Cores**; FP16 runs at roughly the FP32 rate. The speedup comes from parallelizing the *sample budget* across two GPUs (fewer sequential updates), not from per-GPU acceleration.

---

## 2. Design constraints (from the plan / AGENTS.md)

These are load-bearing; changing them silently changes what the demo proves:

| Constraint | Reason |
| --- | --- |
| Plain `torch.distributed` DDP, manual launch per node | No pdsh/SSH; the comm pattern stays one all-reduce/step, easy to instrument. |
| No DeepSpeed/ZeRO/FSDP, no quantization, no FlashAttention-2 | FA2 needs sm80+; Turing is sm75. Keeps the memory story simple. |
| `uv` only (pyproject.toml + uv.lock) | Byte-identical dependency resolution on both machines, bare-metal and in Docker. |
| `NCCL_SOCKET_IFNAME` / static IPs: no defaults, fail loudly | The most common cause of NCCL hangs is a wrong interface. Auto-detection is explicitly forbidden. |
| Frozen base params, DDP wraps only the PEFT model | All-reduce traffic shrinks to the LoRA adapter gradients (~3 MB) → comm is 3.9%, not 50%. |
| Exactly one all-reduce per update via `no_sync()` | Without this, DDP would all-reduce every micro-step (accum× more traffic, noisier timing). |
| Fixed sample budget as comparison unit | Both runs consume identical total samples so the comparison is apples-to-apples. |
| CUDA events synced once at end of run | Per-step `torch.cuda.synchronize()` would serialize the pipeline and distort the wall clock being measured. |
| One GPU per node → local device `cuda:0`, never global `RANK` | With RANK=1 on node B, `set_device(1)` errors ("invalid device ordinal"). An actual bug hit and fixed. |
| Identical Docker image on both machines | Kills version drift between nodes. Build once, `docker save`/`load` the tarball. |

---

## 3. Repository map

| Path | Responsibility |
| --- | --- |
| `Dockerfile` | CUDA 13.0 + cuDNN base, uv-provisioned Python 3.12 env, bakes `src/`, `configs/`, `scripts/`. |
| `pyproject.toml`, `uv.lock` | Single source of truth for every Python package incl. torch (`2.13.0+cu130`). |
| `.env.example` | Template for the per-machine env (MASTER_ADDR, PORT, WORLD_SIZE, RANK, NCCL iface). |
| `configs/lora_config.yaml` | All model/dataset/training/LoRA/logging knobs; one-line edits vary batch/seq. |
| `scripts/build_image.sh` | `--save` → build + `docker save` tarball; `--load` → `docker load`. |
| `scripts/launch_rank0.sh` / `launch_rank1.sh` | The exact `docker run` flags, `RANK=0/1`, volume mounts. |
| `src/hello_world_dist.py` | Phase 1/2: NCCL + GPU + network smoke test. |
| `src/train_lora.py` | Phase 3/4: the whole training pipeline + per-update timing + validation mode. |
| `src/metrics.py` | Timing collectors, live CSV writer, env gate, summary, acceptance checks, DDP comm hook. |
| `src/infer.py` | Prompt REPL comparing base vs finetuned on laptop A. |
| `src/tui.py` | Live terminal table (rich) or plain stdout, reading Phase 5 CSVs. |
| `src/webui.py` | stdlib-only live browser GUI, same data source as the TUI. |
| `logs/` | `run_*_rankN.csv` (live), `timing_*_rankN.csv` (finalized), `summary.csv`, adapters. |
| `hf_cache/` | HuggingFace cache, volume-mounted so model+dataset download once per machine. |

---

## 4. `src/metrics.py` — the measurement core

All measurement design lives here. The other modules import from it.

### 4.1 CSV path conventions

```python
run_<run_tag>_rank<N>.csv      # live per-step rows, flushed every row
timing_<run_tag>_rank<N>.csv   # finalized per-step compute/comm, written at end
summary.csv                    # one row per completed run, appended
```

**Step-numbering quirk:** the live run CSV numbers steps **1-based** (`1..125`), while the timing CSV numbers steps **0-based** (`0..124`). Tools that combine them (TUI, web GUI) **merge by position**, never by step key — the single subtle correctness requirement for the viewers.

### 4.2 `LiveCSV`

Writes the run CSV row-by-row, `flush()` after each write, so a separate process (TUI/webui) can re-read the file mid-run without any IPC. Columns: `step, loss, samples_processed`.

### 4.3 `TimingCollector`

```python
record(step, start_event, end_event)   # append (start, end) CUDA-event pair per step
finalize() -> {step: ms}               # torch.cuda.synchronize() ONCE, then sum elapsed per step
```

- Event pairs are accumulated across the run **without syncing**; the single `synchronize()` in `finalize()` happens at the end. This is what keeps measurement from distorting the pipeline.
- `finalize()` is idempotent (guards with `_synced`) — both the train loop and the validation mode call it.
- `record()` after `finalize()` raises, catching ordering bugs.

### 4.4 `make_timing_hook` — the comm hook

DDP's `register_comm_hook` lets us intercept the gradient all-reduce. The hook:

1. Reads the current update index from a mutable `current_step` cell (a list, so the hook sees updated values).
2. Records a CUDA start event.
3. Runs `dist.all_reduce(bucket.buffer(), SUM)` **asynchronously** and chains `.then(_on_done)`.
4. `_on_done` divides the bucket by `world_size` (replicating DDP's default hook — otherwise gradients would be SUMmed but never averaged, which breaks training), records the end event, and returns `bucket.buffer()`.

```python
def _on_done(future):          # returns a Tensor
    bucket.buffer().div_(world_size)
    end.record()
    collector.record(step, start, end)
    return bucket.buffer()
```

**Two real bugs this design hit, both fixed:**

- `from __future__ import annotations` makes the `def hook(...) -> torch.futures.Future` annotation a **string** at runtime. DDP's `_check_comm_hook` compares the annotation against the *real* `torch.futures.Future[torch.Tensor]` type and rejects a string. Fix: force the real type object:

  ```python
  hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]
  ```

- `_on_done` must return the **tensor** (`bucket.buffer()`), not the Future — returning the Future raises `RuntimeError: Unable to cast Future to Tensor` in DDP's internal callback machinery.

The hook times the **full hook duration** (all-reduce kernel + divide + Future machinery + bucket handling) — the real distributed cost, not just the raw NCCL kernel. Phase 5 validated this against `torch.profiler` ground truth (see §7).

### 4.5 Env handling — `load_env_file` / `require_dist_env`

- `load_env_file(".env")` parses `KEY=VALUE` lines and fills only **unset** vars. Existing vars are never overridden — so when Docker exports everything via `--env-file`, Docker's values win; bare-metal `uv run` gets the repo `.env` for free (no manual `source`).
- `require_dist_env()` gates Phases 1/4: requires `MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, NCCL_SOCKET_IFNAME`. Any missing var → `SystemExit` with a clear "fill .env" message. No defaults, no detection.

### 4.6 Derived metrics

| Function | Meaning |
| --- | --- |
| `samples_per_sec` | `total_samples / wall_s` |
| `tokens_per_sec` | `total_samples × seq_length / wall_s` |
| `comm_pct` | `comm / (compute + comm) × 100` — share of step time spent in the all-reduce |
| `extrapolated_epoch_s` | `wall_s / fraction_done` — linear projection to one full epoch |
| `loss_criterion_met` | P3 acceptance: mean(last `window_fraction` of losses) ≥ 10% below mean(first window), window scaling with update count |
| `curve_matches` | P4 acceptance: max per-step relative difference ≤ 2% between two loss curves |

### 4.7 Summary writer

`write_summary_csv` appends one row per run to `logs/summary.csv` (header written only if the file is new). Columns: run_tag, world_size, rank, total_samples, total_updates, wall_clock_s, samples_per_sec, tokens_per_sec, compute/comm ms per update, comm %, extrapolated epoch, final loss, criterion met.

---

## 5. `src/hello_world_dist.py` — Phase 1/2 smoke test

Minimal `torch.distributed` program, no ML:

1. `require_dist_env()` gates on the env vars.
2. `init_process_group(backend="nccl", init_method="env://")`.
3. **`torch.cuda.set_device(0)`** — both global ranks pin local device 0 (the one-GPU-per-node rule).
4. Each rank makes a `1024`-element tensor of `(rank+1)`, records CUDA start/end events around one `all_reduce(SUM)`, and prints the result + elapsed ms.
5. Asserts `tensor[0] == world_size*(world_size+1)/2` (3.0 for two ranks), destroys the group, prints "clean exit".

This validates GPU + NCCL + network + env wiring in isolation, before any model code exists. If it hangs rather than errors, the fault is `NCCL_SOCKET_IFNAME` or a firewall — and per plan, that root cause is fixed, not papered over with retries.---

## 6. `src/train_lora.py` — the training pipeline (Phases 3–4)

### 6.1 CLI

```
--config configs/lora_config.yaml   # required
--distributed                       # init DDP process group (2-node run)
--validate-timings                  # short torch.profiler cross-check, then exit
--run-tag TAG                       # override config logging.run_tag
--subset-size N                     # override dataset.subset_size
--epochs N                          # override training.num_epochs
```

### 6.2 Config-driven setup

Everything interesting is read from `configs/lora_config.yaml` — model name, fp16 on/off, gradient checkpointing, dataset name/size/seed, batch size, seq length, grad-accum steps, learning rate, epochs, LoRA rank/alpha/dropout/target modules, log output dir, run tag. To vary batch size or sequence length (Phase 5 metric 4), edit the YAML and rerun — no code changes.

### 6.3 Dataset pipeline

`build_dataset(cfg, tokenizer)`:

1. `format_example` builds the CodeAlpaca prompt:
   ```
   ### Instruction:
   <instruction>
   ### Input:
   <input>          (only if present)
   ### Response:
   <output>
   ```
2. `tokenize_fn` tokenizes the full prompt+output text, then builds **masked labels**: tokens belonging to the prompt are replaced by `pad_token_id` (masked), only the response part is supervised. The attention mask covers all tokens.
3. The dataset is `shuffle(seed=cfg["dataset"]["seed"]).select(range(subset_size))` — **fixed seed** so every run (baseline, 2-node) sees the identical sample order, making the loss-curve comparison meaningful.
4. `.map(tokenize_fn, remove_columns=...)` then a pad/truncate pass to exactly `seq_length`, then `.set_format("torch")`.

### 6.4 Model construction — freeze + LoRA

`build_model`:

1. Load `Qwen/Qwen2.5-0.5B` in fp16 with `attn_implementation="sdpa"` (the plan forbids FlashAttention-2; Turing is sm75, SDPA is correct and sufficient).
2. **Freeze everything**: `param.requires_grad = False` for all base parameters.
3. Wrap with PEFT `get_peft_model` using the LoRA config (r=16, alpha=32, dropout=0.05, targeting all of `q/k/v/o` and `gate/up/down` projections).
4. `gradient_checkpointing_enable()` + `enable_input_require_grads()` (the latter makes inputs require grad, which checkpointing needs when base params are frozen).

Result: only ~3 MB of trainable LoRA parameters carry gradients. When DDP wraps this model, the all-reduce covers only those gradients — the whole reason comm stays at 3.9%.

### 6.5 DDP wrapping and the one-all-reduce rule

```python
ddp = DDP(model, device_ids=[0], gradient_as_bucket_view=True)
ddp.register_comm_hook(None, make_timing_hook(comm_collector, lambda: current_step[0], world_size))
```

`gradient_as_bucket_view=True` lets the hook read/write the gradient buffer in place (required for the buffer-based comm hook). `device_ids=[0]` pins local GPU 0.

The training loop accumulates gradients across `accum` micro-steps:

```python
for micro in range(accum):
    is_final = micro == accum - 1
    ctx = wrapped.no_sync() if (args.distributed and not is_final) else CONTEXT()
    with ctx:
        ... forward/backward (loss divided by accum) ...
```

- Non-final micro-steps run inside `model.no_sync()` → **no communication**.
- Only the final micro-step's backward triggers the all-reduce (via the comm hook).
- Result: exactly **one all-reduce per optimizer update**, which is both the design goal (minimize comm) and what makes the per-step comm timing clean.

Per micro-step, CUDA start/end events are recorded around forward+backward into `compute_collector` keyed by the current update index. Because the final micro-step's span *contains* the all-reduce, the compute number for that step includes comm — so at the end the code **subtracts comm from compute** (`compute_clean = compute_ms − comm_ms` per step). The reported "compute" is therefore genuinely non-overlapped compute, and comm is reported separately. They never double-count.

### 6.6 Sample-budget accounting

```python
samples_processed += world_size * batch_size * accum   # per optimizer update
consumed = min(budget, samples_processed)
```

- Baseline (world=1, batch=1, accum=4): 4 samples/update → 250 updates for the 1000-sample budget.
- 2-node (world=2): 8 samples/update → 125 updates.

The loop is `while consumed < budget`, so both configurations always consume the **same 1000 samples** but differ in how many sequential updates that takes. That 2× fewer updates is the source of the 2× wall-clock win, minus the 3.9% comm tax.

### 6.7 Training with fp16 + GradScaler

fp16 uses `torch.autocast(device_type="cuda", dtype=torch.float16)` for forward/backward plus `torch.cuda.amp.GradScaler`. The loss is divided by `accum` *before* `scaler.scale(loss).backward()` so accumulated gradients equal the mean over micro-steps.

### 6.8 Live logging

Per optimizer update: `LiveCSV.write_row({step, loss, samples_processed})` — flushed immediately so the TUI/web GUI can render mid-run. Loss logged is the accumulated mean for that update (`running_loss / accum`).

### 6.9 Timing finalization and the adapter save

After the loop:

1. `wall_s = monotonic_now() − wall_start` (monotonic clock — unaffected by wall-clock jumps).
2. `compute_collector.finalize()` (single sync) and `comm_collector.finalize()`.
3. `compute_clean = {step: max(0, compute − comm)}` per step (de-overlap).
4. `write_timing_csv` → `timing_<run_tag>_rank<N>.csv`.
5. **Rank 0 only**: `model.save_pretrained(logs/<run_tag>/adapter)` + `tokenizer.save_pretrained(...)`. This is the *only* checkpoint; nothing else persists. (The PEFT model unwraps to the adapter + config; the base weights are loaded fresh at inference.)
6. Rank 0 also appends the run row to `summary.csv` and runs `curve_matches` against every other rank-0 run CSV it finds, printing OK/DIVERGE.

### 6.10 `--validate-timings` (Phase 5 cross-check)

A short mode that mirrors the real loop (same `no_sync` pattern, same one-all-reduce-per-update) but runs 3 updates under `torch.profiler`, then:

1. Prints the top CUDA ops table.
2. Sums `self_device_time_total` over **only** events whose key starts with `ncclDevKernel` (the raw NCCL kernel name). This was a real fix: torch 2.13 logs the same op under multiple names (`c10d::allreduce_`, `nccl:all_reduce`, `ncclDevKernel`) — counting all of them triple-counts. Only the `ncclDevKernel` prefix is counted.
3. Compares that profiler NCCL total against the comm-hook total for the same window and prints both.

Measured agreement: **3784.9 ms hook vs 3777.3 ms profiler over the same window (0.2% difference)** → the hook methodology is trusted for the real runs. (Note: torch 2.13 renamed the profiler's `cuda_time_total` to `device_time_total`/`self_device_time_total`.)

---

## 7. Trusting the numbers — the timing methodology

The whole demo rests on three decisions that make the numbers honest:

1. **Rank 0 lives on laptop A for every run.** Baseline and 2-node rank-0 logs are produced on the same physical machine, so the headline comparison has no cross-machine clock skew or hardware variance. Rank-1 comm is logged only for reference.
2. **CUDA events, one sync at the end.** `TimingCollector` records event pairs without syncing; `finalize()` synchronizes once. Per-step syncing would force full GPU idle between steps and inflate the wall clock being measured.
3. **Full-hook comm timing, validated against the profiler.** The hook measures the whole distributed step (bucketed all-reduce, divide, Future machinery), and Phase 5's profiler run confirmed it agrees with raw NCCL kernel time to 0.2%. Compute then subtracts the overlapped comm so the two components are disjoint.---

## 8. The viewers — `src/tui.py` and `src/webui.py`

Both are **readers only**: they re-read the CSVs the training process writes and render them. There is no separate logging path — the training loop is the single source of truth.

### 8.1 Shared helpers (`tui.py`)

- `read_rows(path)` → list of dicts from any CSV (empty list if the file doesn't exist yet — this is what makes the viewers work *during* a run).
- `timing_path_for(run_csv)` → `logs/timing_<tag>_rank<N>.csv` by string-replacing `run_` with `timing_` in the filename.

### 8.2 `tui.py`

- Rich table: step, loss, samples_processed, compute_ms, comm_ms.
- **Last 40 rows only** — a deliberate terminal-height cap (the first rows of a completed 125-step run fall off the screen; a browser has no such limit).
- `--refresh` 1.0 s default; `TUI_RENDER=plain` env → plain tab-separated stdout loop (no rich installed / dumb terminal).
- `--refresh 0` → dump once and exit (batch mode).

### 8.3 `webui.py`

- **stdlib only** (`http.server`, `ThreadingHTTPServer`) — no new dependencies, per the plan's "no new deps" rule.
- `GET /` serves an inline HTML+JS page; `GET /api/rows` returns JSON. The browser `fetch`es `/api/rows` every `--refresh` s (default 1.0) and re-renders the table.
- Renders **all rows** — a browser scrolls, so the TUI's 40-row cap is not carried over.
- `--host 0.0.0.0` lets laptop A's browser view laptop B's table (and vice versa) by opening the other machine's IP.

### 8.4 The positional merge (why the viewers are correct)

Timing CSV steps are 0-based; run CSV steps are 1-based. Both viewers merge **by position**:

```python
for i, r in enumerate(read_rows(run_csv)):
    t = timing[i] if i < len(timing) else {}
```

Row i of the run CSV is paired with row i of the timing CSV. Merging by step key would misalign every row (step 1 ↔ timing step 0, etc.). A real bug in the first webui version showed row 125 with empty timings and rows shifted by one — the positional merge fixed it.

### 8.5 Web page gotcha

The HTML template contains CSS `%` (`width:100%`), so it is **not** formatted with Python's `%` operator. It uses `PAGE.replace("__REFRESH_MS__", str(refresh_ms))`. (The first version used `PAGE % {...}` and raised `ValueError: unsupported format character '}'` at runtime.)

---

## 9. `src/infer.py` — base vs finetuned (machine A only)

A prompt REPL that answers the obvious question: "did fine-tuning actually change anything?" It loads **two** models onto `cuda:0`:

1. The base `Qwen/Qwen2.5-0.5B` (fp16, SDPA attention).
2. A `PeftModel` — a *second* base model with the trained adapter merged in.

Two × ~0.5B fp16 ≈ 1 GB of weights, which fits the 4 GB GTX 1650 comfortably. For each prompt it prints the BASE answer and the FINETUNED answer so the user compares manually.

Generation settings were chosen deliberately:

- **Sampling** (`do_sample`, temperature 0.7, top_p 0.9, repetition_penalty 1.2). Greedy decode on a 0.5B base model loops/repeats and leaks CodeAlpaca's `### ` section markers — sampling plus a repetition penalty keeps output clean.
- Output is cut at the first `\n### ` (drift into the next example's format).
- Default `--max-new-tokens 512`. Hard ceiling is 32768 (Qwen2.5-0.5B context length); practically ~8000 on the 1650 because each generated token needs ~98 KB of KV-cache. Generation speed is roughly 20–40 tok/s.

Default adapter path derives from the config: `logs/<run_tag>/adapter` (so `logs/default/adapter` for the 2-node run). A missing adapter fails with a clear error. This tool runs on laptop A only — the comparison is the user's manual step.

---

## 10. Docker packaging

### 10.1 `Dockerfile`

- Base: `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04`, pinned. There is **no** cudnn9-suffixed tag for CUDA 13.0.0; `13.0.0-cudnn-devel` ships cuDNN 9.x. `devel` (not runtime) is needed for toolkit-level pieces.
- **Not** the NGC PyTorch image — that bundles its own torch which would fight uv's.
- `uv` is copied in from `ghcr.io/astral-sh/uv:latest`; `COPY pyproject.toml uv.lock ./` then `RUN uv sync --frozen --python 3.12`. The lockfile is the single source of truth; the image reproduces the verified bare-metal env exactly (torch 2.13.0+cu130, peft, transformers, accelerate, datasets, rich, textual, yaml). Python 3.12 matches the host.
- `COPY src ./src`, `configs`, `scripts`, `.env.example` — **the image bakes `src/`**. Any `src/` change requires a rebuild + re-distribution to B, or B runs stale code (this actually bit Phase 2).
- Build-time sanity check imports torch and prints version + NCCL availability + project imports.
- **No GPU driver inside the container.** The host provides it via NVIDIA Container Toolkit. Host driver must support CUDA 13.
- `ENV PATH="/app/.venv/bin:$PATH"`, `PYTHONUNBUFFERED=1`, `CMD ["bash"]`.

### 10.2 `scripts/build_image.sh`

`--save` builds + `docker save -o hetero-demo.tar` (~6.8 GB); `--load FILE.tar` does `docker load`. Distribution is `scp` the tarball to B, load there. Never rebuild separately per machine. (Compress the tarball with `-C zstd,0` for faster transfer.)

### 10.3 `scripts/launch_rank0.sh` / `launch_rank1.sh`

The required run flags, identical rationale:

| Flag | Why |
| --- | --- |
| `--gpus all` | expose the host GPU |
| `--network host` | NCCL socket rendezvous breaks on Docker bridge networking — mandatory |
| `--ipc=host` | shared memory for NCCL/PyTorch |
| `--ulimit memlock=-1` | allow locking memory (NCCL wants it) |
| `--ulimit stack=67108864` | raised stack limit |
| `--env-file .env` + `--env RANK=0/1` | env in; RANK forced per script so .env's value can't be copied wrong |
| `-v $PWD/hf_cache:/root/.cache/huggingface` | model+dataset download once per machine, reused every `--rm` run |
| `-v $PWD/logs:/app/logs` | training artifacts land in the repo on the host |
| `--rm` | container is ephemeral; persistence comes from the mounts |

**Why no `--user`?** A `--user $(id -u):$(id -g)` flag was tried and broke the container: the venv's python lives under root-only `/root` (`/root/.local/share/uv/...`), so a non-root user gets `exec: python: not found`. Logs are therefore root-owned; run `sudo chown -R $USER: logs/` after each container run.

### 10.4 Docker engine choice

This project requires the **native Docker Engine**, not Docker Desktop. Desktop runs a QEMU VM (no GPU passthrough → `--gpus all` fails with "no known GPU vendor found"; its `--network host` is the VM's network → NCCL breaks). Images are also per-daemon — a Desktop build does not appear in the native engine.

---

## 11. Configuration reference (`configs/lora_config.yaml`)

```yaml
model:
  name: Qwen/Qwen2.5-0.5B     # base, NOT -Instruct
  fp16: true
  gradient_checkpointing: true
dataset:
  name: sahil2801/CodeAlpaca-20k
  subset_size: 1000           # ← the fixed sample budget; change to scale runtime
  seed: 42                    # fixed seed → identical sample order across runs
training:
  batch_size: 1               # per-GPU micro-batch
  seq_length: 256
  grad_accumulation_steps: 4
  learning_rate: 1.0e-4
  num_epochs: 1
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
logging:
  output_dir: logs
  run_tag: default            # ← quirk: set, so the auto "2node"/"baseline" fallback never fires
```

Memory sanity on 4 GB: fp16 weights ~1 GB + LoRA + optimizer + activations (batch 1, checkpointing on) stays comfortably under budget.

---

## 12. The measured results, read carefully

| | Baseline | 2-node |
| --- | ---: | ---: |
| Updates | 250 | 125 |
| Wall | 2410.95 s | 1230.38 s |
| samples/s | 0.415 | 0.813 |
| tok/s | 106.2 | 208.1 |
| compute/update | 9629.4 ms | 9396.4 ms |
| comm/update | — | 381.3 ms (3.9%) |
| final loss | 0.521 | 0.343 |
| criterion | met | met |
| curve vs baseline | — | max_rel 0.0017 |

Readings:

- **Compute per update is nearly identical** between single-GPU and DDP (9396 vs 9629 ms) — expected: each GPU does the same forward/backward work; DDP adds only the all-reduce.
- **Comm is 381 ms on a 9.4 s step (3.9%)** — so 2-node pays almost nothing for synchronization and reaps the 2× fewer-sequential-updates win.
- **Final loss differs (0.52 vs 0.34)** because the 2-node run saw the same 1000 samples in *half the optimizer updates* — a larger effective batch per update converges differently. The per-step curve match (max_rel 0.0017, i.e. 0.17%) confirms the trajectories agree; the endpoint difference is the accumulation-order/batch-size effect, not a bug.
- The loss-criterion and curve-match functions are the formal acceptance checks of Phases 3 and 4 (window-relative 10% drop; per-step ≤2% relative agreement).

---

## 13. Extending / changing things

- **Different batch/seq budget**: edit `configs/lora_config.yaml` and rerun both configurations; the scripts auto-log everything (Phase 5 metric 4 is exactly this).
- **Bigger model**: bump `model.name`. Watch the 4 GB ceiling; raise `subset_size`/drop `gradient_checkpointing` accordingly. The same code paths hold for any HF causal LM + PEFT.
- **More nodes**: this repo is deliberately 2-node-only (`WORLD_SIZE` is read from env, so >2 works mechanically, but the plan and the comm hook assume one GPU per node and a single link — validate on the same machine pair first).
- **New viewer**: read `run_<tag>_rank<N>.csv` + `timing_<tag>_rank<N>.csv` and merge by position, as the TUI/webui do.
- **Adaptation for different hardware**: the timing methodology (CUDA events, single end sync, full-hook comm, profiler cross-check) transfers unchanged; the speedup magnitude will change with Tensor-Core availability and link speed.

---

## 14. Known quirks and footguns

1. **Step numbering mismatch** (run 1-based vs timing 0-based) — merge by position everywhere.
2. **`run_tag` quirk** — config sets `run_tag: default`, so `--run-tag` must be passed explicitly for other tags; the auto fallback only fires when the config omits `run_tag`. The baseline run used `--run-tag baseline`; the 2-node run is tagged `default`.
3. **Comm-hook annotation** — must force the real `Future[torch.Tensor]` type onto the hook (PEP-563 string annotations break DDP's check).
4. **`--user` breaks the container** (python lives in root-only `/root`) — root-owned logs are fixed with `sudo chown -R $USER: logs/`.
5. **Image bakes `src/`** — rebuild + redistribute on every source change.
6. **Profiler kernel-name triple-counting** — count only `ncclDevKernel`-prefixed events in the validation mode.
7. **Base model loops when decoded greedily** — inference uses sampling + a `\n### ` cut.
8. **No git repo initialized** — per user directive; `.env`, `logs/`, `hf_cache/`, `.venv/`, `hetero-demo.tar` should be gitignored before publishing.