# HeteroTrain Technical Documentation

This document explains current source behavior and the completed academic
campaign. It supersedes descriptions of the historical `baseline` and
`default` runs. Academic analysis uses only:

```text
baseline_1k
2node_1k
baseline_long
2node_long
```

Historical artifacts remain useful for debugging but have no academic meaning.

## 1. Architecture

```text
configs/lora_config.yaml
        |
        v
src/train_lora.py ----> logs/run_<tag>_rank<N>.csv
        |                logs/timing_<tag>_rank<N>.csv
        |                logs/validation_<tag>_rank0.csv
        |                logs/gpu_<tag>_rank<N>.csv
        |                logs/<tag>/{adapter,config,metadata}
        |
        +--> src/metrics.py
        |       CUDA event timing
        |       DDP comm hook
        |       telemetry
        |       summary writer
        |
        +--> src/analyze.py --> logs/analysis/*.png/*.md
        |
        +--> src/webui.py --> metrics dashboard, figures, inference page
```

The application is run in a Docker container. The image contains source and
Python dependencies. Host volumes provide:

```text
hf_cache/ -> /root/.cache/huggingface
logs/     -> /app/logs
configs/  -> /app/configs
```

The `configs/` mount is important: resolved paired configs can change without
rebuilding the image. Source changes still require rebuilding because `src/`
is copied into the image.

## 2. Academic Experiment Contract

### 2.1 Model and optimization

- Base model: `Qwen/Qwen2.5-0.5B`.
- fp16 autocast and GradScaler.
- SDPA attention.
- Base parameters frozen.
- LoRA applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`.
- LoRA rank 16, alpha 32, dropout 0.05.
- AdamW learning rate `1e-4`.
- Batch size 1, sequence length 256, accumulation 4.
- Gradient checkpointing enabled.

### 2.2 Dataset split

`build_datasets()` loads CodeAlpaca-20k and performs exactly one deterministic
shuffle with seed 42:

```text
shuffled[0:100]       -> held-out validation
shuffled[100:]        -> training pool
training_pool[0:N]    -> requested training subset
```

Prompt formatting:

```text
### Instruction:
<instruction>
### Input:
<input>                 optional
### Response:
<output>
```

Label semantics:

```python
labels = [-100] * len(full_ids)
labels[len(prompt_ids):] = full_ids[len(prompt_ids):]
```

Transformers ignores `-100` labels. Consequently, validation and training
loss measure response-token prediction rather than prompt formatting tokens.
This is intentionally different from historical runs, which used invalid
pad-token prompt labels.

### 2.3 Rank-disjoint training data

The shuffled training subset is partitioned by rank with deterministic striding:

```python
indices = list(range(rank, len(dataset), world_size))
```

This prevents both DDP ranks from training on the same examples while the
sample counter claims global progress.

Validation is also partitioned by rank. Both ranks evaluate their local share,
then reduce summed loss and token count. Rank 0 writes the authoritative
validation CSV and both ranks enter a barrier before training continues.

### 2.4 Fixed budgets

The loop counts global samples per optimizer update:

```python
samples_processed += world_size * batch_size * accumulation
```

The four runs use:

| Tag | World | Subset | Updates |
| --- | ---: | ---: | ---: |
| `baseline_1k` | 1 | 1000 | 250 |
| `2node_1k` | 2 | 1000 | 125 |
| `baseline_long` | 1 | 3288 | 822 |
| `2node_long` | 2 | 3288 | 411 |

The long subset 3288 was derived from measured `baseline_1k` end-to-end rate
and rounded to a value divisible by both global update sizes. No fifth
calibration run was performed.

## 3. `src/metrics.py`

### 3.1 Existing CUDA timing

`TimingCollector.record()` stores CUDA event pairs by optimizer update. It does
not synchronize. `TimingCollector.finalize()` performs one
`torch.cuda.synchronize()`, then calls `start.elapsed_time(end)` for all stored
pairs.

This preserves the verified rule:

```text
accumulate CUDA events during training
synchronize once after the run
never synchronize or query elapsed time per update
```

The WebUI cannot show finalized compute/comm values during an active run. It
shows live loss and samples immediately, while timing columns remain empty
until `timing_*.csv` is written at the end. This is intentional.

### 3.2 DDP comm hook

`make_timing_hook()` performs:

1. CUDA start event.
2. `dist.all_reduce(bucket.buffer(), async_op=True)`.
3. Future callback after all-reduce completion.
4. Divide by world size to reproduce averaged-gradient semantics.
5. CUDA end event.
6. Store event pair.
7. Return `bucket.buffer()` from the callback.

The outer hook returns `fut.then(_on_done)`, which is a `Future[Tensor]` as
required by DDP. Returning a Future from `_on_done` would create a nested
Future and caused the historical `Unable to cast Future to Tensor` failure.

The hook's timing was cross-checked against profiler NCCL kernels with 0.2%
agreement. The hook itself remains unchanged by the academic instrumentation.

### 3.3 Timing CSV

Current timing columns:

```text
step,compute_ms,comm_ms,data_ms,eval_ms,other_ms
```

- `compute_ms`: CUDA event compute after comm de-overlap.
- `comm_ms`: full comm hook duration.
- `data_ms`: loader plus host-to-device preparation timer.
- `eval_ms`: validation duration assigned to the validation update.
- `other_ms`: remaining per-update wall time not assigned elsewhere.

The timing CSV step is zero-based. Run CSV steps are one-based. The old viewer
merge-by-position rule remains valid for the single-run table. Academic curve
comparisons use `tokens_seen`, not timing/run step keys.

### 3.4 GPU telemetry

`GPUUtilizationSampler` starts a daemon thread that invokes:

```text
nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits
```

approximately once per second. It writes timestamp, elapsed seconds, rank,
GPU index, utilization, and memory used. It does not call CUDA synchronization
and does not use the CUDA event collectors.

Availability is checked before NCCL process-group initialization. Missing
`nvidia-smi` aborts the run before rendezvous, preventing one rank from hanging
while the other fails.

### 3.5 Metadata and academic summary

Each run saves:

```text
logs/<tag>/config.json
logs/<tag>/metadata.json
logs/<tag>/metadata_rank<N>.json
```

The metadata includes tag, rank, world size, hostname, model, dataset,
training, and LoRA configuration. `academic_summary.csv` contains rank-0
headline metrics. Historical `summary.csv` is not read by academic analysis.

## 4. `src/train_lora.py`

### 4.1 Modes

Explicit config mode is scriptable:

```bash
uv run python -m src.train_lora \
  --config configs/resolved/baseline_long.yaml
```

No-argument single-GPU mode prompts for tunable values. Distributed launchers
always pass `--config`, so no rank prompts interactively.

Supported overrides:

```text
--config
--distributed
--validate-timings
--run-tag
--subset-size
--epochs
--batch-size
--seq-length
--grad-accumulation-steps
--learning-rate
--lora-r
--lora-alpha
--lora-dropout
```

Academic training tags are restricted to the four campaign names.

### 4.2 Device selection

Both machines have one local GPU. Global rank is not a CUDA device index:

```python
DDP(model, device_ids=[0])
```

This prevents rank 1 from attempting `cuda:1` on a one-GPU laptop.

### 4.3 Training update

For each optimizer update, the loop:

1. Fetches `accumulation` local batches.
2. Moves each batch to local GPU 0.
3. Runs non-final micro-steps under `no_sync()`.
4. Runs final micro-step with DDP synchronization.
5. Performs one optimizer update.
6. Writes live loss, samples, and actual `tokens_seen`.
7. Evaluates when the update is a 10% boundary or final update.

The final validation pass is included in end-to-end wall-clock and is also
reported separately as `validation_wall_clock_s`.

### 4.4 Validation loss

Validation calculates summed response-token loss and summed valid-token count.
DDP reduces both values across ranks:

```text
validation_loss = total_loss_sum / total_valid_tokens
```

This is token-weighted rather than example-weighted. Perplexity is safe:
finite losses below overflow threshold use `exp(loss)`; otherwise `inf` is
reported.

### 4.5 Adapter and interruption behavior

Only rank 0 saves the adapter:

```text
logs/<tag>/adapter/
```

At run start, rank 0 creates `logs/.training_active`. An existing marker
blocks a new run. This protects partial runs from silent overwrite. On normal
completion, the marker is removed. If a process is killed, preserve partial
files, inspect them, and remove the marker deliberately before retrying.

## 5. Launch Scripts and Resolved Configs

### 5.1 Common Docker flags

Both launchers use:

```text
--gpus all
--network host
--ipc=host
--ulimit memlock=-1
--ulimit stack=67108864
--env-file .env
--volume hf_cache:/root/.cache/huggingface
--volume logs:/app/logs
--volume configs:/app/configs
```

The rank launcher adds `RANK=0` or `RANK=1`.

### 5.2 Single-GPU flag

`launch_rank0.sh` defaults to distributed mode. To use the same container
launcher for a single GPU:

```bash
scripts/launch_rank0.sh \
  --single-gpu \
  --config configs/resolved/baseline_long.yaml
```

The script omits `--distributed` in this mode. This is the recommended short
command for a baseline run.

### 5.3 Distributed mode

Without `--single-gpu`, the launchers add `--distributed`:

```bash
# B first
scripts/launch_rank1.sh --config configs/resolved/2node_long.yaml

# A second
scripts/launch_rank0.sh --config configs/resolved/2node_long.yaml
```

Both ranks independently resolve the config, calculate its canonical hash,

### 5.4 `prepare_2node.sh`

The pairing script reads `academic_summary.csv`, lists completed single-node
baselines, and creates the matching DDP config:

```bash
scripts/prepare_2node.sh
```

It writes `configs/resolved/2node_1k.yaml` or

Use separate `scp` commands to preserve the `configs/resolved/` directory:

```bash
scp configs/resolved/2node_long.yaml \
  james@192.168.50.2:~/lora_zero3_demo/configs/resolved/
```

## 6. `src/analyze.py`

This is an offline, measured-only analysis command:

```bash
uv run python -m src.analyze --scale 1k
uv run python -m src.analyze --scale long
uv run python -m src.analyze --scale all
```

It refuses a scale pair if required artifacts are missing. For DDP it requires

Generated outputs include:

```text
logs/analysis/1k/loss_tokens_1k.png
logs/analysis/1k/throughput_1k.png
logs/analysis/1k/speedup_efficiency_1k.png
logs/analysis/1k/validation_tokens_1k.md
logs/analysis/1k/comm_stability_2node_1k.png
logs/analysis/long/loss_tokens_long.png
logs/analysis/long/throughput_long.png
logs/analysis/long/speedup_efficiency_long.png
logs/analysis/long/validation_tokens_long.md
logs/analysis/long/comm_stability_2node_long.png
logs/analysis/speedup_consistency.png
```

Other figures cover GPU utilization, timing breakdown, bandwidth, VRAM, and

### Token Alignment

`baseline_1k` has 250 updates and `2node_1k` has 125. Raw step 100 therefore
does not represent the same amount of data in the two runs. The loss plot uses
the actual `tokens_seen` column. Validation comparison tables use the exact
intersection of token counts. The same rule applies to the long pair.

## 7. WebUI Architecture and Usage

`src/webui.py` remains the entrypoint. The implementation also contains:

```text
src/web/inference.py
src/web/static/style.css
```

The server uses Python stdlib `ThreadingHTTPServer`. Matplotlib runs offline in
`src/analyze.py`; WebUI serves generated PNGs instead of drawing charts on
requests.

### 7.1 Legacy run mode

```bash
uv run python -m src.webui \
  --csv logs/run_2node_1k_rank0.csv \
  --port 8000
```

Routes:

```text
GET /
GET /api/rows
GET /static/style.css
```

The table merges run and timing CSVs by row position because run steps are
one-based and timing steps are zero-based. Live loss/samples appear during
training; compute/comm timing appears only after timing finalization.

### 7.2 Pair dashboard

Generate analysis first:

```bash
uv run python -m src.analyze --scale all
```

Start pair mode:

```bash
uv run python -m src.webui \
  --pair 1k \
  --analysis-dir logs/analysis \
  --port 8000
```

Use `--pair long` for the long pair. The scale selector changes the displayed
PNG figures. The dashboard also reads `academic_summary.csv` through

### 7.3 Inference page

Inference is A-only and opt-in:

```bash
uv run python -m src.webui \
  --pair 1k \
  --analysis-dir logs/analysis \
  --inference \
  --port 8000
```

Open `http://localhost:8000`, then select the **Inference** tab.

The page lists whichever of these four adapter directories exist:

```text
logs/baseline_1k/adapter
logs/2node_1k/adapter
logs/baseline_long/adapter
logs/2node_long/adapter
```

One base model remains resident. One selected adapter is loaded at a time.
BASE output disables adapter layers; FINETUNED output enables them. Both
generations use the same seed and independent single-example prompts.

The page exposes max tokens, temperature, top-p, repetition penalty, and seed.
Answers are safely escaped and rendered with local lightweight Markdown-like
formatting. Raw HTML is not executed. Prompt/answer history is visual only and
is not sent back as conversation context. Requests are serialized by a lock.
Inference refuses to run while `.training_active` exists.

### 7.4 Export

The dashboard's Download all link calls `/download-all` and creates a local ZIP
with locally available:

- CSV artifacts
- run metadata/configs
- resolved configs
- generated analysis figures

It does not retrieve files from B. Transfer rank-1 artifacts to A before
creating a complete A-side academic bundle.

### 7.5 Network exposure

Default host is localhost. To view from another machine on a trusted network:

```bash
uv run python -m src.webui \
  --pair 1k \
  --host 0.0.0.0 \
  --port 8000
```

Open `http://<machine-ip>:8000`. There is no authentication or TLS; do not
expose this server outside a trusted local network.

## 8. `src/infer.py`

The standalone REPL remains useful when a browser is unnecessary:

```bash
uv run python -m src.infer \
  --adapter logs/2node_long/adapter \
  --max-new-tokens 512
```

It loads a base model and selected PEFT model copy, uses sampling, resets at
the CodeAlpaca `\n### ` boundary, and prints BASE/FINETUNED answers. The WebUI
inference page is more memory-conscious because it keeps one base model and
switches one adapter at a time.

## 9. Docker Details

The Dockerfile uses:

```text
nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
Python 3.12 from uv
torch 2.13.0+cu130
```

The host driver is provided by NVIDIA Container Toolkit. Drivers are never
installed inside the image. The image runs as root because the uv-managed
Python environment lives under root-only paths; do not add `--user`.

Every `src/`, `pyproject.toml`, or `uv.lock` change requires:

```bash
scripts/build_image.sh --save
scp hetero-demo.tar james@192.168.50.2:~/lora_zero3_demo/
scripts/build_image.sh --load hetero-demo.tar  # on B
```

## 10. Troubleshooting

### Training fails before startup

- Check `nvidia-smi` inside and outside Docker.
- Check `nvidia-smi` telemetry command manually.
- Check `logs/.training_active`.
- Check config path exists inside the host-mounted `configs/` directory.
- Check both ranks use the same canonical config hash.

### NCCL hangs

Check static IPs, `NCCL_SOCKET_IFNAME`, firewall, `MASTER_PORT`, direct cable,
and `--network host`. Start B then A. Do not use Docker bridge networking.

### Validation OOM

Current validation batch size is deliberately 1. If B is stale, load the latest
image. The original academic attempt OOMed with validation batch size 4 and was
preserved as an incomplete diagnostic run.

### Missing rank-1 files

Rank 1 does not write validation CSV. It must write run/timing/GPU/metadata files.
After B finishes:

```bash
sudo chown -R "$USER:$USER" logs/
scp logs/run_<tag>_rank1.csv logs/timing_<tag>_rank1.csv \
  logs/gpu_<tag>_rank1.csv abhi@192.168.50.1:~/lora_zero3_demo/logs/
scp logs/<tag>/metadata_rank1.json \
  abhi@192.168.50.1:~/lora_zero3_demo/logs/<tag>/
```

### WebUI has no figures

Run:

```bash
uv run python -m src.analyze --scale all
```

Start WebUI with:

```bash
--analysis-dir logs/analysis
```

### WebUI inference tab disabled

Start with `--inference` on A. Confirm at least one campaign adapter exists.
The page is intentionally disabled on B and while training is active.

### Port occupied

Use another port:

```bash
--port 8001
```

### Model downloads again

Check the mount and cache:

```bash
docker inspect <container>
```

Required mount:

```text
<repo>/hf_cache:/root/.cache/huggingface
```

### Root-owned logs

Containers run as root by design:

```bash
sudo chown -R "$USER:$USER" logs/
```

## 11. Measured Four-Run Evaluation

The authoritative source is `logs/academic_summary.csv`; full narrative is in

```text
1k speedup:   1.969x
long speedup: 1.976x
```

Communication remained approximately 2.9% of rank-0 step time. Validation
quality was similar for the long pair. These results apply only to this model,
LoRA payload, hardware, network, software environment, and two-node topology.

## 12. Final Checks

```bash
uv run python -m py_compile src/*.py src/web/*.py
bash -n scripts/*.sh
uv run python -m src.analyze --scale all
```

No new experiment is needed to regenerate figures or the WebUI dashboard from
