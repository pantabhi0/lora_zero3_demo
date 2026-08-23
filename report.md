# HeteroTrain Academic Campaign Report

**Model:** `Qwen/Qwen2.5-0.5B` base
**Method:** fp16 LoRA, rank 16, alpha 32, dropout 0.05
**Dataset:** `sahil2801/CodeAlpaca-20k`
**Hardware:** two GTX 1650 4 GB laptops, Turing capability 7.5
**Network:** direct 1 Gbps Ethernet, approximately 0.58 ms ping latency
**Academic runs:** exactly four: `baseline_1k`, `2node_1k`, `baseline_long`, `2node_long`

Historical `baseline` and `default` artifacts are excluded from all academic
results in this report.

---

## Executive Summary

The campaign measured single-GPU LoRA fine-tuning against two-node PyTorch DDP
using identical models, fixed seed, disjoint held-out validation data, and
equal total sample budgets within each pair.

| Scale | 1-node wall | 2-node wall | Speedup | 2-node comm share |
| --- | ---: | ---: | ---: | ---: |
| 1k | 3286.125 s | 1668.567 s | **1.969x** | 2.87% |
| Long | 8841.677 s | 4475.444 s | **1.976x** | 2.93% |

Parallel efficiency, defined as `speedup / 2`, was:

| Scale | Efficiency |
| --- | ---: |
| 1k | 98.47% |
| Long | 98.78% |

The speedup is consistent across both scales. The long-run result does not
support a claim about arbitrary GPU counts or heterogeneous scaling; this
campaign has only one GPU per node and two nodes.

Primary quality metric is held-out response-only validation loss/perplexity.
The corrected academic label construction masks prompt tokens with `-100`.
The historical training-loss acceptance rule was defined for different,
incorrect label semantics and is therefore not used as an academic pass/fail
criterion here.

---

## Experimental Design

### Dataset Split

The full dataset is shuffled once with seed `42`. The first 100 shuffled
examples are permanently reserved for validation. Training subsets are then
selected only from the remaining examples.

This produces the same disjoint validation set for all four runs:

```text
full CodeAlpaca-20k
  -> fixed shuffle(seed=42)
  -> first 100 examples: held-out validation
  -> remaining examples: training pool
```

Prompt tokens are masked:

```python
labels = [-100] * len(full_ids)
labels[len(prompt_ids):] = full_ids[len(prompt_ids):]
```

Only response tokens contribute to causal-LM loss.

### Pair Budgets

| Pair | Training samples | 1-node updates | 2-node updates | Final tokens seen |
| --- | ---: | ---: | ---: | ---: |
| 1k | 1000 | 250 | 125 | 88092 |
| Long | 3288 | 822 | 411 | 282309 |

The per-GPU configuration is batch size 1 and gradient accumulation 4.
Therefore:

```text
1-node update size = 1 × 1 × 4 = 4 samples
2-node update size = 2 × 1 × 4 = 8 samples
```

Training samples are rank-disjoint in DDP. Non-final accumulation micro-steps
use `no_sync()`, giving exactly one all-reduce per optimizer update.

### Validation

Both ranks evaluate the same held-out split at approximately 10% update
intervals and at the final update. Validation uses batch size 1 to fit the
4 GB GTX 1650. Both ranks synchronize with a barrier after evaluation. Rank 0
writes authoritative validation CSVs.

Validation loss is token-weighted. Perplexity is computed safely as
`exp(validation_loss)` when finite and below overflow threshold.

### Timing

Existing validated CUDA timing behavior was preserved:

- CUDA events accumulate through the run.
- `torch.cuda.synchronize()` occurs only during end-of-run finalization.
- `.elapsed_time()` is not called inside the training loop.
- Communication is timed by the existing asynchronous DDP comm hook.
- Compute is de-overlapped by subtracting measured communication time.

New metrics use independent timers:

- loader plus host-to-device data time
- validation duration
- explicit residual time
- 1 Hz `nvidia-smi` telemetry
- peak allocated/reserved VRAM

Comm bandwidth is an effective payload estimate based on measured LoRA payload
bytes and comm duration, compared against the theoretical 125 MB/s Ethernet
ceiling. It is not packet-level network instrumentation.

---

## Results

### Summary

| Metric | `baseline_1k` | `2node_1k` | `baseline_long` | `2node_long` |
| --- | ---: | ---: | ---: | ---: |
| World size | 1 | 2 | 1 | 2 |
| Samples | 1000 | 1000 | 3288 | 3288 |
| Updates | 250 | 125 | 822 | 411 |
| Wall clock (s) | 3286.125 | 1668.567 | 8841.677 | 4475.444 |
| Samples/sec | 0.304 | 0.599 | 0.372 | 0.735 |
| Tokens/sec | 26.807 | 52.795 | 31.929 | 63.080 |
| Compute/update (ms) | 9716.168 | 9581.403 | 9704.481 | 9534.843 |
| Comm/update (ms) | 0 | 283.488 | 0 | 287.709 |
| Comm share | 0% | 2.87% | 0% | 2.93% |
| Final training loss | 1.023930 | 2.097480 | 0.451276 | 0.625020 |
| Final validation loss | 0.606460 | 0.572788 | 0.567638 | 0.565360 |
| Final validation PPL | 1.833927 | 1.773204 | 1.764096 | 1.760081 |
| Peak allocated VRAM (MiB) | 1631.491 | 1665.054 | 1631.491 | 1665.054 |

### Throughput and Speedup

| Scale | Baseline wall (s) | DDP wall (s) | Speedup | Parallel efficiency |
| --- | ---: | ---: | ---: | ---: |
| 1k | 3286.125 | 1668.567 | 1.969x | 98.47% |
| Long | 8841.677 | 4475.444 | 1.976x | 98.78% |

The long run confirms the 1k result. Speedup changed by only approximately
0.6 percentage points between scales. No scaling curve is claimed because
only one-node and two-node measurements exist.

### Communication

| Metric | `2node_1k` | `2node_long` |
| --- | ---: | ---: |
| Rank-0 mean comm/update | 283.488 ms | 287.709 ms |
| Rank-0 steady-state mean | 283.348 ms | 287.674 ms |
| Rank-0 steady-state median | 281.805 ms | 282.096 ms |
| Rank-0 steady-state minimum | 280.324 ms | 280.033 ms |
| Rank-0 steady-state maximum | 366.735 ms | 392.805 ms |
| Rank-0 steady-state standard deviation | 10.367 ms | 18.213 ms |
| Effective bandwidth estimate | 124.142 MB/s | 122.321 MB/s |
| Theoretical-ceiling utilization | 99.314% | 97.857% |

Communication is stable after initial warm-up. The rolling-average figures are
in `logs/analysis/1k/comm_stability_2node_1k.png` and
`logs/analysis/long/comm_stability_2node_long.png`.

The all-reduce payload is the LoRA gradient payload. The measured trainable
parameter count is 8,798,208 and the logged payload byte estimate is
35,192,832 bytes. The earlier historical documentation estimate of roughly
3 MB is superseded by this measured academic value.

### GPU Utilization and VRAM

Mean `nvidia-smi` utilization was approximately 99% for all runs. Short zero
or low samples correspond to startup, validation, or shutdown intervals.

| Run | Rank | Mean GPU utilization | Peak nvidia-smi memory |
| --- | ---: | ---: | ---: |
| `baseline_1k` | 0 | 99.70% | 1946 MiB |
| `2node_1k` | 0 | 99.57% | 2094 MiB |
| `2node_1k` | 1 | 99.78% | 2094 MiB |
| `baseline_long` | 0 | 99.83% | 1946 MiB |
| `2node_long` | 0 | 99.76% | 2094 MiB |
| `2node_long` | 1 | 99.84% | 2244 MiB |

Peak PyTorch allocated VRAM was 1631.491 MiB for baseline and 1665.054 MiB
for DDP. The DDP process has slightly higher memory use because of
distributed buffers and communication state.

### Data and Residual Time

| Run | Data ms/update | Other ms/update | Validation wall time |
| --- | ---: | ---: | ---: |
| `baseline_1k` | 6.443 | 10.621 | 852.816 s |
| `2node_1k` | 57.760 | 12.380 | 426.688 s |
| `baseline_long` | 5.692 | 8.447 | 852.972 s |
| `2node_long` | 21.512 | 8.951 | 425.855 s |

The 2-node data figure includes initial distributed startup/rendezvous effects,
which makes its average larger than steady-state data rows. Validation takes
approximately 85.3 seconds per single-GPU evaluation pass and approximately
42.6 seconds per DDP rank-0 pass.

### Held-Out Quality

Validation is primary. Training loss is a noisy diagnostic because the two
configurations use different effective global batch sizes and update counts.

| Run | Final validation loss | Final validation PPL |
| --- | ---: | ---: |
| `baseline_1k` | 0.606460 | 1.833927 |
| `2node_1k` | 0.572788 | 1.773204 |
| `baseline_long` | 0.567638 | 1.764096 |
| `2node_long` | 0.565360 | 1.760081 |

The long pair has nearly identical final held-out quality. The 2-node long run
has slightly lower validation loss and perplexity than the single-node long
run. This is an observed result, not a claim that DDP inherently improves
quality.

Validation comparisons are aligned by `tokens_seen`, not raw step number. The
generated tables are:

```text
logs/analysis/1k/validation_tokens_1k.md
logs/analysis/long/validation_tokens_long.md
```

### Historical Label-Semantics Note

The old runs used prompt positions as pad-token labels. The academic campaign
uses correct `-100` masking, so historical losses such as `2.46` first-window
loss are not numerically comparable to the new response-only losses near
`0.6`. No old run is used in this report.

---

## Acceptance Evaluation

| Check | Result |
| --- | --- |
| Four required campaign tags completed | PASS |
| Same 1k sample budget within pair | PASS |
| Same long sample budget within pair | PASS |
| Fixed disjoint validation split | PASS |
| Correct response-only masking | PASS |
| Rank-disjoint DDP training data | PASS |
| Rank-0/rank-1 long artifacts complete | PASS |
| Config hash agreement across DDP ranks | PASS |
| End-of-run CUDA timing behavior preserved | PASS |
| GPU telemetry captured per required rank | PASS |
| Token-aligned comparison artifacts generated | PASS |
| Static PNG analysis generated | PASS |
| Final validation data available for all runs | PASS |

The original historical training-loss drop criterion is **not used** as an
academic acceptance test because it was defined before the corrected `-100`
label masking. Academic quality evaluation uses held-out validation loss and
perplexity.

---

## Limitations

1. Only one GPU per node and two total nodes were tested. No multi-GPU scaling
   curve or heterogeneity ratio is claimed.
2. The GTX 1650 has no Tensor Cores. Results should not be generalized to
   Tensor-Core hardware without new measurements.
3. The LoRA adapter has 8.8 million trainable parameters. Full-model training
   or a larger adapter would change communication behavior.
4. Each scale has one run per configuration. Within-run timing variance is
   available; run-to-run variance is not.
5. The baseline and DDP configurations have different effective global batch
   sizes. This is part of the fixed-sample-budget experiment and affects
   optimizer trajectories and training loss.
6. Bandwidth is an effective payload estimate, not a packet capture or NIC
   counter measurement.
7. Validation batch size is 1 because larger validation batches OOMed the
   4 GB card. Evaluation overhead is therefore substantial and explicitly
   reported.

---

## Artifact Index

Primary source:

```text
logs/academic_summary.csv
```

Per-run artifacts:

```text
logs/run_<tag>_rank<N>.csv
logs/timing_<tag>_rank<N>.csv
logs/validation_<tag>_rank0.csv
logs/gpu_<tag>_rank<N>.csv
logs/<tag>/config.json
logs/<tag>/metadata*.json
logs/<tag>/adapter/
```

Static analysis:

```text
logs/analysis/1k/
logs/analysis/long/
logs/analysis/speedup_consistency.png
```

Regenerate all static figures after artifacts are present:

```bash
uv run python -m src.analyze --scale all
```

The WebUI can serve generated figures and local exports. Rank-1 raw files are
collected on A manually after each distributed run, as required by the
two-machine launch design.

---

## Conclusion

Across both measured scales, two GTX 1650 laptops completed the same fixed
sample budget approximately **1.97× faster** than one laptop. Parallel
efficiency was approximately **98.5–98.8%**, while communication consumed
approximately **2.9%** of rank-0 step time. Effective communication bandwidth
was approximately 122–124 MB/s against the theoretical 125 MB/s ceiling.

Held-out validation quality was similar across the long pair:

```text
baseline_long: 0.567638 loss, 1.764096 perplexity
2node_long:    0.565360 loss, 1.760081 perplexity
```

The defensible conclusion is narrow: for this base 0.5B model, fp16 LoRA
configuration, two identical GTX 1650 laptops, and direct gigabit Ethernet,
plain two-node DDP nearly halves wall-clock time for fixed sample budgets with
low measured synchronization overhead. It is not a general scaling claim.
