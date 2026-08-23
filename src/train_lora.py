"""Phases 3-4 — LoRA fine-tuning, single-GPU baseline and 2-node DDP.

One rank-aware script. Bare-metal or in Docker; launched manually per node
with RANK env (scripts/launch_rank*.sh).

    # single-GPU baseline (laptop A, Phase 3):
    uv run python -m src.train_lora --config configs/lora_config.yaml

    # 2-node DDP (laptop A rank 0, laptop B rank 1, Phase 4):
    uv run python -m src.train_lora --config configs/lora_config.yaml --distributed

Hard rules (AGENTS.md): fixed sample budget is the comparison unit; exactly
one all-reduce per optimizer update (no_sync on non-final micro-steps); comm
timed via comm hook; CUDA events synced ONCE at end of run; frozen base params
so DDP syncs only LoRA adapter grads.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import (
    LiveCSV,
    GPUUtilizationSampler,
    TimingCollector,
    ValidationCSV,
    comm_pct,
    curve_matches,
    extrapolated_epoch_s,
    load_losses,
    loss_criterion_met,
    make_timing_hook,
    monotonic_now,
    new_cuda_events,
    require_dist_env,
    run_csv_path,
    samples_per_sec,
    tokens_per_sec,
    write_summary_csv,
    write_timing_csv,
    write_metadata,
)

CONTEXT = contextlib.nullcontext
ACADEMIC_TAGS = {"baseline_1k", "2node_1k", "baseline_long", "2node_long"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/lora_config.yaml", help="configs/lora_config.yaml")
    p.add_argument("--distributed", action="store_true", help="init DDP process group")
    p.add_argument("--validate-timings", action="store_true",
                   help="short profiler run to cross-check hook timing, then exit")
    p.add_argument("--run-tag", default=None, help="override config run_tag")
    p.add_argument("--subset-size", type=int, default=None, help="override dataset size")
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--grad-accumulation-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    return p.parse_args()


def _prompt_value(label: str, default, cast):
    value = input(f"{label} [{default}]: ").strip()
    return default if not value else cast(value)


def interactive_config() -> dict:
    cfg = load_config("configs/lora_config.yaml")
    cfg["dataset"]["subset_size"] = _prompt_value(
        "subset_size", cfg["dataset"]["subset_size"], int)
    cfg["training"]["batch_size"] = _prompt_value(
        "batch_size", cfg["training"]["batch_size"], int)
    cfg["training"]["seq_length"] = _prompt_value(
        "seq_length", cfg["training"]["seq_length"], int)
    cfg["training"]["grad_accumulation_steps"] = _prompt_value(
        "grad_accumulation_steps", cfg["training"]["grad_accumulation_steps"], int)
    cfg["training"]["learning_rate"] = _prompt_value(
        "learning_rate", cfg["training"]["learning_rate"], float)
    cfg["training"]["num_epochs"] = _prompt_value(
        "num_epochs", cfg["training"]["num_epochs"], int)
    cfg["lora"]["r"] = _prompt_value("lora_r", cfg["lora"]["r"], int)
    cfg["lora"]["alpha"] = _prompt_value("lora_alpha", cfg["lora"]["alpha"], int)
    cfg["lora"]["dropout"] = _prompt_value("lora_dropout", cfg["lora"]["dropout"], float)
    cfg["logging"]["run_tag"] = _prompt_value(
        "run_tag", cfg["logging"].get("run_tag", "baseline_1k"), str)
    return cfg


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tokenize_dataset(ds, tokenizer, seq_length: int):
    def format_example(ex):
        prompt = f"### Instruction:\n{ex['instruction']}\n"
        if ex.get("input"):
            prompt += f"### Input:\n{ex['input']}\n"
        prompt += "### Response:\n"
        full = prompt + ex["output"] + "\n"
        return prompt, full

    def tokenize_fn(ex):
        prompt, full = format_example(ex)
        full_ids = tokenizer(
            full, truncation=True, max_length=seq_length, return_tensors=None
        )["input_ids"]
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=seq_length, return_tensors=None
        )["input_ids"]
        labels = [-100] * len(full_ids)
        labels[len(prompt_ids):] = full_ids[len(prompt_ids):]
        attn = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attn,
            "labels": labels,
        }

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds = ds.map(
        lambda ex: {
            "input_ids": ex["input_ids"][:seq_length]
            + [tokenizer.pad_token_id] * max(0, seq_length - len(ex["input_ids"])),
            "attention_mask": ex["attention_mask"][:seq_length]
            + [0] * max(0, seq_length - len(ex["attention_mask"])),
            "labels": ex["labels"][:seq_length]
            + [-100] * max(0, seq_length - len(ex["labels"])),
        }
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def build_datasets(cfg: dict, tokenizer):
    dcfg = cfg["dataset"]
    seq_length = cfg["training"]["seq_length"]
    all_ds = load_dataset(dcfg["name"], split="train").shuffle(seed=dcfg["seed"])
    validation_size = int(dcfg.get("validation_size", 100))
    if validation_size <= 0 or validation_size >= len(all_ds):
        raise ValueError("validation_size must be positive and smaller than dataset")
    val_raw = all_ds.select(range(validation_size))
    train_pool = all_ds.select(range(validation_size, len(all_ds)))
    subset_size = int(dcfg["subset_size"])
    if subset_size > len(train_pool):
        raise ValueError(f"subset_size={subset_size} exceeds train pool {len(train_pool)}")
    train_raw = train_pool.select(range(subset_size))
    return (_tokenize_dataset(train_raw, tokenizer, seq_length),
            _tokenize_dataset(val_raw, tokenizer, seq_length))


def partition_dataset(ds: Dataset, rank: int, world_size: int) -> Dataset:
    """Split already-shuffled training rows without DistributedSampler padding."""
    if world_size == 1:
        return ds
    indices = list(range(rank, len(ds), world_size))
    return torch.utils.data.Subset(ds, indices)


def write_validation_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "step", "tokens_seen", "val_loss", "val_perplexity", "eval_ms",
        ])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, loader, device, rank: int, world_size: int, step: int,
             tokens_seen: int) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    token_count = torch.zeros(1, device=device, dtype=torch.float64)
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            valid = labels[..., 1:].ne(-100)
            token_count += valid.sum().to(torch.float64)
            loss_sum += out.loss.detach().to(torch.float64) * valid.sum()
    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    val_loss = (loss_sum / token_count.clamp_min(1)).item()
    val_ppl = float(np.exp(val_loss)) if np.isfinite(val_loss) and val_loss < 80 else float("inf")
    elapsed_ms = (time.monotonic() - start) * 1000.0
    if was_training:
        model.train()
    return val_loss, val_ppl, elapsed_ms


def build_model(cfg: dict, rank: int):
    mcfg = cfg["model"]
    device = torch.device("cuda", 0)
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        torch_dtype=torch.float16 if mcfg.get("fp16", True) else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if mcfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.train()
    return model, device


def log_rank(msg: str, rank: int) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def train(args: argparse.Namespace, cfg: dict) -> None:
    tcfg = cfg["training"]
    dcfg = cfg["dataset"]
    lcfg = cfg["logging"]
    rank = 0
    world_size = 1

    if not args.distributed:
        GPUUtilizationSampler.check_available()
    if args.distributed:
        env = require_dist_env()
        rank = int(env["RANK"])
        world_size = int(env["WORLD_SIZE"])
        GPUUtilizationSampler.check_available()
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        # One GPU per node: both global ranks use local device 0 (NOT rank).
        torch.cuda.set_device(0)

    device = torch.device("cuda", 0)
    set_seed(dcfg["seed"])

    run_tag = args.run_tag or lcfg.get("run_tag") or (
        "2node" if args.distributed else "baseline"
    )
    subset_size = args.subset_size or dcfg["subset_size"]
    num_epochs = args.epochs or tcfg["num_epochs"]
    batch_size = args.batch_size or tcfg["batch_size"]
    accum = args.grad_accumulation_steps or tcfg["grad_accumulation_steps"]
    seq_length = args.seq_length or tcfg["seq_length"]
    learning_rate = args.learning_rate or tcfg["learning_rate"]
    cfg["dataset"]["subset_size"] = subset_size
    cfg["training"].update({
        "batch_size": batch_size,
        "grad_accumulation_steps": accum,
        "seq_length": seq_length,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    })
    cfg["lora"].update({
        "r": args.lora_r or cfg["lora"]["r"],
        "alpha": args.lora_alpha or cfg["lora"]["alpha"],
        "dropout": args.lora_dropout if args.lora_dropout is not None else cfg["lora"]["dropout"],
    })
    lcfg["run_tag"] = run_tag
    if not args.validate_timings and run_tag not in ACADEMIC_TAGS:
        raise SystemExit(
            f"academic run_tag must be one of {sorted(ACADEMIC_TAGS)}; got {run_tag!r}"
        )
    config_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    log_rank(f"resolved config sha256: {config_hash}", rank)
    if args.distributed:
        hashes = [None] * world_size
        dist.all_gather_object(hashes, config_hash)
        if len(set(hashes)) != 1:
            raise SystemExit(f"resolved config mismatch across ranks: {hashes}")
    budget = subset_size * num_epochs
    update_samples = world_size * batch_size * accum
    if budget % update_samples:
        raise SystemExit(
            f"sample budget {budget} must divide evenly by global update size "
            f"{update_samples}; choose subset_size accordingly"
        )
    log_rank(f"config: budget={budget} samples, batch={batch_size}, "
             f"accum={accum}, world={world_size}", rank)

    output_dir = lcfg.get("output_dir", "logs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    active_marker = Path(output_dir) / ".training_active"
    if rank == 0 and not args.validate_timings:
        if active_marker.exists():
            raise SystemExit(
                f"incomplete training marker exists at {active_marker}; "
                "inspect partial artifacts before clearing it"
            )
        active_marker.write_text(run_tag + "\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds, val_ds = build_datasets(cfg, tokenizer)
    local_ds = partition_dataset(ds, rank, world_size)
    loader = DataLoader(
        local_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    local_val_ds = partition_dataset(val_ds, rank, world_size)
    val_loader = DataLoader(local_val_ds, batch_size=1, shuffle=False)
    token_counts = [int(sum(row["attention_mask"])) for row in ds]
    local_count = batch_size * accum

    model, _ = build_model(cfg, rank)
    torch.cuda.reset_peak_memory_stats(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_rank(f"trainable LoRA params: {trainable}", rank)

    comm_collector = TimingCollector()
    current_step = [0]
    if args.distributed:
        ddp = DDP(model, device_ids=[0], gradient_as_bucket_view=True)
        ddp.register_comm_hook(None, make_timing_hook(comm_collector,
                                                      lambda: current_step[0],
                                                      world_size))
        wrapped = ddp
    else:
        wrapped = model

    optimizer = torch.optim.AdamW(
        [p for p in wrapped.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True) if cfg["model"].get("fp16", True) else None

    if args.validate_timings:
        _validate_timings(args, cfg, wrapped, optimizer, loader, device, rank,
                          batch_size, accum, seq_length, world_size, comm_collector)
        return

    write_metadata(output_dir, run_tag, cfg, rank, world_size)
    live = LiveCSV(run_csv_path(output_dir, run_tag, rank), ["step", "loss", "samples_processed", "tokens_seen"])
    compute_collector = TimingCollector()
    validation_path = Path(output_dir) / f"validation_{run_tag}_rank{rank}.csv"
    if rank == 0 and validation_path.exists():
        validation_path.unlink()
    gpu_sampler = GPUUtilizationSampler(
        Path(output_dir) / f"gpu_{run_tag}_rank{rank}.csv", rank
    )
    gpu_sampler.start()

    wall_start = monotonic_now()
    consumed = 0
    step = 0
    running_loss = 0.0
    samples_processed = 0
    tokens_seen = 0
    data_ms_by_step = {}
    eval_ms_by_step = {}
    update_wall_ms_by_step = {}
    updates_total = max(1, int(np.ceil(budget / (world_size * batch_size * accum))))
    eval_steps = set(max(1, int(round(updates_total * frac)))
                    for frac in np.arange(0.1, 1.01, 0.1))

    iterator = iter(loader)
    while consumed < budget:
        update_wall_start = time.monotonic()
        current_step[0] = step
        for micro in range(accum):
            micro_data_start = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            data_ms_by_step[step] = data_ms_by_step.get(step, 0.0) + (
                (time.monotonic() - micro_data_start) * 1000.0
            )
            is_final = micro == accum - 1

            ctx = wrapped.no_sync() if (args.distributed and not is_final) else CONTEXT()
            with ctx:
                start_e, end_e = new_cuda_events()
                start_e.record()
                if scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = wrapped(input_ids=input_ids, attention_mask=attn, labels=labels)
                    loss = out.loss / accum
                    scaler.scale(loss).backward()
                else:
                    out = wrapped(input_ids=input_ids, attention_mask=attn, labels=labels)
                    loss = out.loss / accum
                    loss.backward()
                end_e.record()
                compute_collector.record(step, start_e, end_e)
                running_loss += float(out.loss.item())
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        samples_processed += world_size * batch_size * accum
        consumed = min(budget, samples_processed)
        local_start = step * local_count
        tokens_seen += sum(
            token_counts[((local_start + local_pos) * world_size + other_rank) % subset_size]
            for local_pos in range(local_count)
            for other_rank in range(world_size)
        )
        step += 1
        step_loss = running_loss / accum
        running_loss = 0.0
        live.write_row({
            "step": step,
            "loss": round(step_loss, 6),
            "samples_processed": samples_processed,
            "tokens_seen": tokens_seen,
        })
        if step in eval_steps or consumed >= budget:
            val_loss, val_ppl, eval_ms = evaluate(
                wrapped, val_loader, device, rank, world_size, step, tokens_seen
            )
            eval_ms_by_step[step - 1] = eval_ms
            if rank == 0:
                write_validation_row(validation_path, {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "val_loss": round(val_loss, 6),
                    "val_perplexity": round(val_ppl, 6) if np.isfinite(val_ppl) else "inf",
                    "eval_ms": round(eval_ms, 3),
                })
            if args.distributed:
                dist.barrier()
        if step % 10 == 0 or step == 1:
            log_rank(f"step {step}: loss={step_loss:.4f} samples={samples_processed}/{budget}", rank)
        update_wall_ms_by_step[step - 1] = (time.monotonic() - update_wall_start) * 1000.0

    wall_end = monotonic_now()
    wall_s = wall_end - wall_start
    live.close()
    gpu_sampler.stop()

    if rank == 0:
        adapter_dir = Path(output_dir) / run_tag / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        log_rank(f"adapter saved to {adapter_dir}", rank)

    compute_ms_by_step = compute_collector.finalize()
    comm_ms_by_step = comm_collector.finalize() if args.distributed else {}
    compute_clean = {
        st: max(0.0, compute_ms_by_step.get(st, 0.0) - comm_ms_by_step.get(st, 0.0))
        for st in sorted(set(compute_ms_by_step) | set(comm_ms_by_step))
    }
    other_ms_by_step = {
        st: max(0.0, update_wall_ms_by_step.get(st, 0.0)
                - compute_clean.get(st, 0.0)
                - comm_ms_by_step.get(st, 0.0)
                - data_ms_by_step.get(st, 0.0)
                - eval_ms_by_step.get(st, 0.0))
        for st in update_wall_ms_by_step
    }
    timing_path = write_timing_csv(output_dir, run_tag, rank,
                                   compute_clean, comm_ms_by_step,
                                   data_ms_by_step, eval_ms_by_step, other_ms_by_step)

    losses = load_losses(run_csv_path(output_dir, run_tag, rank))
    met, first_mean, last_mean, window = loss_criterion_met(losses)
    final_loss = losses[-1] if losses else float("nan")
    tot_compute = sum(compute_clean.values())
    tot_comm = sum(comm_ms_by_step.values())
    tot_data = sum(data_ms_by_step.values())
    tot_eval = sum(eval_ms_by_step.values())
    other_ms = max(0.0, wall_s * 1000.0 - tot_compute - tot_comm - tot_data - tot_eval)
    trainable_bytes = sum(
        p.numel() * p.element_size() for p in wrapped.parameters() if p.requires_grad
    )
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    effective_bw = ((trainable_bytes * step) / (tot_comm / 1000.0) / 1e6) if tot_comm > 0 else 0.0
    final_val_loss = ""
    final_val_ppl = ""
    if rank == 0 and validation_path.exists():
        rows = list(csv.DictReader(open(validation_path, newline="")))
        if rows:
            final_val_loss = float(rows[-1]["val_loss"])
            final_val_ppl = float(rows[-1]["val_perplexity"])
    frac_done = consumed / budget if budget > 0 else 0.0

    log_rank(f"done: {step} updates, {wall_s:.2f}s wall, "
             f"{samples_per_sec(budget, wall_s):.2f} samples/s, "
             f"compute={tot_compute:.0f}ms comm={tot_comm:.0f}ms "
             f"comm%={comm_pct(tot_compute, tot_comm):.1f}", rank)
    log_rank(f"loss criterion: {'MET' if met else 'not met'} "
             f"(first{window}={first_mean:.4f} last{window}={last_mean:.4f}, "
             f"final={final_loss:.4f})", rank)

    if rank == 0:
        write_summary_csv(output_dir, {
            "run_tag": run_tag,
            "world_size": world_size,
            "rank": rank,
            "total_samples": budget,
            "total_updates": step,
            "wall_clock_s": round(wall_s, 3),
            "samples_per_sec": round(samples_per_sec(budget, wall_s), 3),
            "tokens_per_sec": round(tokens_seen / wall_s if wall_s > 0 else 0.0, 3),
            "compute_ms_per_update_avg": round(tot_compute / max(1, step), 3),
            "comm_ms_per_update_avg": round(tot_comm / max(1, step), 3),
            "comm_pct_of_step": round(comm_pct(tot_compute, tot_comm), 2),
            "extrapolated_epoch_s": round(extrapolated_epoch_s(wall_s, frac_done), 3),
            "final_loss": round(final_loss, 6),
            "loss_criterion_met": int(met),
            "train_wall_clock_s": round(max(0.0, wall_s - tot_eval / 1000.0), 3),
            "validation_wall_clock_s": round(tot_eval / 1000.0, 3),
            "data_ms_per_update_avg": round(tot_data / max(1, step), 3),
            "other_ms_per_update_avg": round(other_ms / max(1, step), 3),
            "peak_vram_allocated_mib": round(peak_allocated, 3),
            "peak_vram_reserved_mib": round(peak_reserved, 3),
            "trainable_param_bytes": trainable_bytes,
            "effective_bandwidth_mb_s": round(effective_bw, 3),
            "bandwidth_utilization_pct": round(effective_bw / 125.0 * 100.0, 3),
            "final_val_loss": final_val_loss,
            "final_val_perplexity": final_val_ppl,
        })
        # Academic pair comparisons happen in offline analysis after both
        # campaign runs exist. Never compare against historical run_*.csv.

    if args.distributed:
        dist.destroy_process_group()
    if rank == 0:
        active_marker.unlink(missing_ok=True)


def _validate_timings(args, cfg, model, optimizer, loader, device, rank,
                      batch_size, accum, seq_length, world_size, comm_collector):
    """Phase 5 cross-check: torch.profiler ground truth vs comm-hook numbers.

    Mirrors the real training loop (no_sync on non-final micro-steps, one
    all-reduce set per optimizer update) so the profiled comm pattern matches
    a real update.
    """
    import torch.profiler as profiler

    updates = 0
    micros = 0
    with profiler.profile(activities=[
            profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
        iterator = iter(loader)
        for _ in range(3):
            for micro in range(accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                is_final = micro == accum - 1
                ctx = (model.no_sync() if (args.distributed and not is_final)
                       else CONTEXT())
                with ctx:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(input_ids=input_ids, attention_mask=attn,
                                    labels=labels)
                    loss = out.loss / accum
                    loss.backward()
                micros += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
    torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
    log_rank("--- profiler table (top CUDA ops) ---", rank)
    print(table, flush=True)
    # Count ONLY the raw NCCL kernels (c10d::allreduce_ wrapper + torch's
    # "nccl:all_reduce" alias are the SAME ops — including them triple-counts).
    kernel_evs = [e for e in prof.key_averages()
                  if e.key.startswith("ncclDevKernel")]
    prof_comm_ms = sum(e.self_device_time_total for e in kernel_evs) / 1e3
    hook_ms = sum(comm_collector.finalize().values())
    log_rank(f"profiler NCCL kernel time: {prof_comm_ms:.1f} ms over {micros} "
             f"micro-steps / {updates} updates "
             f"= {prof_comm_ms / max(1, updates):.1f} ms/update", rank)
    log_rank(f"comm-hook wall time (same window): {hook_ms:.1f} ms "
             f"= {hook_ms / max(1, updates):.1f} ms/update", rank)


def main() -> None:
    args = parse_args()
    if not args.distributed and len(sys.argv) == 1:
        cfg = interactive_config()
    else:
        cfg = load_config(args.config)
    train(args, cfg)


if __name__ == "__main__":
    main()
