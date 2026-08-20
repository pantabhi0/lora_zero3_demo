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
import random
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
    TimingCollector,
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
)

CONTEXT = contextlib.nullcontext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="configs/lora_config.yaml")
    p.add_argument("--distributed", action="store_true", help="init DDP process group")
    p.add_argument("--validate-timings", action="store_true",
                   help="short profiler run to cross-check hook timing, then exit")
    p.add_argument("--run-tag", default=None, help="override config run_tag")
    p.add_argument("--subset-size", type=int, default=None, help="override dataset size")
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, tokenizer) -> Dataset:
    dcfg = cfg["dataset"]
    seq_length = cfg["training"]["seq_length"]

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
        labels = [tokenizer.pad_token_id] * len(full_ids)
        labels[len(prompt_ids):] = full_ids[len(prompt_ids):]
        attn = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attn,
            "labels": labels,
        }

    ds = load_dataset(dcfg["name"], split="train")
    ds = ds.shuffle(seed=dcfg["seed"]).select(range(dcfg["subset_size"]))
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


def build_model(cfg: dict, rank: int):
    mcfg = cfg["model"]
    device = torch.device("cuda", 0)
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"],
        torch_dtype=torch.float16 if mcfg.get("fp16", True) else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
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

    if args.distributed:
        env = require_dist_env()
        rank = int(env["RANK"])
        world_size = int(env["WORLD_SIZE"])
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
    batch_size = tcfg["batch_size"]
    accum = tcfg["grad_accumulation_steps"]
    seq_length = tcfg["seq_length"]
    budget = subset_size * num_epochs
    log_rank(f"config: budget={budget} samples, batch={batch_size}, "
             f"accum={accum}, world={world_size}", rank)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = build_dataset(cfg, tokenizer)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(dcfg["seed"]),
    )

    model, _ = build_model(cfg, rank)
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
        lr=tcfg["learning_rate"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True) if cfg["model"].get("fp16", True) else None

    if args.validate_timings:
        _validate_timings(args, cfg, wrapped, optimizer, loader, device, rank,
                          batch_size, accum, seq_length, world_size, comm_collector)
        return

    output_dir = lcfg.get("output_dir", "logs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    live = LiveCSV(run_csv_path(output_dir, run_tag, rank), ["step", "loss", "samples_processed"])
    compute_collector = TimingCollector()

    wall_start = monotonic_now()
    consumed = 0
    step = 0
    running_loss = 0.0
    samples_processed = 0

    iterator = iter(loader)
    while consumed < budget:
        for micro in range(accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
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

        current_step[0] = step
        samples_processed += world_size * batch_size * accum
        consumed = min(budget, samples_processed)
        step += 1
        step_loss = running_loss / accum
        running_loss = 0.0
        live.write_row({
            "step": step,
            "loss": round(step_loss, 6),
            "samples_processed": samples_processed,
        })
        if step % 10 == 0 or step == 1:
            log_rank(f"step {step}: loss={step_loss:.4f} samples={samples_processed}/{budget}", rank)

    wall_end = monotonic_now()
    wall_s = wall_end - wall_start
    live.close()

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
    timing_path = write_timing_csv(output_dir, run_tag, rank,
                                   compute_clean, comm_ms_by_step)

    losses = load_losses(run_csv_path(output_dir, run_tag, rank))
    met, first_mean, last_mean, window = loss_criterion_met(losses)
    final_loss = losses[-1] if losses else float("nan")
    tot_compute = sum(compute_clean.values())
    tot_comm = sum(comm_ms_by_step.values())
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
            "tokens_per_sec": round(tokens_per_sec(budget, seq_length, wall_s), 3),
            "compute_ms_per_update_avg": round(tot_compute / max(1, step), 3),
            "comm_ms_per_update_avg": round(tot_comm / max(1, step), 3),
            "comm_pct_of_step": round(comm_pct(tot_compute, tot_comm), 2),
            "extrapolated_epoch_s": round(extrapolated_epoch_s(wall_s, frac_done), 3),
            "final_loss": round(final_loss, 6),
            "loss_criterion_met": int(met),
        })
        if losses:
            for other in sorted(Path(output_dir).glob("run_*_rank0.csv")):
                if other.name == run_csv_path(output_dir, run_tag, rank).name:
                    continue
                other_losses = load_losses(other)
                ok, max_rel = curve_matches(losses, other_losses)
                log_rank(f"curve match vs {other.name}: "
                         f"{'OK' if ok else 'DIVERGE'} max_rel={max_rel:.4f}", rank)

    if args.distributed:
        dist.destroy_process_group()


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
    cfg = load_config(args.config)
    train(args, cfg)


if __name__ == "__main__":
    main()