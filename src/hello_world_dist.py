"""Phase 1 — distributed hello-world (no ML, no Docker yet).

Validates GPU + NCCL + network wiring before any ML complexity. Runs manually
on each node: laptop A RANK=0, laptop B RANK=1, MASTER_ADDR=laptop A IP.
Reads NCCL_SOCKET_IFNAME and friends from env (no defaults, no auto-detect).

Usage (bare metal, from repo root):
    RANK=0 python -m src.hello_world_dist
    RANK=1 python -m src.hello_world_dist
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from src.metrics import require_dist_env


def main() -> None:
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

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    tensor = torch.ones(1024, dtype=torch.float32, device=device) * (rank + 1)
    start_event.record()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_event.record()
    torch.cuda.synchronize()

    expected = float(world_size * (world_size + 1) / 2)
    got = float(tensor[0].item())
    print(
        f"[rank {rank}] all_reduce ok: tensor[0]={got} "
        f"(expected {expected}) in {start_event.elapsed_time(end_event):.3f} ms"
    )
    assert abs(got - expected) < 1e-3, f"mismatch: {got} != {expected}"

    dist.destroy_process_group()
    print(f"[rank {rank}] clean exit")


if __name__ == "__main__":
    main()