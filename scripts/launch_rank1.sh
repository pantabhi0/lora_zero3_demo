#!/usr/bin/env bash
# Run on laptop B (rank 1). Image must already be built and identical on both.
set -euo pipefail

if [ ! -f .env ]; then
    echo "missing .env — fill MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, NCCL_SOCKET_IFNAME per Phase 0" >&2
    exit 1
fi

IMG="${IMG:-hetero-demo:latest}"

exec docker run --rm \
    --gpus all \
    --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env-file .env \
    --env RANK=1 \
    --volume "$(pwd)/hf_cache:/root/.cache/huggingface" \
    --volume "$(pwd)/logs:/app/logs" \
    "$IMG" \
    python -m src.train_lora --config configs/lora_config.yaml --distributed "$@"