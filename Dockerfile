# CUDA 13.0 + cuDNN devel base. NOTE: no cudnn9-suffixed tag exists for CUDA
# 13.0.0; 13.0.0-cudnn-devel ships cuDNN 9.x. uv.lock is the single source of
# truth for Python package versions (torch 2.13.0+cu130), reproducing the
# verified bare-metal environment exactly. Host driver comes from the host via
# NVIDIA Container Toolkit — never installed inside the container.
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# System deps: python runtime libs for uv-provisioned CPython + torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

# uv (dependency manager only — no pip install of python packages).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

WORKDIR /app

# Let uv.lock drive everything: Python 3.12 (matches host/known-good env),
# torch 2.13.0+cu130, peft, transformers, accelerate — all from lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python 3.12

# Copy app code.
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY .env.example ./

# Build-time surface of the runtime stack (no GPU needed for this check).
RUN /app/.venv/bin/python -c \
    "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); \
print('nccl', torch.distributed.is_nccl_available()); \
print('gpu_avail(cpu-build)', torch.cuda.is_available())" \
    && /app/.venv/bin/python -c \
    "import torch, torch.distributed, transformers, peft, accelerate, datasets, rich, textual, yaml; \
print('project imports OK')"

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

CMD ["bash"]