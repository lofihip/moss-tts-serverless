FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    MODEL_NAME=OpenMOSS-Team/MOSS-TTS \
    TORCH_DTYPE=bfloat16 \
    OUTPUT_DIR=/workspace/outputs \
    MODEL_VENV=/opt/model-venv \
    WORKER_VENV=/opt/worker-venv

WORKDIR /workspace/app

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git git-lfs ffmpeg curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# ===== model venv =====
RUN python -m venv /opt/model-venv
RUN /opt/model-venv/bin/python -m pip install --upgrade pip setuptools wheel

RUN /opt/model-venv/bin/pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1+cu128 torchaudio==2.9.1+cu128

# optional flash-attn wheel build arg
ARG FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-linux_x86_64.whl"
RUN if [ -n "$FLASH_ATTN_WHEEL_URL" ]; then \
      echo "Installing flash-attn from wheel: $FLASH_ATTN_WHEEL_URL" && \
      /opt/model-venv/bin/pip install "$FLASH_ATTN_WHEEL_URL"; \
    else \
      echo "FLASH_ATTN_WHEEL_URL is empty, skipping flash-attn wheel install"; \
    fi

COPY requirements-model.txt .
RUN /opt/model-venv/bin/pip install -r requirements-model.txt

# ===== worker venv =====
RUN python -m venv /opt/worker-venv
RUN /opt/worker-venv/bin/python -m pip install --upgrade pip setuptools wheel

COPY requirements-worker.txt .
RUN /opt/worker-venv/bin/pip install -r requirements-worker.txt

# sanity check for pyworker api
RUN /opt/worker-venv/bin/python -c "from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig; print('vast pyworker api ok')"

COPY server.py worker.py start.sh test.py ./
RUN chmod +x /workspace/app/start.sh

EXPOSE 8000
CMD ["/workspace/app/start.sh"]