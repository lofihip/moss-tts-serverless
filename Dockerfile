FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    MODEL_NAME=OpenMOSS-Team/MOSS-TTS \
    TORCH_DTYPE=bfloat16 \
    OUTPUT_DIR=/workspace/outputs

WORKDIR /workspace/app

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git git-lfs ffmpeg curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py worker.py start.sh ./
RUN chmod +x /workspace/app/start.sh

EXPOSE 8000
CMD ["/workspace/app/start.sh"]