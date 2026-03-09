#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/outputs
touch /workspace/model.log

# Локальный model server
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /workspace/model.log 2>&1 &

# Ждём готовность model server
for i in $(seq 1 300); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    echo "[INFO] model server is healthy"
    break
  fi
  sleep 2
done

# Запуск Vast PyWorker
python worker.py