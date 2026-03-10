#!/usr/bin/env bash
set -euxo pipefail

echo "[BOOT] start.sh launched"
echo "[BOOT] pwd=$(pwd)"
echo "[BOOT] MODEL_VENV=${MODEL_VENV:-/opt/model-venv}"
echo "[BOOT] WORKER_VENV=${WORKER_VENV:-/opt/worker-venv}"
echo "[BOOT] MODEL_NAME=${MODEL_NAME:-}"
echo "[BOOT] TORCH_DTYPE=${TORCH_DTYPE:-}"
echo "[BOOT] OUTPUT_DIR=${OUTPUT_DIR:-}"

mkdir -p /workspace/outputs
touch /workspace/model.log

echo "[BOOT] model python=$(${MODEL_VENV:-/opt/model-venv}/bin/python --version 2>&1)"
echo "[BOOT] worker python=$(${WORKER_VENV:-/opt/worker-venv}/bin/python --version 2>&1)"

echo "[BOOT] starting uvicorn from model venv"
${MODEL_VENV:-/opt/model-venv}/bin/uvicorn server:app --host 0.0.0.0 --port 8000 2>&1 | tee /workspace/model.log &
SERVER_PID=$!

for i in $(seq 1 300); do
  echo "[BOOT] healthcheck attempt $i"

  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    echo "[BOOT] model server is healthy"
    break
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[BOOT] uvicorn exited unexpectedly"
    cat /workspace/model.log || true
    exit 1
  fi

  sleep 2
done

echo "[BOOT] starting worker.py from worker venv"
exec ${WORKER_VENV:-/opt/worker-venv}/bin/python -u worker.py