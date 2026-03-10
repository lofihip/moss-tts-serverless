# MOSS-TTS Vast.ai PyWorker

## Build
docker build -t ghcr.io/yourname/moss-tts-vast:latest .

## Run local
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=OpenMOSS-Team/MOSS-TTS \
  -e TORCH_DTYPE=bfloat16 \
  ghcr.io/yourname/moss-tts-vast:latest

## Health
curl http://127.0.0.1:8000/health

## Local generate
curl -X POST http://127.0.0.1:8000/generate/sync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from MOSS TTS",
    "return_base64": false,
    "save_to_disk": true
  }'