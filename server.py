import os
import io
import uuid
import base64
import importlib.util
import traceback
from pathlib import Path
from typing import Optional

print("[SERVER] server.py imported", flush=True)

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

# ====== Config ======
MODEL_NAME = os.getenv("MODEL_NAME", "OpenMOSS-Team/MOSS-TTS")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_NAME = os.getenv("TORCH_DTYPE", "bfloat16" if DEVICE == "cuda" else "float32")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if DTYPE_NAME == "float16":
    DTYPE = torch.float16
elif DTYPE_NAME == "bfloat16":
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32

print(f"[SERVER] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[SERVER] DEVICE={DEVICE}", flush=True)
print(f"[SERVER] DTYPE_NAME={DTYPE_NAME}", flush=True)
print(f"[SERVER] OUTPUT_DIR={OUTPUT_DIR}", flush=True)

if DEVICE == "cuda":
    try:
        print(f"[SERVER] CUDA device count={torch.cuda.device_count()}", flush=True)
        print(f"[SERVER] CUDA device name={torch.cuda.get_device_name(0)}", flush=True)
        print(f"[SERVER] CUDA capability={torch.cuda.get_device_capability(0)}", flush=True)
    except Exception as e:
        print(f"[SERVER] failed to inspect CUDA device: {e}", flush=True)

# README MOSS-TTS style backend settings
if DEVICE == "cuda":
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SERVER] configured torch SDP backends", flush=True)
    except Exception as e:
        print(f"[SERVER] failed to configure torch SDP backends: {e}", flush=True)


def resolve_attn_implementation() -> str:
    if (
        DEVICE == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and DTYPE in {torch.float16, torch.bfloat16}
    ):
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        except Exception as e:
            print(f"[SERVER] flash_attn check failed: {e}", flush=True)

    if DEVICE == "cuda":
        return "sdpa"
    return "eager"


ATTN_IMPL = resolve_attn_implementation()
print(f"[SERVER] ATTN_IMPL={ATTN_IMPL}", flush=True)

app = FastAPI(title="MOSS-TTS Local API", version="1.0.0")

processor = None
model = None
sampling_rate = 24000


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    reference_audio: Optional[str] = None
    tokens: Optional[int] = None
    max_new_tokens: int = 4096
    return_base64: bool = True
    save_to_disk: bool = False


class GenerateResponse(BaseModel):
    ok: bool
    sample_rate: int
    wav_base64: Optional[str] = None
    output_path: Optional[str] = None
    attn_implementation: str
    device: str
    model_name: str


@app.on_event("startup")
def startup_event():
    global processor, model, sampling_rate

    try:
        print("[SERVER] startup_event begin", flush=True)
        print(f"[SERVER] loading processor from {MODEL_NAME}", flush=True)

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        print("[SERVER] processor loaded", flush=True)

        processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)
        print("[SERVER] processor.audio_tokenizer moved to device", flush=True)

        print(f"[SERVER] loading model from {MODEL_NAME}", flush=True)
        print(f"[SERVER] using attn_implementation={ATTN_IMPL}", flush=True)
        print(f"[SERVER] using torch_dtype={DTYPE}", flush=True)

        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
            torch_dtype=DTYPE,
        ).to(DEVICE)

        model.eval()
        print("[SERVER] model loaded and set to eval()", flush=True)

        if hasattr(processor, "model_config") and hasattr(processor.model_config, "sampling_rate"):
            sampling_rate = int(processor.model_config.sampling_rate)

        print(f"[SERVER] sampling_rate={sampling_rate}", flush=True)
        print("[SERVER] Application startup complete.", flush=True)

    except Exception as e:
        print(f"[SERVER] startup failed: {e}", flush=True)
        traceback.print_exc()
        raise


@app.get("/health")
def health():
    print("[SERVER] /health called", flush=True)
    return {
        "ok": True,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "attn_implementation": ATTN_IMPL,
        "model_name": MODEL_NAME,
        "sample_rate": sampling_rate,
    }


@app.post("/generate/sync", response_model=GenerateResponse)
def generate_sync(req: GenerateRequest):
    print(
        f"[SERVER] /generate/sync called text_len={len(req.text)} "
        f"reference_audio={'yes' if req.reference_audio else 'no'} "
        f"tokens={req.tokens} max_new_tokens={req.max_new_tokens} "
        f"return_base64={req.return_base64} save_to_disk={req.save_to_disk}",
        flush=True,
    )

    try:
        message_kwargs = {"text": req.text}

        if req.reference_audio:
            message_kwargs["reference"] = [req.reference_audio]

        if req.tokens is not None:
            message_kwargs["tokens"] = req.tokens

        print("[SERVER] building user message", flush=True)
        conversations = [[processor.build_user_message(**message_kwargs)]]

        print("[SERVER] tokenizing inputs", flush=True)
        batch = processor(conversations, mode="generation")

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        print(
            f"[SERVER] input_ids shape={tuple(input_ids.shape)} "
            f"attention_mask shape={tuple(attention_mask.shape)}",
            flush=True,
        )

        with torch.no_grad():
            print("[SERVER] calling model.generate()", flush=True)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
            )

        print("[SERVER] decoding outputs", flush=True)
        decoded = list(processor.decode(outputs))
        if not decoded:
            raise RuntimeError("processor.decode returned no messages")

        audio = decoded[0].audio_codes_list[0]
        print(f"[SERVER] audio tensor shape={tuple(audio.shape)}", flush=True)

        out_path = None
        wav_b64 = None

        if req.save_to_disk:
            out_path = str(OUTPUT_DIR / f"{uuid.uuid4().hex}.wav")
            print(f"[SERVER] saving wav to {out_path}", flush=True)
            torchaudio.save(out_path, audio.unsqueeze(0).cpu(), sampling_rate)

        if req.return_base64:
            print("[SERVER] encoding wav to base64", flush=True)
            buf = io.BytesIO()
            torchaudio.save(buf, audio.unsqueeze(0).cpu(), sampling_rate, format="wav")
            wav_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print("[SERVER] /generate/sync completed successfully", flush=True)
        return GenerateResponse(
            ok=True,
            sample_rate=sampling_rate,
            wav_base64=wav_b64,
            output_path=out_path,
            attn_implementation=ATTN_IMPL,
            device=DEVICE,
            model_name=MODEL_NAME,
        )

    except Exception as e:
        print(f"[SERVER] /generate/sync failed: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))