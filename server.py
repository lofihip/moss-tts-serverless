import os
import io
import uuid
import base64
import importlib.util
from pathlib import Path
from typing import Optional, Literal

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

# README MOSS-TTS recommends disabling cuDNN SDPA backend and keeping fallbacks enabled.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def resolve_attn_implementation() -> str:
    if (
        DEVICE == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and DTYPE in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if DEVICE == "cuda":
        return "sdpa"
    return "eager"

ATTN_IMPL = resolve_attn_implementation()

app = FastAPI(title="MOSS-TTS Local API", version="1.0.0")

processor = None
model = None
sampling_rate = 24000  # fallback; overwritten after model load if available


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    reference_audio: Optional[str] = None  # URL or local path
    tokens: Optional[int] = None           # duration control
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

    print(f"[INFO] Loading processor from {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

    print(f"[INFO] Loading model from {MODEL_NAME}")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    if hasattr(processor, "model_config") and hasattr(processor.model_config, "sampling_rate"):
        sampling_rate = int(processor.model_config.sampling_rate)

    print("[INFO] Application startup complete.")
    print(f"[INFO] Using device={DEVICE}, dtype={DTYPE}, attn={ATTN_IMPL}, sr={sampling_rate}")


@app.get("/health")
def health():
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
    try:
        message_kwargs = {"text": req.text}

        if req.reference_audio:
            message_kwargs["reference"] = [req.reference_audio]

        if req.tokens is not None:
            message_kwargs["tokens"] = req.tokens

        conversations = [[processor.build_user_message(**message_kwargs)]]
        batch = processor(conversations, mode="generation")

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
            )

        decoded = list(processor.decode(outputs))
        if not decoded:
            raise RuntimeError("processor.decode returned no messages")

        audio = decoded[0].audio_codes_list[0]

        out_path = None
        wav_b64 = None

        if req.save_to_disk:
            out_path = str(OUTPUT_DIR / f"{uuid.uuid4().hex}.wav")
            torchaudio.save(out_path, audio.unsqueeze(0).cpu(), sampling_rate)

        if req.return_base64:
            buf = io.BytesIO()
            torchaudio.save(buf, audio.unsqueeze(0).cpu(), sampling_rate, format="wav")
            wav_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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
        raise HTTPException(status_code=500, detail=str(e))