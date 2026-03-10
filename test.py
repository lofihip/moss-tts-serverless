import base64
import json
import sys
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"
OUT_FILE = Path("test_output.wav")


def check_health():
    print("[1/2] Проверка /health ...")
    r = requests.get(f"{BASE_URL}/health", timeout=60)
    r.raise_for_status()
    data = r.json()
    print("Health response:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def check_generate():
    print("\n[2/2] Проверка /generate/sync ...")
    payload = {
        "text": "Привет! Это тест синтеза речи через MOSS TTS.",
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 2048
    }

    r = requests.post(
        f"{BASE_URL}/generate/sync",
        json=payload,
        timeout=600
    )
    r.raise_for_status()

    data = r.json()
    print("Generate response keys:", list(data.keys()))

    if not data.get("ok"):
        raise RuntimeError(f"API returned ok=false: {data}")

    wav_b64 = data.get("wav_base64")
    if not wav_b64:
        raise RuntimeError("В ответе нет wav_base64")

    wav_bytes = base64.b64decode(wav_b64)
    OUT_FILE.write_bytes(wav_bytes)

    print(f"WAV сохранён: {OUT_FILE.resolve()}")
    print(f"Sample rate: {data.get('sample_rate')}")
    print(f"Device: {data.get('device')}")
    print(f"Model: {data.get('model_name')}")
    print(f"Attention: {data.get('attn_implementation')}")


def main():
    try:
        check_health()
        check_generate()
        print("\nOK: API работает корректно.")
    except requests.HTTPError as e:
        print("\nHTTP ошибка:")
        print(e)
        if e.response is not None:
            print("Status:", e.response.status_code)
            try:
                print("Body:", json.dumps(e.response.json(), ensure_ascii=False, indent=2))
            except Exception:
                print("Body:", e.response.text)
        sys.exit(1)
    except Exception as e:
        print("\nОшибка:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()