import argparse
import base64
import hashlib
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


ROUTE_URL = "https://run.vast.ai/route/"
DEFAULT_TIMEOUT = 1800
DEFAULT_OUT_DIR = "test_outputs"


def log(msg: str) -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fail(msg: str, exit_code: int = 1) -> None:
    print(f"[FAIL] {msg}", flush=True)
    sys.exit(exit_code)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def save_wav_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def inspect_wav_file(path: Path) -> Tuple[int, int, float]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
        return sample_rate, n_frames, duration


def wav_pcm_sha256(path: Path) -> str:
    with wave.open(str(path), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    return hashlib.sha256(pcm).hexdigest()


class VastServerlessMossTTSClient:
    def __init__(
        self,
        endpoint_name: str,
        api_key: str,
        route_url: str = ROUTE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        route_cost: float = 1.0,
        extra_headers_json: str = "",
        max_route_wait_seconds: int = 120,
        route_poll_interval_seconds: float = 2.0,
    ):
        self.endpoint_name = endpoint_name
        self.api_key = api_key
        self.route_url = route_url
        self.timeout = timeout
        self.route_cost = route_cost
        self.request_idx: Optional[int] = None
        self.max_route_wait_seconds = max_route_wait_seconds
        self.route_poll_interval_seconds = route_poll_interval_seconds

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

        if extra_headers_json:
            extra_headers = json.loads(extra_headers_json)
            if not isinstance(extra_headers, dict):
                raise ValueError("--extra-headers-json must decode to an object/dict")
            for k, v in extra_headers.items():
                self.session.headers[str(k)] = str(v)

    def route(
        self,
        cost: Optional[float] = None,
        retry_same_request: bool = False,
        max_wait_seconds: Optional[int] = None,
        poll_interval_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        max_wait_seconds = (
            self.max_route_wait_seconds if max_wait_seconds is None else max_wait_seconds
        )
        poll_interval_seconds = (
            self.route_poll_interval_seconds
            if poll_interval_seconds is None
            else poll_interval_seconds
        )

        started = time.time()
        first = True

        while True:
            payload: Dict[str, Any] = {
                "endpoint": self.endpoint_name,
                "cost": float(self.route_cost if cost is None else cost),
            }

            if retry_same_request and self.request_idx is not None:
                payload["request_idx"] = self.request_idx
            elif not first and self.request_idx is not None:
                payload["request_idx"] = self.request_idx

            log(
                f"[ROUTE] requesting worker for endpoint={self.endpoint_name} "
                f"cost={payload['cost']} request_idx={payload.get('request_idx')}"
            )

            r = self.session.post(self.route_url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            log("[ROUTE] response:")
            log(json.dumps(data, ensure_ascii=False, indent=2))

            if "request_idx" in data:
                self.request_idx = data["request_idx"]

            if "url" in data and "signature" in data and "reqnum" in data:
                return data

            status = data.get("status", "")
            if status:
                elapsed = time.time() - started
                if elapsed >= max_wait_seconds:
                    raise RuntimeError(
                        f"/route did not return worker url within {max_wait_seconds}s. "
                        f"Last response: {data}"
                    )

                log(
                    f"[ROUTE] worker not assigned yet, waiting {poll_interval_seconds}s "
                    f"(elapsed={elapsed:.1f}s)"
                )
                time.sleep(poll_interval_seconds)
                first = False
                continue

            raise RuntimeError(f"/route returned unexpected response: {data}")

    def call_worker(
        self, route_data: Dict[str, Any], route_path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        base_url = route_data["url"].rstrip("/")
        worker_url = f"{base_url}{route_path}"

        body = {
            "auth_data": {
                "signature": route_data["signature"],
                "cost": route_data["cost"],
                "endpoint": route_data["endpoint"],
                "reqnum": route_data["reqnum"],
                "url": route_data["url"],
                "request_idx": route_data.get("request_idx"),
            },
            "payload": payload,
        }

        log(f"[WORKER] POST {worker_url}")
        started = time.time()
        r = self.session.post(worker_url, json=body, timeout=self.timeout)
        elapsed = time.time() - started
        log(f"[WORKER] status={r.status_code} elapsed_sec={elapsed:.3f}")

        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"Worker response is not JSON: {r.text[:1000]}")

        if r.status_code >= 400:
            raise RuntimeError(
                f"Worker returned HTTP {r.status_code}: "
                f"{json.dumps(data, ensure_ascii=False)}"
            )

        return data

    def health(self) -> Dict[str, Any]:
        route_data = self.route(cost=1.0)
        return self.call_worker(route_data, "/health", {})

    def generate(self, payload: Dict[str, Any], cost: Optional[float] = None) -> Dict[str, Any]:
        if cost is None:
            raw_steps = payload.get("max_new_tokens", 1024)
            try:
                cost = max(1.0, round(float(raw_steps), 3))
            except Exception:
                cost = 1024.0

        route_data = self.route(cost=cost)
        return self.call_worker(route_data, "/generate/sync", payload)


def decode_response_audio(resp: Dict[str, Any], out_dir: Path, filename: str) -> Path:
    wav_b64 = resp.get("wav_base64")
    if not wav_b64:
        raise RuntimeError("Response does not contain wav_base64")

    wav_bytes = base64.b64decode(wav_b64)
    out_path = out_dir / filename
    save_wav_bytes(out_path, wav_bytes)
    return out_path


def run_health_test(client: VastServerlessMossTTSClient) -> Dict[str, Any]:
    log("\n=== TEST: /health via Vast route ===")
    data = client.health()
    log(json.dumps(data, ensure_ascii=False, indent=2))

    if not data.get("ok"):
        raise RuntimeError("/health returned ok=false")

    ok("/health works through Vast serverless")
    return data


def run_basic_tts_test(client: VastServerlessMossTTSClient, out_dir: Path) -> Path:
    log("\n=== TEST: basic TTS ===")
    payload = {
        "text": "Привет! Это удалённый тест синтеза речи через MOSS TTS на Vast.ai.",
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1024,
    }

    resp = client.generate(payload)

    if not resp.get("ok"):
        raise RuntimeError("Basic TTS returned ok=false")

    out_path = decode_response_audio(resp, out_dir, "01_basic_remote.wav")
    sr, frames, duration = inspect_wav_file(out_path)

    log(
        json.dumps(
            {
                "sample_rate": sr,
                "frames": frames,
                "duration_sec": round(duration, 3),
                "device": resp.get("device"),
                "attn_implementation": resp.get("attn_implementation"),
                "model_name": resp.get("model_name"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if frames <= 0 or duration <= 0:
        raise RuntimeError("Basic TTS produced empty audio")

    ok(f"basic TTS works -> {out_path}")
    return out_path


def run_multilingual_test(client: VastServerlessMossTTSClient, out_dir: Path) -> Path:
    log("\n=== TEST: multilingual / code-switch ===")
    payload = {
        "text": "Привет! This is a multilingual test. Hola! Сегодня мы проверяем MOSS TTS через Vast.ai.",
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1400,
    }

    resp = client.generate(payload)
    if not resp.get("ok"):
        raise RuntimeError("Multilingual test returned ok=false")

    out_path = decode_response_audio(resp, out_dir, "02_multilingual_remote.wav")
    sr, frames, duration = inspect_wav_file(out_path)

    if frames <= 0 or duration <= 0:
        raise RuntimeError("Multilingual test produced empty audio")

    log(
        json.dumps(
            {
                "sample_rate": sr,
                "frames": frames,
                "duration_sec": round(duration, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    ok(f"multilingual test works -> {out_path}")
    return out_path


def run_duration_test(
    client: VastServerlessMossTTSClient,
    out_dir: Path,
    text: str,
    short_tokens: int = 180,
    long_tokens: int = 360,
) -> Tuple[Path, Path]:
    log("\n=== TEST: duration control ===")

    payload_short = {
        "text": text,
        "tokens": short_tokens,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }
    payload_long = {
        "text": text,
        "tokens": long_tokens,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }

    resp_short = client.generate(payload_short)
    resp_long = client.generate(payload_long)

    if not resp_short.get("ok"):
        raise RuntimeError("Duration short test returned ok=false")
    if not resp_long.get("ok"):
        raise RuntimeError("Duration long test returned ok=false")

    short_path = decode_response_audio(resp_short, out_dir, "03_duration_short.wav")
    long_path = decode_response_audio(resp_long, out_dir, "04_duration_long.wav")

    sr1, frames1, dur1 = inspect_wav_file(short_path)
    sr2, frames2, dur2 = inspect_wav_file(long_path)

    log(
        json.dumps(
            {
                "short_tokens": short_tokens,
                "short_duration_sec": round(dur1, 3),
                "long_tokens": long_tokens,
                "long_duration_sec": round(dur2, 3),
                "short_sample_rate": sr1,
                "long_sample_rate": sr2,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if frames1 <= 0 or frames2 <= 0:
        raise RuntimeError("Duration test produced empty audio")

    if dur2 <= dur1:
        warn("Request succeeded, but long_tokens did not produce longer audio.")
    else:
        ok("duration control behaves as expected: longer tokens -> longer audio")

    return short_path, long_path


def run_reference_test(
    client: VastServerlessMossTTSClient,
    out_dir: Path,
    reference_url: str,
    suffix: str,
) -> Path:
    log(f"\n=== TEST: voice cloning / reference ({suffix}) ===")

    if not is_url(reference_url):
        raise ValueError("reference_url must be a public http/https URL")

    payload = {
        "text": "Это удалённый тест клонирования голоса по референсу через Vast.ai.",
        "reference_audio": reference_url,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }

    resp = client.generate(payload)
    if not resp.get("ok"):
        raise RuntimeError(f"Reference test failed for {suffix}")

    out_path = decode_response_audio(resp, out_dir, f"05_clone_{suffix}.wav")
    sr, frames, duration = inspect_wav_file(out_path)

    log(
        json.dumps(
            {
                "reference_audio": reference_url,
                "sample_rate": sr,
                "frames": frames,
                "duration_sec": round(duration, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if frames <= 0 or duration <= 0:
        raise RuntimeError(f"Reference test produced empty audio for {suffix}")

    ok(f"voice cloning / reference test works -> {out_path}")
    return out_path


def run_reference_usage_test(
    client: VastServerlessMossTTSClient,
    out_dir: Path,
    reference_url: str,
) -> Dict[str, Any]:
    log("\n=== TEST: reference usage check (A/B) ===")

    if not is_url(reference_url):
        raise ValueError("reference_url must be a public http/https URL")

    text = (
        "Это A/B проверка влияния референса. "
        "Нужно подтвердить, что референс меняет итоговый голос."
    )

    payload_no_ref = {
        "text": text,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }
    payload_with_ref = {
        "text": text,
        "reference_audio": reference_url,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }

    resp_no_ref = client.generate(payload_no_ref)
    resp_with_ref = client.generate(payload_with_ref)

    if not resp_no_ref.get("ok"):
        raise RuntimeError("Reference usage test failed: no-reference request returned ok=false")
    if not resp_with_ref.get("ok"):
        raise RuntimeError("Reference usage test failed: with-reference request returned ok=false")

    path_no_ref = decode_response_audio(resp_no_ref, out_dir, "06_ref_usage_no_ref.wav")
    path_with_ref = decode_response_audio(resp_with_ref, out_dir, "07_ref_usage_with_ref.wav")

    sr1, frames1, dur1 = inspect_wav_file(path_no_ref)
    sr2, frames2, dur2 = inspect_wav_file(path_with_ref)

    if frames1 <= 0 or dur1 <= 0:
        raise RuntimeError("Reference usage test produced empty audio for no-reference run")
    if frames2 <= 0 or dur2 <= 0:
        raise RuntimeError("Reference usage test produced empty audio for with-reference run")

    hash_no_ref = wav_pcm_sha256(path_no_ref)
    hash_with_ref = wav_pcm_sha256(path_with_ref)

    log(
        json.dumps(
            {
                "reference_audio": reference_url,
                "no_ref": {
                    "path": str(path_no_ref.resolve()),
                    "sample_rate": sr1,
                    "frames": frames1,
                    "duration_sec": round(dur1, 3),
                    "pcm_sha256": hash_no_ref,
                },
                "with_ref": {
                    "path": str(path_with_ref.resolve()),
                    "sample_rate": sr2,
                    "frames": frames2,
                    "duration_sec": round(dur2, 3),
                    "pcm_sha256": hash_with_ref,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if hash_no_ref == hash_with_ref:
        raise RuntimeError(
            "Reference usage check failed: outputs are bit-identical with and without reference_audio"
        )

    ok("reference usage check passed: output differs with reference_audio")
    return {
        "no_ref": str(path_no_ref.resolve()),
        "with_ref": str(path_with_ref.resolve()),
        "no_ref_pcm_sha256": hash_no_ref,
        "with_ref_pcm_sha256": hash_with_ref,
    }


def run_voice_cloning_test_two_refs(
    client: VastServerlessMossTTSClient,
    out_dir: Path,
    reference_url_a: str,
    reference_url_b: str,
) -> Dict[str, Any]:
    log("\n=== TEST: voice cloning check (two references) ===")

    if not is_url(reference_url_a) or not is_url(reference_url_b):
        raise ValueError("reference_url_a/reference_url_b must be public http/https URLs")

    text = (
        "Это тест клонирования голоса с двумя разными референсами. "
        "Ожидается различие между итоговыми голосами."
    )

    payload_a = {
        "text": text,
        "reference_audio": reference_url_a,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }
    payload_b = {
        "text": text,
        "reference_audio": reference_url_b,
        "return_base64": True,
        "save_to_disk": False,
        "max_new_tokens": 1200,
    }

    resp_a = client.generate(payload_a)
    resp_b = client.generate(payload_b)

    if not resp_a.get("ok"):
        raise RuntimeError("Voice cloning test failed: reference A request returned ok=false")
    if not resp_b.get("ok"):
        raise RuntimeError("Voice cloning test failed: reference B request returned ok=false")

    path_a = decode_response_audio(resp_a, out_dir, "08_clone_ref_a.wav")
    path_b = decode_response_audio(resp_b, out_dir, "09_clone_ref_b.wav")

    sr_a, frames_a, dur_a = inspect_wav_file(path_a)
    sr_b, frames_b, dur_b = inspect_wav_file(path_b)

    if frames_a <= 0 or dur_a <= 0:
        raise RuntimeError("Voice cloning test produced empty audio for reference A")
    if frames_b <= 0 or dur_b <= 0:
        raise RuntimeError("Voice cloning test produced empty audio for reference B")

    hash_a = wav_pcm_sha256(path_a)
    hash_b = wav_pcm_sha256(path_b)

    log(
        json.dumps(
            {
                "reference_a": reference_url_a,
                "reference_b": reference_url_b,
                "clone_a": {
                    "path": str(path_a.resolve()),
                    "sample_rate": sr_a,
                    "frames": frames_a,
                    "duration_sec": round(dur_a, 3),
                    "pcm_sha256": hash_a,
                },
                "clone_b": {
                    "path": str(path_b.resolve()),
                    "sample_rate": sr_b,
                    "frames": frames_b,
                    "duration_sec": round(dur_b, 3),
                    "pcm_sha256": hash_b,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if hash_a == hash_b:
        raise RuntimeError(
            "Voice cloning check failed: outputs are bit-identical for two different references"
        )

    ok("voice cloning check passed: two references produce different outputs")
    return {
        "ref_a": str(path_a.resolve()),
        "ref_b": str(path_b.resolve()),
        "ref_a_pcm_sha256": hash_a,
        "ref_b_pcm_sha256": hash_b,
    }


def run_save_to_disk_test(client: VastServerlessMossTTSClient) -> Optional[str]:
    log("\n=== TEST: save_to_disk on remote worker ===")

    payload = {
        "text": "Проверка сохранения на диск на стороне Vast.ai worker.",
        "return_base64": False,
        "save_to_disk": True,
        "max_new_tokens": 800,
    }

    resp = client.generate(payload)
    if not resp.get("ok"):
        raise RuntimeError("save_to_disk test returned ok=false")

    output_path = resp.get("output_path")
    if not output_path:
        warn("save_to_disk succeeded, but output_path is empty")
        return None

    ok(f"remote save_to_disk works -> server path: {output_path}")
    return output_path


def run_negative_test_empty_text(client: VastServerlessMossTTSClient) -> None:
    log("\n=== TEST: negative case / empty text ===")

    try:
        client.generate(
            {
                "text": "",
                "return_base64": True,
                "save_to_disk": False,
            },
            cost=0.1,
        )
        raise RuntimeError("Empty text unexpectedly succeeded")
    except Exception as e:
        msg = str(e)
        if "HTTP 400" in msg or "HTTP 422" in msg or "HTTP 500" in msg:
            ok("empty text rejected as expected")
            return
        raise


def print_summary(results: Dict[str, Any]) -> None:
    log("\n=== SUMMARY ===")
    log(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote Vast.ai Serverless test script for MOSS-TTS"
    )
    parser.add_argument("--endpoint-name", required=True, help="Vast Serverless endpoint name")
    parser.add_argument("--api-key", required=True, help="Vast API key")
    parser.add_argument("--route-url", default=ROUTE_URL, help="Vast route URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--route-cost", type=float, default=1.0, help="Default route cost")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for WAV files on your PC")
    parser.add_argument("--reference-url", default="", help="Public URL to reference audio for cloning test")
    parser.add_argument(
        "--reference-url-2",
        default="",
        help="Second public URL to reference audio for two-reference cloning check",
    )
    parser.add_argument("--skip-health", action="store_true", help="Skip health test")
    parser.add_argument("--skip-negative", action="store_true", help="Skip empty-text negative test")
    parser.add_argument(
        "--extra-headers-json",
        default="",
        help='Extra headers JSON, e.g. \'{"X-Test":"1"}\'',
    )
    parser.add_argument(
        "--max-route-wait-seconds",
        type=int,
        default=120,
        help="How long to wait for /route to return a worker URL",
    )
    parser.add_argument(
        "--route-poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval for repeated /route calls",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    client = VastServerlessMossTTSClient(
        endpoint_name=args.endpoint_name,
        api_key=args.api_key,
        route_url=args.route_url,
        timeout=args.timeout,
        route_cost=args.route_cost,
        extra_headers_json=args.extra_headers_json,
        max_route_wait_seconds=args.max_route_wait_seconds,
        route_poll_interval_seconds=args.route_poll_interval_seconds,
    )

    results: Dict[str, Any] = {
        "endpoint_name": args.endpoint_name,
        "route_url": args.route_url,
        "out_dir": str(out_dir.resolve()),
        "tests": {},
    }

    try:
        if not args.skip_health:
            health = run_health_test(client)
            results["tests"]["health"] = health
        else:
            warn("Health test skipped")
            results["tests"]["health"] = "skipped"

        basic_path = run_basic_tts_test(client, out_dir)
        results["tests"]["basic_tts"] = str(basic_path.resolve())

        multilingual_path = run_multilingual_test(client, out_dir)
        results["tests"]["multilingual"] = str(multilingual_path.resolve())

        short_path, long_path = run_duration_test(
            client=client,
            out_dir=out_dir,
            text="Это тест управления длительностью речи. Мы проверяем, можно ли сделать реплику короче или длиннее.",
            short_tokens=180,
            long_tokens=360,
        )
        results["tests"]["duration_control"] = {
            "short": str(short_path.resolve()),
            "long": str(long_path.resolve()),
        }

        save_path = run_save_to_disk_test(client)
        results["tests"]["save_to_disk"] = save_path

        if args.reference_url:
            clone_path = run_reference_test(client, out_dir, args.reference_url, "url")
            results["tests"]["voice_clone_url"] = str(clone_path.resolve())
            results["tests"]["reference_usage_ab"] = run_reference_usage_test(
                client=client,
                out_dir=out_dir,
                reference_url=args.reference_url,
            )
            if args.reference_url_2:
                results["tests"]["voice_clone_two_refs"] = run_voice_cloning_test_two_refs(
                    client=client,
                    out_dir=out_dir,
                    reference_url_a=args.reference_url,
                    reference_url_b=args.reference_url_2,
                )
            else:
                warn("Two-reference cloning test skipped: --reference-url-2 not provided")
                results["tests"]["voice_clone_two_refs"] = "skipped"
        else:
            warn("Reference cloning test skipped: --reference-url not provided")
            results["tests"]["voice_clone_url"] = "skipped"
            results["tests"]["reference_usage_ab"] = "skipped"
            results["tests"]["voice_clone_two_refs"] = "skipped"

        if not args.skip_negative:
            run_negative_test_empty_text(client)
            results["tests"]["negative_empty_text"] = "passed"
        else:
            warn("Negative test skipped")
            results["tests"]["negative_empty_text"] = "skipped"

        print_summary(results)
        ok("All Vast serverless tests completed")

    except Exception as e:
        results["error"] = str(e)
        print_summary(results)
        fail(str(e))


if __name__ == "__main__":
    main()
