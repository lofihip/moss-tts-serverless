import traceback

print("[WORKER] worker.py started", flush=True)

try:
    from vastai import (
        Worker,
        WorkerConfig,
        HandlerConfig,
        BenchmarkConfig,
        LogActionConfig,
    )
    print("[WORKER] imported vastai worker classes successfully", flush=True)
except Exception as e:
    print(f"[WORKER] failed to import vastai worker classes: {e}", flush=True)
    traceback.print_exc()
    raise

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 8000
MODEL_LOG_FILE = "/workspace/model.log"

print(f"[WORKER] MODEL_SERVER_URL={MODEL_SERVER_URL}", flush=True)
print(f"[WORKER] MODEL_SERVER_PORT={MODEL_SERVER_PORT}", flush=True)
print(f"[WORKER] MODEL_LOG_FILE={MODEL_LOG_FILE}", flush=True)


def workload_calculator(payload: dict) -> float:
    print(f"[WORKER] workload_calculator payload keys={list(payload.keys())}", flush=True)

    # Use generation steps as workload unit so Vast Perf is close to steps/sec (~it/s).
    raw_steps = payload.get("max_new_tokens", 1024)
    try:
        steps = float(raw_steps)
    except Exception:
        steps = 1024.0

    result = float(max(1.0, round(steps, 3)))
    print(f"[WORKER] workload_calculator result={result}", flush=True)
    return result


try:
    print("[WORKER] building WorkerConfig", flush=True)

    worker_config = WorkerConfig(
        model_server_url=MODEL_SERVER_URL,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=MODEL_LOG_FILE,
        handlers=[
            HandlerConfig(
                route="/generate/sync",
                allow_parallel_requests=False,
                max_queue_time=300.0,
                workload_calculator=workload_calculator,
                benchmark_config=BenchmarkConfig(
                    generator=lambda: {
                        "text": "Hello. This is a benchmark request for MOSS TTS.",
                        "return_base64": False,
                        "save_to_disk": False,
                        "max_new_tokens": 1024,
                    },
                    runs=3,
                    concurrency=1,
                ),
            ),
            HandlerConfig(
                route="/health",
                allow_parallel_requests=True,
                max_queue_time=30.0,
                workload_calculator=lambda payload: 0.01,
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=["Application startup complete."],
            on_error=[
                "Traceback (most recent call last):",
                "RuntimeError:",
                "CUDA out of memory",
            ],
            on_info=["[INFO]", "[BOOT]", "[SERVER]", "[WORKER]"],
        ),
    )

    print("[WORKER] WorkerConfig built successfully", flush=True)

except Exception as e:
    print(f"[WORKER] failed to build WorkerConfig: {e}", flush=True)
    traceback.print_exc()
    raise


if __name__ == "__main__":
    try:
        print("[WORKER] launching worker", flush=True)
        Worker(worker_config).run()
    except Exception as e:
        print(f"[WORKER] Worker run failed: {e}", flush=True)
        traceback.print_exc()
        raise
