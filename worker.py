import math
from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 8000
MODEL_LOG_FILE = "/workspace/model.log"


def workload_calculator(payload: dict) -> float:
    """
    Для TTS лучше считать нагрузку от длины текста.
    Vast прямо пишет, что для non-LLM constant cost часто достаточен,
    но длина текста для TTS полезнее.
    """
    text = payload.get("text", "") or ""
    char_cost = max(1.0, len(text) / 200.0)  # 200 символов ~= 1 unit
    if payload.get("reference_audio"):
        char_cost += 0.5  # cloning дороже
    if payload.get("tokens"):
        char_cost += float(payload["tokens"]) / 500.0
    return float(round(char_cost, 3))


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
        on_info=["[INFO]"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()