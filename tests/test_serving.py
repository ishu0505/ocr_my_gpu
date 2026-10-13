"""Meaningful SSE, bounded concurrency, failure and timing tests without real engines."""

import asyncio
import json
from pathlib import Path

import pytest

from gpu_my_ocr.benchmark.concurrency import bounded_requests, concurrency_metrics
from gpu_my_ocr.benchmark.server_process import server_command
from gpu_my_ocr.engines.http import OpenAIImageEngine, RequestFailure, consume_stream
from gpu_my_ocr.models import InferenceRequest


def sse(payload):
    return ("data: " + json.dumps(payload) + "\n").encode()


def test_sse_ttft_usage_and_finish_reason():
    lines = [
        sse({"choices": [{"delta": {"role": "assistant"}}]}),
        sse({"choices": [{"delta": {"content": "hello"}}]}),
        sse({"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}),
        sse({"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 20}}),
        b"data: [DONE]\n",
    ]
    response = consume_stream(lines, 10, clock=lambda: 10.5)
    assert response["text"] == "hello world"
    assert response["time_to_first_token_ms"] == 500
    assert response["output_tokens"] == 3
    assert response["input_tokens"] == 20
    assert not response["truncated"]


def test_incomplete_or_error_stream_retains_partial_predictions():
    start = sse({"choices": [{"delta": {"content": "partial"}}]})
    with pytest.raises(RequestFailure) as raised:
        consume_stream([start], 0)
    assert raised.value.partial_text == "partial"
    with pytest.raises(RequestFailure):
        consume_stream([sse({"error": {"message": "OOM"}})], 0)
    output = consume_stream(
        [sse({"choices": [{"delta": {}, "finish_reason": "length"}]}), b"data: [DONE]"], 0
    )
    assert output["truncated"]
    assert output["output_tokens"] is None


def test_payload_fixed_controls_and_optional_schema(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"PNG bytes")
    request = InferenceRequest("test", image, "original prompt", 1024)
    engine = OpenAIImageEngine("http://127.0.0.1:8008", "model", "vllm")
    body = engine.payload(request)
    assert body["temperature"] == 0 and body["max_tokens"] == 1024
    assert body["messages"][0]["content"][1]["text"] == request.prompt
    assert "response_format" not in body
    constrained = OpenAIImageEngine(
        "http://127.0.0.1:8008", "model", "vllm", schema={"type": "object"}
    )
    assert constrained.payload(request)["response_format"]["json_schema"]["schema"] == {
        "type": "object"
    }
    with pytest.raises(ValueError):
        OpenAIImageEngine("https://external.example", "model", "vllm")


def test_client_concurrency_is_bounded_and_wall_time_is_not_sum():
    active = peak = 0

    async def operation(item):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"item": item}

    results, elapsed = asyncio.run(bounded_requests(list(range(8)), 2, operation))
    assert peak == 2 and len(results) == 8
    assert all(r["client_total_ms"] >= r["client_queue_ms"] for r in results)
    assert elapsed > 0
    with pytest.raises(ValueError):
        asyncio.run(bounded_requests([], 0, operation))


def test_metrics_distinguish_failures_missing_usage_and_correctness():
    successful = {
        "status": "ok",
        "end_to_end_ms": 2000,
        "output_tokens": 100,
        "time_to_first_token_ms": 100,
        "score": {"schema_valid": True, "page_success": False},
    }
    failed = {"status": "oom", "score": {"schema_valid": False, "page_success": False}}
    metrics = concurrency_metrics([successful, failed], 4)
    assert metrics["successful_pages_per_second"] == 0.25
    assert metrics["correct_pages_per_second"] == 0
    assert metrics["output_tokens_per_second"] == 25
    assert metrics["failure_rate"] == 0.5 and metrics["ooms"] == 1
    assert metrics["ttft_mean_seconds"] == 0.1
    missing = {**successful, "output_tokens": None}
    assert concurrency_metrics([missing], 4)["output_tokens_per_second"] is None


def test_server_commands_preserve_precision_model_and_image_policy():
    for name in ("vllm", "sglang"):
        command = server_command(name, 8008)
        assert "bfloat16" in command and "4096" in command and "0.8" in command
        assert command[command.index("--cpu-offload-gb") + 1] == "0"
        assert "1310720" in " ".join(command)
        assert "127.0.0.1" in command


def test_metric_deltas_use_server_histograms_without_inventing_missing_data():
    from gpu_my_ocr.benchmark.serving_report import engine_metric_delta

    before = 'vllm:request_queue_time_seconds_sum{engine="0"} 2\nvllm:request_queue_time_seconds_count{engine="0"} 4\nvllm:num_preemptions_total{engine="0"} 0\n'
    after = 'vllm:request_queue_time_seconds_sum{engine="0"} 5\nvllm:request_queue_time_seconds_count{engine="0"} 10\nvllm:num_preemptions_total{engine="0"} 2\n'
    delta = engine_metric_delta(before, after)
    assert delta["queue_mean_seconds_by_metric"]["vllm:request_queue_time_seconds_sum"] == 0.5
    assert delta["preemptions"] == 2
    assert engine_metric_delta("", "")["queue_mean_seconds_by_metric"] is None


def test_skipped_requests_are_not_counted_as_attempted_ooms():
    skipped = {"status": "skipped", "score": {"schema_valid": False, "page_success": False}}
    failed = {"status": "oom", "score": {"schema_valid": False, "page_success": False}}
    metrics = concurrency_metrics([skipped, failed], 1)
    assert metrics["attempted_requests"] == 1 and metrics["skipped_requests"] == 1
    assert metrics["failure_rate"] == 1 and metrics["ooms"] == 1


def test_failed_transport_with_correct_partial_json_is_not_completed_correct_throughput():
    failed = {"status": "failed", "score": {"schema_valid": True, "page_success": True}}
    assert concurrency_metrics([failed], 1)["correct_pages_per_second"] == 0


def test_sglang_discovers_only_installed_local_toolkit(tmp_path: Path):
    from gpu_my_ocr.benchmark.server_process import server_environment

    assert "CUDA_HOME" not in server_environment("sglang", tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvcc").write_text("local compiler")
    assert server_environment("sglang", tmp_path)["CUDA_HOME"] == str(tmp_path)
    assert "CUDA_HOME" not in server_environment("vllm", tmp_path)


def test_legacy_sglang_flags_preserve_shared_controls():
    command = server_command("sglang", 8008, "0.5.6")
    assert "--disable-cuda-graph" in command
    assert "--cuda-graph-backend-decode" not in command
    assert "--mm-preprocess-cache-size-mb" not in command
    assert "--mm-attention-backend" in command and "sdpa" in command
    assert "--disable-radix-cache" in command and "bfloat16" in command


def test_legacy_cuda_runtime_root_does_not_require_a_host_compiler(tmp_path: Path):
    from gpu_my_ocr.benchmark.server_process import server_environment

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libcudart.so.12").write_text("runtime library")
    assert server_environment("sglang", tmp_path)["CUDA_HOME"] == str(tmp_path)


def test_processor_overlay_changes_only_bounds_and_never_checkpoint(tmp_path: Path):
    from gpu_my_ocr.benchmark.processor_overlay import create_processor_overlay

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    original = {"size": {"shortest_edge": 65536, "longest_edge": 16777216}, "patch_size": 16}
    (snapshot / "preprocessor_config.json").write_text(json.dumps(original))
    (snapshot / "model.safetensors").write_bytes(b"frozen weights")
    output = create_processor_overlay(snapshot, tmp_path / "overlay")
    modified = json.loads((output / "preprocessor_config.json").read_text())
    assert modified["size"] == {"shortest_edge": 262144, "longest_edge": 1310720}
    assert modified["patch_size"] == original["patch_size"]
    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").read_bytes() == b"frozen weights"
    assert json.loads((snapshot / "preprocessor_config.json").read_text()) == original


def test_sglang_stage_histograms_keep_waiting_separate_from_decoding():
    from gpu_my_ocr.benchmark.serving_report import engine_metric_delta

    before = 'sglang:per_stage_req_latency_seconds_sum{stage="prefill_waiting"} 2\nsglang:per_stage_req_latency_seconds_count{stage="prefill_waiting"} 1\nsglang:num_retractions_sum 0\n'
    after = 'sglang:per_stage_req_latency_seconds_sum{stage="prefill_waiting"} 6\nsglang:per_stage_req_latency_seconds_count{stage="prefill_waiting"} 3\nsglang:per_stage_req_latency_seconds_sum{stage="decoding"} 100\nsglang:per_stage_req_latency_seconds_count{stage="decoding"} 3\nsglang:num_retractions_sum 2\n'
    metrics = engine_metric_delta(before, after)
    assert metrics["prefill_waiting_mean_seconds"] == 2
    assert metrics["prefill_waiting_sample_count"] == 2
    assert metrics["sglang_retractions"] == 2
    assert metrics["preemptions"] is None
