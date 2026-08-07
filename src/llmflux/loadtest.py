#!/usr/bin/env python3
"""Concurrency load testing for a running `llmflux serve` endpoint.

Ramps a pool of simulated users against one endpoint and reports, per
concurrency level, client-side latency percentiles alongside the server's own
queue depth and KV-cache pressure. Use it to find the level at which p95
latency crosses an interactive budget — that level is "too many users".

Unlike `llmflux benchmark`, which submits a SLURM job and drives it one request
at a time, this runs locally against an endpoint that already exists and keeps
many requests in flight, which is what exercises vLLM's continuous batching.
"""

import json
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .benchmark_utils import VllmMetricsScraper
from .converters.vision import encode_image, get_image_mime_type

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)

IMAGE_PROMPT = (
    "Describe this image in detail. List every object you can identify and "
    "explain what is happening in the scene."
)
TEXT_PROMPT = (
    "Explain how continuous batching works in an LLM inference server, "
    "and why it changes throughput under concurrent load."
)


class RequestSample:
    """Timing for a single completion, or the error that ended it."""

    __slots__ = ("ttft_ms", "total_ms", "output_tokens", "error")

    def __init__(self, ttft_ms=None, total_ms=None, output_tokens=0, error=None):
        self.ttft_ms = ttft_ms
        self.total_ms = total_ms
        self.output_tokens = output_tokens
        self.error = error


def image_data_url(path: str) -> str:
    """Return a data: URL for an image, matching the vision JSONL converter."""
    return f"data:{get_image_mime_type(path)};base64,{encode_image(path)}"


def build_payload(
    model: str,
    data_urls: Optional[List[str]] = None,
    max_tokens: int = 300,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a streaming chat-completion request body.

    Streaming is required: it is the only way to measure time-to-first-token
    separately from total latency, and TTFT is what degrades first under load.
    """
    data_urls = data_urls or []
    if prompt is None:
        prompt = IMAGE_PROMPT if data_urls else TEXT_PROMPT

    if data_urls:
        content: Any = [{"type": "text", "text": prompt}]
        for url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content = prompt

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def send_request(session, url: str, payload: Dict[str, Any], timeout: float) -> RequestSample:
    """Issue one streaming completion, timing first token and full response."""
    start = time.perf_counter()
    ttft = None
    content_chunks = 0
    usage_tokens = 0
    try:
        response = session.post(url, json=payload, stream=True, timeout=timeout)
        if response.status_code != 200:
            detail = response.text[:200].replace("\n", " ")
            return RequestSample(error=f"HTTP {response.status_code}: {detail}")
        for raw in response.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens", 0) or 0
            for choice in chunk.get("choices") or []:
                if (choice.get("delta") or {}).get("content"):
                    content_chunks += 1
    except Exception as exc:  # noqa: BLE001 - any client failure is a data point
        return RequestSample(error=f"{type(exc).__name__}: {exc}")

    return RequestSample(
        ttft_ms=ttft,
        total_ms=(time.perf_counter() - start) * 1000,
        # Prefer the server's count; fall back to counting content deltas.
        output_tokens=usage_tokens or content_chunks,
    )


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 1)


def _user_loop(deadline, session_factory, url, payload, think_time, timeout, out, lock):
    """One simulated user: request, pause like a human, repeat until the phase ends."""
    session = session_factory()
    while time.monotonic() < deadline:
        sample = send_request(session, url, payload, timeout)
        with lock:
            out.append(sample)
        if think_time:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(think_time, remaining))


def run_phase(
    concurrency: int,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    phase_seconds: float,
    think_time: float = 0.0,
    timeout: float = 300.0,
    metrics_base_url: Optional[str] = None,
    metrics_interval: float = 2.0,
) -> Dict[str, Any]:
    """Hold `concurrency` users against the endpoint and summarize the result."""
    samples: List[RequestSample] = []
    lock = threading.Lock()

    scraper = None
    if metrics_base_url:
        scraper = VllmMetricsScraper(base_url=metrics_base_url, interval_seconds=metrics_interval)
        scraper.start()

    def session_factory():
        session = requests.Session()
        session.headers.update(headers)
        return session

    started = time.monotonic()
    deadline = started + phase_seconds
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(
                _user_loop, deadline, session_factory, url, payload,
                think_time, timeout, samples, lock,
            )
    elapsed = time.monotonic() - started

    server_metrics = scraper.stop() if scraper else {}

    completed = [s for s in samples if s.error is None]
    errors = [s for s in samples if s.error is not None]
    ttfts = [s.ttft_ms for s in completed if s.ttft_ms is not None]
    totals = [s.total_ms for s in completed if s.total_ms is not None]
    output_tokens = sum(s.output_tokens for s in completed)

    return {
        "concurrency": concurrency,
        "completed": len(completed),
        "errors": len(errors),
        "error_examples": sorted({s.error for s in errors})[:3],
        "requests_per_sec": round(len(completed) / elapsed, 3) if elapsed else None,
        "output_tokens_per_sec": round(output_tokens / elapsed, 1) if elapsed else None,
        "ttft_p50_ms": _percentile(ttfts, 50),
        "ttft_p95_ms": _percentile(ttfts, 95),
        "e2e_p50_ms": _percentile(totals, 50),
        "e2e_p95_ms": _percentile(totals, 95),
        "server": server_metrics,
    }


def run_ramp(
    endpoint: str,
    model: str,
    api_key: Optional[str] = None,
    levels: Any = DEFAULT_LEVELS,
    phase_seconds: float = 60.0,
    think_time: float = 0.0,
    max_tokens: int = 300,
    images: Optional[List[str]] = None,
    timeout: float = 300.0,
    cooldown: float = 10.0,
    scrape_metrics: bool = True,
    metrics_interval: float = 2.0,
    on_phase_start=None,
    on_phase_end=None,
) -> List[Dict[str, Any]]:
    """Run each concurrency level in turn and return one result row per level.

    Args:
        endpoint: Base URL ending in /v1, as printed by `llmflux connect`
        model: Exact model string the server reports (see GET /v1/models)
        api_key: Bearer token for the endpoint
        levels: Concurrency levels to ramp through
        phase_seconds: Seconds of steady load at each level
        think_time: Pause between a user's requests. 0 means worst-case burst,
            where every simulated user always has a request in flight.
        images: Image paths to attach, exercising the vision path
        cooldown: Idle seconds between levels so the queue drains
    """
    base = endpoint.rstrip("/")
    url = f"{base}/chat/completions"
    # /metrics sits at the server root, not under /v1, and vLLM leaves it
    # unauthenticated — only /v1 routes are behind --api-key.
    metrics_base = base[:-3].rstrip("/") if scrape_metrics and base.endswith("/v1") else None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data_urls = [image_data_url(path) for path in (images or [])]
    payload = build_payload(model, data_urls, max_tokens)

    rows = []
    level_list = list(levels)
    for index, level in enumerate(level_list):
        if on_phase_start:
            on_phase_start(index, len(level_list), level)
        row = run_phase(
            concurrency=level,
            url=url,
            payload=payload,
            headers=headers,
            phase_seconds=phase_seconds,
            think_time=think_time,
            timeout=timeout,
            metrics_base_url=metrics_base,
            metrics_interval=metrics_interval,
        )
        rows.append(row)
        if on_phase_end:
            on_phase_end(row)
        if cooldown and index < len(level_list) - 1:
            time.sleep(cooldown)
    return rows


def warm_up(endpoint: str, model: str, api_key: Optional[str], timeout: float = 300.0) -> RequestSample:
    """Send one short request so a bad key or model name fails before the ramp."""
    session = requests.Session()
    if api_key:
        session.headers["Authorization"] = f"Bearer {api_key}"
    payload = build_payload(model, max_tokens=8)
    return send_request(session, f"{endpoint.rstrip('/')}/chat/completions", payload, timeout)


def format_report(rows: List[Dict[str, Any]], slo_ms: Optional[float] = None) -> str:
    """Render the ramp as a table plus a plain-language reading of the knee."""
    columns = [
        ("conc", lambda r: r["concurrency"], 5),
        ("done", lambda r: r["completed"], 5),
        ("err", lambda r: r["errors"], 4),
        ("req/s", lambda r: r["requests_per_sec"], 7),
        ("tok/s", lambda r: r["output_tokens_per_sec"], 8),
        ("ttft p50", lambda r: r["ttft_p50_ms"], 9),
        ("ttft p95", lambda r: r["ttft_p95_ms"], 9),
        ("e2e p50", lambda r: r["e2e_p50_ms"], 9),
        ("e2e p95", lambda r: r["e2e_p95_ms"], 9),
        ("running", lambda r: r["server"].get("vllm_effective_batch_size_avg"), 8),
        ("waiting", lambda r: r["server"].get("vllm_num_requests_waiting_max"), 8),
        ("kv max%", lambda r: r["server"].get("vllm_kv_cache_peak_pct"), 8),
    ]

    lines = [
        "  ".join(head.rjust(width) for head, _, width in columns),
        "  ".join("-" * width for _, _, width in columns),
    ]
    for row in rows:
        cells = []
        for _, getter, width in columns:
            value = getter(row)
            cells.append(("-" if value is None else str(value)).rjust(width))
        flags = ""
        if slo_ms and row["e2e_p95_ms"] and row["e2e_p95_ms"] > slo_ms:
            flags += "  <-- over SLO"
        if row["errors"]:
            flags += "  <-- errors"
        lines.append("  ".join(cells) + flags)

    lines.append("")
    over_budget = [r for r in rows if slo_ms and r["e2e_p95_ms"] and r["e2e_p95_ms"] > slo_ms]
    if over_budget:
        ceiling = over_budget[0]["concurrency"]
        lines.append(
            f"First level over the {slo_ms:.0f}ms p95 budget: {ceiling} concurrent users. "
            f"The safe ceiling is the level below it."
        )
    elif slo_ms:
        lines.append(f"No level exceeded the {slo_ms:.0f}ms p95 budget — try higher levels.")

    failed = [r for r in rows if r["errors"]]
    if failed:
        lines.append(
            f"Errors first appeared at concurrency {failed[0]['concurrency']}: "
            f"{failed[0]['error_examples']}"
        )

    with_throughput = [r for r in rows if r["output_tokens_per_sec"]]
    if with_throughput:
        peak_tokens = max(r["output_tokens_per_sec"] for r in with_throughput)
        # The knee is the cheapest level that already reaches the plateau, not the
        # highest level tested — beyond it, throughput is flat and only latency grows.
        knee = min(
            (r for r in with_throughput if r["output_tokens_per_sec"] >= 0.95 * peak_tokens),
            key=lambda r: r["concurrency"],
        )
        lines.append(
            f"Peak output throughput: {peak_tokens} tok/s, first reached at concurrency "
            f"{knee['concurrency']}. Past that, added users buy latency, not throughput."
        )
        if knee["e2e_p50_ms"]:
            latency_s = knee["e2e_p50_ms"] / 1000
            lines.append(
                f"At that level a request takes ~{latency_s:.1f}s, so with users pausing T "
                f"seconds between prompts, sustainable users ≈ {knee['concurrency']} × "
                f"(T + {latency_s:.1f}) / {latency_s:.1f}."
            )

    return "\n".join(lines)


def save_results(rows: List[Dict[str, Any]], path: str) -> None:
    """Write the raw per-level rows as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2))
