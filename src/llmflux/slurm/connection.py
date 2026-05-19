"""Connection helpers for llmflux serve jobs."""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


def _connection_file_path(job_id: str) -> Path:
    return Path.home() / ".llmflux" / "serve" / str(job_id) / "connection.json"


def read_connection_info(job_id: str) -> Optional[dict]:
    """Return parsed connection file contents, or None if the file does not exist yet."""
    path = _connection_file_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def wait_for_connection_file(job_id: str, poll_interval: int = 5, timeout: int = 600) -> dict:
    """Block until the connection file appears, polling every poll_interval seconds.

    The file is written by the SLURM job only after the engine passes its health
    check, so it may appear several minutes after the job starts.

    Raises:
        TimeoutError: if the file does not appear within timeout seconds.
    """
    deadline = time.monotonic() + timeout
    dots = 0
    while time.monotonic() < deadline:
        info = read_connection_info(job_id)
        if info is not None:
            print()
            return info
        dots += 1
        print(f"\rWaiting for model to finish loading{'.' * (dots % 4):<3}", end="", flush=True)
        time.sleep(poll_interval)
    print()
    raise TimeoutError(
        f"Model did not finish loading within {timeout}s. "
        f"Check job logs with: llmflux logs {job_id}"
    )


def _ping_endpoint(node: str, port: int, engine: str) -> bool:
    """Return True if the engine's health endpoint responds on node:port."""
    health_path = "/health" if engine == "vllm" else "/api/version"
    url = f"http://{node}:{port}{health_path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def connect(job_id: str, local_port: int = 8000, wait_timeout: int = 600) -> int:
    """Resolve and display the endpoint for a running serve job.

    Waits for the model to finish loading if needed, pings the endpoint
    directly on the cluster network, then prints the URL, API key, and
    example code. No SSH tunnel is created — the head node can reach the
    compute node directly on the cluster's internal network.

    Args:
        job_id: SLURM job ID of a running serve job.
        local_port: Unused — kept for CLI compatibility. Access is direct to node:port.
        wait_timeout: Seconds to wait for the model to finish loading.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    info = read_connection_info(job_id)
    if info is None:
        print(f"Model is still loading. Waiting up to {wait_timeout}s...")
        try:
            info = wait_for_connection_file(job_id, timeout=wait_timeout)
        except TimeoutError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    node    = info["node"]
    port    = int(info["port"])
    model   = info.get("model", "unknown")
    engine  = info.get("engine", "vllm")
    api_key = info.get("api_key", "")

    # Confirm the endpoint is reachable before showing info
    print(f"Pinging {node}:{port}...", end=" ", flush=True)
    if _ping_endpoint(node, port, engine):
        print("OK")
    else:
        print(f"unreachable.\nThe endpoint did not respond. "
              f"Check logs with: llmflux logs {job_id}", file=sys.stderr)
        return 1

    endpoint = f"http://{node}:{port}/v1"

    print()
    print("Service is ready.")
    print()
    print(f"  Endpoint:  {endpoint}")
    print(f"  API Key:   {api_key}")
    print(f"  Model:     {model}")
    print(f"  Engine:    {engine}")
    print()
    print("Example usage:")
    print()
    print("  from openai import OpenAI")
    print(f"  client = OpenAI(base_url=\"{endpoint}\", api_key=\"{api_key}\")")
    print(f"  response = client.chat.completions.create(")
    print(f"      model=\"{model}\",")
    print(f"      messages=[{{\"role\": \"user\", \"content\": \"Hello!\"}}]")
    print(f"  )")
    print()

    return 0
