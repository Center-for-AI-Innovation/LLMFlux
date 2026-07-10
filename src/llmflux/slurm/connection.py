"""Connection helpers for llmflux serve jobs."""

import ipaddress
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# RFC 1123 hostname: dot-separated labels of alphanumerics and hyphens,
# no leading/trailing hyphen. Also matches IPv4 literals, which are
# checked separately against reserved ranges below.
_HOSTNAME_RE = re.compile(
    r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
)

# Engine servers always bind unprivileged ports; anything below 1024
# (ssh, smtp, ...) is an SSRF probe target, not a legitimate endpoint.
_MIN_PORT = 1024
_MAX_PORT = 65535


def _sanitize(value: str) -> str:
    """Strip non-ASCII bytes to prevent terminal control-character injection."""
    return value.encode("ascii", errors="replace").decode("ascii")


def _validate_node(node: str) -> None:
    """Reject node values that could redirect the request to an unintended target.

    The connection file lives on a shared filesystem; if another user managed
    to plant one, an unvalidated node/port would let them steer our HTTP
    request (SSRF) or inject options into the suggested ssh tunnel command.

    Raises:
        ValueError: if node is not a plausible cluster-node hostname.
    """
    if not node or not _HOSTNAME_RE.match(node):
        raise ValueError(
            f"node {node!r} is not a valid hostname "
            f"(letters, digits, hyphens, and dots only)"
        )
    labels = node.lower().split(".")
    if "localhost" in (labels[0], labels[-1]):
        raise ValueError(f"node {node!r} is a loopback address")
    try:
        addr = ipaddress.ip_address(node)
    except ValueError:
        pass  # a hostname, not an IP literal
    else:
        if (
            addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_unspecified
            or addr.is_reserved
        ):
            raise ValueError(
                f"node {node!r} is a loopback, link-local, or reserved address"
            )
    pattern = os.environ.get("LLMFLUX_NODE_PATTERN")
    if pattern:
        try:
            matched = re.fullmatch(pattern, node)
        except re.error as exc:
            raise ValueError(
                f"LLMFLUX_NODE_PATTERN is not a valid regular expression: {exc}"
            )
        if not matched:
            raise ValueError(
                f"node {node!r} does not match LLMFLUX_NODE_PATTERN ({pattern!r})"
            )


def _validate_port(port: int) -> None:
    """Reject ports outside the unprivileged range.

    Raises:
        ValueError: if port is not in [1024, 65535].
    """
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise ValueError(
            f"port {port} is outside the allowed range "
            f"{_MIN_PORT}-{_MAX_PORT} for serve endpoints"
        )


def _connection_file_path(job_id: str) -> Path:
    if not job_id.isdigit():
        raise ValueError(f"Invalid job ID: {job_id!r}")
    return Path.home() / ".llmflux" / "serve" / job_id / "connection.json"


def read_connection_info(job_id: str) -> Optional[dict]:
    """Return parsed connection file contents, or None if the file does not exist yet.

    Raises:
        PermissionError: if the file exists but is not owned by the current user.
    """
    path = _connection_file_path(job_id)
    if not path.exists():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_uid != os.getuid():
        raise PermissionError(
            f"Connection file {path} is owned by uid {stat.st_uid}, "
            f"not the current user (uid {os.getuid()}). "
            f"Refusing to read — the file may have been tampered with."
        )
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
    except urllib.error.HTTPError:
        # Any HTTP error (e.g. 500 from broken prometheus middleware) still
        # means the server process is running and accepting connections.
        return True
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
    try:
        info = read_connection_info(job_id)
    except PermissionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if info is None:
        print(f"Model is still loading. Waiting up to {wait_timeout}s...")
        try:
            info = wait_for_connection_file(job_id, timeout=wait_timeout)
        except PermissionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except TimeoutError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    missing = [f for f in ("node", "port") if not info.get(f)]
    if missing:
        print(
            f"Error: connection file is missing required fields: {', '.join(missing)}. "
            f"The file may be corrupt or incomplete. "
            f"Check logs with: llmflux logs {job_id}",
            file=sys.stderr,
        )
        return 1

    node    = _sanitize(info["node"])
    try:
        port = int(info["port"])
    except (ValueError, TypeError):
        print(
            f"Error: connection file has an invalid port value: {info['port']!r}. "
            f"Check logs with: llmflux logs {job_id}",
            file=sys.stderr,
        )
        return 1
    try:
        _validate_node(node)
        _validate_port(port)
    except ValueError as exc:
        print(
            f"Error: refusing to connect — {exc}. "
            f"The connection file may have been tampered with. "
            f"Check logs with: llmflux logs {job_id}",
            file=sys.stderr,
        )
        return 1
    model   = _sanitize(info.get("model", "unknown"))
    engine  = _sanitize(info.get("engine", "vllm"))
    api_key = _sanitize(info.get("api_key", ""))

    # Confirm the endpoint is reachable before showing info
    print(f"Pinging {node}:{port}...", end=" ", flush=True)
    reachable = _ping_endpoint(node, port, engine)
    if reachable:
        print("OK")
    else:
        print("unreachable (server may be behind a firewall — see note below).")

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
    print("from openai import OpenAI")
    print(f"client = OpenAI(base_url=\"{endpoint}\", api_key=\"{api_key}\")")
    print(f"response = client.chat.completions.create(")
    print(f"    model=\"{model}\",")
    print(f"    messages=[{{\"role\": \"user\", \"content\": \"Hello!\"}}]")
    print(f")")
    print()
    if not reachable:
        short_node = node.split(".")[0]
        print(f"Note: The compute node is not directly reachable on port {port} from this login node.")
        print(f"To use this endpoint, first run this in a separate terminal to open a port tunnel:")
        print()
        print(f"  ssh -N -L {port}:localhost:{port} {short_node} &")
        print()
        print(f"Then use this URL in your code:")
        print()
        print(f"  http://localhost:{port}/v1")
        print()

    return 0
