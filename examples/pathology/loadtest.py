#!/usr/bin/env python3
"""Concurrency smoke test for serve.py.

Simulates N teams submitting requests to a running pathology server at the
same time and checks: everyone gets back a correctly-shaped, distinct result,
nothing crashes, and concurrent requests are actually being batched together
server-side rather than serialized one-GPU-call-per-request.

Usage:
    python loadtest.py --endpoint http://localhost:8000 --teams 20 --images-per-team 3
"""

import argparse
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image


def make_test_image(team_id: int) -> str:
    """A solid color distinct per team, repeated for every image that team
    sends. This lets us check for cross-request mix-ups under concurrent
    batching without knowing anything about the model's embedding function:
    all of a team's returned embeddings should match each other (same input),
    and should differ from another team's (different input)."""
    color = ((team_id * 37) % 256, (team_id * 59) % 256, (team_id * 83) % 256)
    image = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def one_team(endpoint: str, team_id: int, images_per_team: int, api_key):
    image = make_test_image(team_id)
    images = [image] * images_per_team
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    start = time.perf_counter()
    response = requests.post(f"{endpoint}/embed", json={"images": images}, headers=headers, timeout=60)
    latency = time.perf_counter() - start
    response.raise_for_status()
    embeddings = np.array(response.json()["embeddings"])
    if len(embeddings) != images_per_team:
        raise AssertionError(f"team {team_id}: expected {images_per_team} embeddings, got {len(embeddings)}")
    if images_per_team > 1 and not np.allclose(embeddings, embeddings[0], atol=1e-4):
        raise AssertionError(
            f"team {team_id}: got different embeddings for identical images "
            f"(cross-request mix-up under concurrent batching)"
        )
    return team_id, latency, embeddings[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--teams", type=int, default=20)
    parser.add_argument("--images-per-team", type=int, default=3)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")

    before = requests.get(f"{endpoint}/health", timeout=10).json()

    print(f"Firing {args.teams} concurrent teams x {args.images_per_team} images each...")
    latencies = []
    errors = []
    results_by_team = {}
    with ThreadPoolExecutor(max_workers=args.teams) as pool:
        futures = [
            pool.submit(one_team, endpoint, team_id, args.images_per_team, args.api_key)
            for team_id in range(args.teams)
        ]
        for future in as_completed(futures):
            try:
                team_id, latency, embeddings = future.result()
                latencies.append(latency)
                results_by_team[team_id] = embeddings
            except Exception as e:  # noqa: BLE001 - collecting per-team failures, not re-raising here
                errors.append(str(e))

    after = requests.get(f"{endpoint}/health", timeout=10).json()

    print(f"\n{len(results_by_team)}/{args.teams} teams succeeded, {len(errors)} errors")
    for e in errors:
        print(f"  error: {e}")

    team_ids = list(results_by_team)
    collisions = [
        (a, b) for i, a in enumerate(team_ids) for b in team_ids[i + 1:]
        if np.allclose(results_by_team[a], results_by_team[b], atol=1e-4)
    ]
    if collisions:
        print(f"CROSS-TALK: {len(collisions)} team pair(s) got identical embeddings for different inputs: {collisions}")
        errors.append(f"{len(collisions)} cross-team embedding collision(s)")
    elif results_by_team:
        print("No cross-request mix-ups: every team's embedding is distinct from every other team's.")

    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"Latency: min={min(latencies):.3f}s p50={p50:.3f}s p95={p95:.3f}s max={max(latencies):.3f}s")

    batches = after["batches_run"] - before["batches_run"]
    requests_served = after["requests_run"] - before["requests_run"]
    print(f"Server-side: {requests_served} requests were served across {batches} GPU batch(es)")
    if batches and requests_served > batches:
        print(f"  -> batching is working: {requests_served / batches:.1f} requests/batch on average")
    elif requests_served:
        print(
            "  -> no batching observed (each request ran as its own GPU call); "
            "consider raising PATHOLOGY_MAX_WAIT_MS if teams submit near-simultaneously"
        )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
