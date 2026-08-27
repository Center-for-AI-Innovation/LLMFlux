#!/usr/bin/env python3
"""Minimal client for the pathology embedding server (serve.py).

Example:
    python client.py --endpoint http://gpub001:8000 --api-key ... \
        tile_001.png tile_002.png

    python client.py --endpoint http://gpub001:8000 --api-key ... \
        --prompts prompts.json tile_001.png tile_002.png
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests


def _encode(paths: List[str]) -> List[str]:
    encoded = []
    for path in paths:
        with open(path, "rb") as f:
            encoded.append(base64.b64encode(f.read()).decode("ascii"))
    return encoded


def embed(endpoint: str, image_paths: List[str], api_key: Optional[str] = None, timeout: float = 60.0) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(
        f"{endpoint.rstrip('/')}/embed",
        json={"images": _encode(image_paths)},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def classify(
    endpoint: str,
    image_paths: List[str],
    prompts: Dict[str, str],
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(
        f"{endpoint.rstrip('/')}/classify",
        json={"images": _encode(image_paths), "prompts": prompts},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image files to embed/classify")
    parser.add_argument("--endpoint", required=True, help="Server base URL, e.g. http://gpub001:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompts", help="JSON file of {label: prompt} to classify instead of embed")
    args = parser.parse_args()

    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())
        result = classify(args.endpoint, args.images, prompts, api_key=args.api_key)
    else:
        result = embed(args.endpoint, args.images, api_key=args.api_key)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
