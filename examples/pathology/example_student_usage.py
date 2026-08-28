#!/usr/bin/env python3
"""Student quickstart for the pathology embedding server.

See STUDENT_GUIDE.md for the full walkthrough (what /embed vs /classify
return, troubleshooting, using your own images/prompts) - this script is the
runnable version of that walkthrough.

Fill in ENDPOINT and API_KEY below (the organizers will give you both), then:

    pip install requests pillow   # pillow is only needed for the placeholder
                                   # tiles this demo generates - see note below
    python example_student_usage.py

This is everything you need on your own machine: no CONCH/MUSK, no torch, no
GPU. Those all run on the server; you're just sending it HTTP requests, the
same way client.py does.
"""

import base64
import io

import requests

ENDPOINT = "http://<node>:<port>"  # from the organizers, or connection.json
API_KEY = "<api-key>"  # from the organizers, or connection.json


def image_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def load_your_own_image(path: str) -> str:
    """This is all you need for real data - no PIL, no preprocessing.
    The server handles decoding, resizing, and normalization."""
    with open(path, "rb") as f:
        return image_bytes_to_b64(f.read())


def make_placeholder_tiles_b64():
    """Three differently-colored synthetic stand-ins, so /classify has more
    than one image to tell apart. These are NOT real pathology images - swap
    this for [load_your_own_image(p) for p in ["tile1.png", "tile2.png"]]
    once you have real data."""
    from PIL import Image

    colors = [(180, 120, 160), (230, 210, 220), (90, 40, 60)]
    tiles = []
    for color in colors:
        tile = Image.new("RGB", (224, 224), color)
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        tiles.append(image_bytes_to_b64(buf.getvalue()))
    return tiles


def check_health():
    """Fail fast with a clear message instead of a confusing stack trace if
    the endpoint is wrong, the server isn't up yet, or the API key is stale."""
    try:
        response = requests.get(f"{ENDPOINT}/health", timeout=10)
    except requests.exceptions.ConnectionError as e:
        raise SystemExit(
            f"Could not reach {ENDPOINT} - check ENDPOINT matches what the "
            f"organizers gave you, and that the server is still running ({e})"
        ) from e
    if not response.ok:
        raise SystemExit(f"Server at {ENDPOINT} responded with HTTP {response.status_code}: {response.text}")
    info = response.json()
    print(f"Connected. Server is running model={info['model']} on device={info['device']}.\n")


def _post(path, payload):
    response = requests.post(
        f"{ENDPOINT}{path}",
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    if response.status_code == 401:
        raise SystemExit("Got HTTP 401 - check API_KEY matches what the organizers gave you.")
    response.raise_for_status()
    return response.json()


def embed(images_b64):
    return _post("/embed", {"images": images_b64})["embeddings"]


def classify(images_b64, prompts):
    return _post("/classify", {"images": images_b64, "prompts": prompts})["results"]


def main():
    check_health()

    tiles = make_placeholder_tiles_b64()

    embeddings = embed(tiles)
    print(f"/embed: got {len(embeddings)} embeddings, each {len(embeddings[0])}-dimensional.")
    print("Use these for clustering/kNN/training your own classifier - see STUDENT_GUIDE.md.\n")

    prompts = {
        "tumor": "a histopathology image of tumor tissue",
        "stroma": "a histopathology image of stroma",
        "necrosis": "a histopathology image of necrotic tissue",
    }
    results = classify(tiles, prompts)
    for i, r in enumerate(results):
        print(f"/classify tile {i}: predicted '{r['predicted_label']}' (score={r['score']:.3f})")
    print("(these predictions are meaningless here - solid-color placeholders, not real tissue)\n")

    print("Next steps:")
    print("  1. Replace make_placeholder_tiles_b64() with your own image paths")
    print("     (see load_your_own_image)")
    print("  2. Edit `prompts` above to the labels your team actually cares about")
    print("  3. See STUDENT_GUIDE.md for what to do with /embed output beyond zero-shot classify")


if __name__ == "__main__":
    main()
