#!/usr/bin/env python3
"""Student quickstart for the pathology embedding server.

Fill in ENDPOINT and API_KEY below (the organizers will give you both), then:

    pip install requests pillow   # pillow is only needed for the placeholder
                                   # image this demo generates - see note below
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


def make_placeholder_tile_b64() -> str:
    """A synthetic stand-in image so this script runs before you have real
    data wired up. It is NOT a real pathology image - swap it for
    load_your_own_image("path/to/your_tile.png") once you're ready."""
    from PIL import Image

    tile = Image.new("RGB", (224, 224), (180, 120, 160))
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return image_bytes_to_b64(buf.getvalue())


def embed(images_b64):
    response = requests.post(
        f"{ENDPOINT}/embed",
        json={"images": images_b64},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def classify(images_b64, prompts):
    response = requests.post(
        f"{ENDPOINT}/classify",
        json={"images": images_b64, "prompts": prompts},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["results"]


def main():
    tile = make_placeholder_tile_b64()

    embeddings = embed([tile])
    print(f"/embed: got a {len(embeddings[0])}-dimensional embedding for the tile.")

    prompts = {
        "tumor": "a histopathology image of tumor tissue",
        "stroma": "a histopathology image of stroma",
        "necrosis": "a histopathology image of necrotic tissue",
    }
    results = classify([tile], prompts)
    print(f"/classify: predicted '{results[0]['predicted_label']}' (score={results[0]['score']:.3f})")
    print("(this prediction is meaningless here - it's a solid-color placeholder, not real tissue)")

    print("\nNext: replace make_placeholder_tile_b64() with load_your_own_image('your_tile.png'),")
    print("and edit `prompts` above to the labels your team actually cares about.")


if __name__ == "__main__":
    main()
