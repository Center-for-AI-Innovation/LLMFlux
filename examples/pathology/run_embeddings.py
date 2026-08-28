#!/usr/bin/env python3
"""Run CONCH or MUSK over a directory of pathology image tiles.

Standalone script — does not use any LLMFlux package code (see README.md for
why). Writes one row per image to a CSV; rerun with the same --output to
resume, skipping images already present.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from models import ADAPTERS, resolve_adapter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover_images(input_dir: str) -> List[str]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    paths = sorted(
        str(p) for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(
            f"No images with extensions {sorted(IMAGE_EXTENSIONS)} found under {input_dir}"
        )
    return paths


def load_done(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    try:
        return set(pd.read_csv(output_path)["image_path"])
    except Exception:
        logger.warning(f"Could not read existing output {output_path}; starting fresh")
        return set()


def classify(image_embs: np.ndarray, text_embs: np.ndarray, labels: List[str]):
    """Cosine-similarity zero-shot classification against a fixed label set."""
    image_norm = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_norm = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
    sims = image_norm @ text_norm.T
    exp = np.exp(sims - sims.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    best = probs.argmax(axis=1)
    return [labels[i] for i in best], probs[np.arange(len(best)), best]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            f"One of {sorted(ADAPTERS)}, 'openclip:<hf-repo>' / "
            "'openclip:<arch>@<hf-repo>/<filename>' for any OpenCLIP model, "
            "or 'hfclip:<hf-repo>' for any transformers CLIPModel repo "
            "(e.g. 'hfclip:openai/clip-vit-base-patch32')"
        ),
    )
    parser.add_argument("--input-dir", required=True, help="Directory of image tiles (searched recursively)")
    parser.add_argument("--output", required=True, help="CSV path for results; rerun to resume")
    parser.add_argument("--text-prompts", help="JSON file of {label: prompt} for zero-shot classification")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Flush to --output every N images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-token", default=None, help="Falls back to HF_TOKEN env var")
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir)
    done = load_done(args.output)
    remaining = [p for p in image_paths if p not in done]
    logger.info(f"{len(image_paths)} images found, {len(done)} already done, {len(remaining)} remaining")

    if not remaining:
        logger.info("Nothing to do.")
        return

    logger.info(f"Loading {args.model} on {args.device}")
    adapter = resolve_adapter(args.model, device=args.device, hf_token=args.hf_token)

    labels: Optional[List[str]] = None
    text_embs: Optional[np.ndarray] = None
    if args.text_prompts:
        with open(args.text_prompts) as f:
            prompts: Dict[str, str] = json.load(f)
        if adapter.encode_texts is None:
            raise ValueError(f"{args.model} adapter does not support text encoding")
        labels = list(prompts.keys())
        text_embs = adapter.encode_texts(list(prompts.values()))
        logger.info(f"Zero-shot classification against {len(labels)} labels: {labels}")

    rows: List[Dict] = []
    batch_paths: List[str] = []
    batch_tensors: List[torch.Tensor] = []

    def flush():
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        if os.path.exists(args.output):
            new_df = pd.concat([pd.read_csv(args.output), new_df], ignore_index=True)
        new_df.to_csv(args.output, index=False)
        rows.clear()

    def process_batch():
        if not batch_paths:
            return
        batch = torch.stack(batch_tensors)
        embs = adapter.encode_images(batch)
        if labels is not None:
            pred_labels, scores = classify(embs, text_embs, labels)
            for path, pred_label, score in zip(batch_paths, pred_labels, scores):
                rows.append({"image_path": path, "predicted_label": pred_label, "score": float(score)})
        else:
            for path, emb in zip(batch_paths, embs):
                rows.append({"image_path": path, "embedding": json.dumps(emb.tolist())})
        batch_paths.clear()
        batch_tensors.clear()

    processed_since_checkpoint = 0
    for path in tqdm(remaining, desc=f"Embedding ({args.model})"):
        try:
            image = Image.open(path).convert("RGB")
            batch_tensors.append(adapter.preprocess(image))
            batch_paths.append(path)
        except Exception as e:
            logger.warning(f"Skipping unreadable image {path}: {e}")
            continue

        if len(batch_paths) >= args.batch_size:
            process_batch()

        processed_since_checkpoint += 1
        if processed_since_checkpoint >= args.checkpoint_every:
            process_batch()
            flush()
            processed_since_checkpoint = 0

    process_batch()
    flush()
    logger.info(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
