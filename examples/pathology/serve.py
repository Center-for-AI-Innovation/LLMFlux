#!/usr/bin/env python3
"""Persistent CONCH/MUSK embedding server for concurrent use by many teams.

Loads one model once and keeps it resident on the GPU, then serves
concurrent HTTP requests through a single background worker thread that
micro-batches images across requests before each forward pass. This is the
piece that answers "stage the model once, N teams hit it at the same time" —
run_embeddings.py is a one-shot batch script for a single team/dataset and
doesn't fit that shape.

Endpoints:
  GET  /health                        -> {"status": "ok", "model": "..."}
  POST /embed    {"images": [b64,...]} -> {"embeddings": [[float,...],...]}
  POST /classify {"images": [b64,...], "prompts": {label: prompt, ...}}
       -> {"results": [{"predicted_label": ..., "score": ...}, ...]}

Auth: if API_KEY is set in the environment, both POST endpoints require
`Authorization: Bearer <API_KEY>`.

Run directly for local testing:
    uvicorn serve:app --host 0.0.0.0 --port 8000
See submit_serve.sbatch for launching this on a Delta GPU node.
"""

import base64
import io
import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel

from models import resolve_adapter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("PATHOLOGY_MODEL", "conch")
DEVICE = os.environ.get("PATHOLOGY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.environ.get("PATHOLOGY_MAX_BATCH_SIZE", "32"))
MAX_WAIT_MS = float(os.environ.get("PATHOLOGY_MAX_WAIT_MS", "50"))
API_KEY = os.environ.get("API_KEY") or None


class BatchWorker:
    """Serializes all GPU forward passes through one background thread.

    Concurrent requests each enqueue their preprocessed image tensors and
    block on an Event; the worker thread drains the queue up to
    max_batch_size (or until max_wait_ms has passed with no new arrivals),
    runs a single encode_images() call for the combined batch, and wakes each
    waiting request with its slice of the results. This keeps GPU access
    single-threaded (required — encode_images is not safe to call from
    multiple threads at once) while still getting the throughput benefit of
    batching concurrent requests together.
    """

    def __init__(self, adapter, max_batch_size: int, max_wait_s: float):
        self.adapter = adapter
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s
        self._queue: "queue.Queue" = queue.Queue()
        self._gpu_lock = threading.Lock()  # also guards encode_texts calls
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.batches_run = 0
        self.requests_run = 0

    def submit_images(self, tensors: List[torch.Tensor]) -> np.ndarray:
        if not tensors:
            return np.empty((0,))
        result: Dict = {}
        done = threading.Event()
        self._queue.put((tensors, result, done))
        done.wait()
        if "error" in result:
            raise result["error"]
        return result["embeddings"]

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        with self._gpu_lock:
            return self.adapter.encode_texts(texts)

    def _run(self):
        while True:
            item = self._queue.get()
            batch_items = [item]
            total = len(item[0])
            deadline = time.monotonic() + self.max_wait_s
            while total < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                batch_items.append(nxt)
                total += len(nxt[0])

            all_tensors = [t for tensors, _, _ in batch_items for t in tensors]
            try:
                with self._gpu_lock:
                    stacked = torch.stack(all_tensors)
                    embeddings = self.adapter.encode_images(stacked)
            except Exception as e:  # noqa: BLE001 - surface to every waiter, then keep serving
                logger.exception("Batch inference failed")
                for _, result, done in batch_items:
                    result["error"] = e
                    done.set()
                continue

            offset = 0
            for tensors, result, done in batch_items:
                n = len(tensors)
                result["embeddings"] = embeddings[offset:offset + n]
                offset += n
                done.set()

            self.batches_run += 1
            self.requests_run += len(batch_items)
            logger.debug(
                f"Ran batch of {total} images from {len(batch_items)} requests "
                f"(batch #{self.batches_run})"
            )


def decode_image(b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}") from e


def classify(image_embs: np.ndarray, text_embs: np.ndarray, labels: List[str]):
    image_norm = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_norm = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
    sims = image_norm @ text_norm.T
    exp = np.exp(sims - sims.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    best = probs.argmax(axis=1)
    return [labels[i] for i in best], probs[np.arange(len(best)), best]


class EmbedRequest(BaseModel):
    images: List[str]


class ClassifyRequest(BaseModel):
    images: List[str]
    prompts: Dict[str, str]


app = FastAPI(title="pathology-embedding-server")
worker: Optional[BatchWorker] = None
adapter = None


@app.on_event("startup")
def _startup():
    global worker, adapter
    logger.info(f"Loading {MODEL_NAME} on {DEVICE}")
    adapter = resolve_adapter(MODEL_NAME, device=DEVICE)
    worker = BatchWorker(adapter, max_batch_size=MAX_BATCH_SIZE, max_wait_s=MAX_WAIT_MS / 1000)
    logger.info(
        f"Ready: model={MODEL_NAME} device={DEVICE} "
        f"max_batch_size={MAX_BATCH_SIZE} max_wait_ms={MAX_WAIT_MS}"
    )


def _check_auth(request: Request):
    if API_KEY is None:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "batches_run": worker.batches_run if worker else 0,
        "requests_run": worker.requests_run if worker else 0,
    }


@app.post("/embed")
def embed(req: EmbedRequest, request: Request):
    _check_auth(request)
    images = [decode_image(b64) for b64 in req.images]
    tensors = [adapter.preprocess(img) for img in images]
    embeddings = worker.submit_images(tensors)
    return {"embeddings": [e.tolist() for e in embeddings]}


@app.post("/classify")
def classify_endpoint(req: ClassifyRequest, request: Request):
    _check_auth(request)
    if adapter.encode_texts is None:
        raise HTTPException(status_code=400, detail=f"{MODEL_NAME} does not support text encoding")
    images = [decode_image(b64) for b64 in req.images]
    tensors = [adapter.preprocess(img) for img in images]
    embeddings = worker.submit_images(tensors)
    labels = list(req.prompts.keys())
    text_embs = worker.encode_texts(list(req.prompts.values()))
    pred_labels, scores = classify(embeddings, text_embs, labels)
    return {
        "results": [
            {"predicted_label": label, "score": float(score)}
            for label, score in zip(pred_labels, scores)
        ]
    }
