"""Model adapters for CONCH and MUSK.

Both are small research packages installed straight from GitHub, not stable
PyPI releases (see README.md). The loading/inference calls below reflect each
project's own documented usage as of when this was written — if loading
raises, the error message points at the project's README so you can check
what changed rather than guessing.
"""

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

CONCH_REPO = "https://github.com/mahmoodlab/CONCH"
MUSK_REPO = "https://github.com/lilab-stanford/MUSK"


@dataclass
class Adapter:
    """Uniform interface run_embeddings.py drives regardless of model."""

    name: str
    preprocess: Callable  # PIL.Image -> torch.Tensor
    encode_images: Callable  # torch.Tensor [B,C,H,W] -> np.ndarray [B,D]
    encode_texts: Optional[Callable] = None  # List[str] -> np.ndarray [N,D]


def load_conch(device: str = "cuda", hf_token: Optional[str] = None) -> Adapter:
    """Load CONCH. Requires accepted access to huggingface.co/MahmoodLab/CONCH
    and `pip install git+https://github.com/mahmoodlab/CONCH.git`.
    """
    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    except ImportError as e:
        raise ImportError(
            f"CONCH is not installed. Run: pip install git+{CONCH_REPO}.git"
        ) from e

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    try:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH", hf_auth_token=hf_token
        )
        model = model.to(device).eval()
        tokenizer = get_tokenizer()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CONCH ({e}). Check {CONCH_REPO} for the current "
            "loading API and confirm your HF token has accepted access to "
            "MahmoodLab/CONCH."
        ) from e

    @torch.no_grad()
    def encode_images(batch: torch.Tensor) -> np.ndarray:
        emb = model.encode_image(batch.to(device))
        return emb.float().cpu().numpy()

    @torch.no_grad()
    def encode_texts(texts: List[str]) -> np.ndarray:
        tokens = tokenize(texts=texts, tokenizer=tokenizer).to(device)
        emb = model.encode_text(tokens)
        return emb.float().cpu().numpy()

    return Adapter(name="conch", preprocess=preprocess, encode_images=encode_images, encode_texts=encode_texts)


def load_musk(device: str = "cuda", hf_token: Optional[str] = None) -> Adapter:
    """Load MUSK. Requires accepted access to huggingface.co/xiangjx/musk and
    `pip install git+https://github.com/lilab-stanford/MUSK.git`.
    """
    try:
        import timm
        from musk import utils as musk_utils
        from musk import modeling  # noqa: F401  (registers musk_large_patch16_384 with timm)
        from transformers import XLMRobertaTokenizer
        from torchvision import transforms
    except ImportError as e:
        raise ImportError(
            f"MUSK is not installed. Run: pip install git+{MUSK_REPO}.git"
        ) from e

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    try:
        model = timm.create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        model = model.to(device).eval()
        tokenizer = musk_utils.xlm_tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MUSK ({e}). Check {MUSK_REPO} for the current "
            "loading API and confirm your HF token has accepted access to "
            "xiangjx/musk."
        ) from e

    preprocess = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    @torch.no_grad()
    def encode_images(batch: torch.Tensor) -> np.ndarray:
        emb, _ = model(image=batch.to(device), with_head=False, out_norm=True)
        return emb.float().cpu().numpy()

    @torch.no_grad()
    def encode_texts(texts: List[str]) -> np.ndarray:
        ids, padding_mask = tokenizer(texts, device=device)
        _, emb = model(text_description=ids, padding_mask=padding_mask, with_head=False, out_norm=True)
        return emb.float().cpu().numpy()

    return Adapter(name="musk", preprocess=preprocess, encode_images=encode_images, encode_texts=encode_texts)


def load_test(device: str = "cpu", hf_token: Optional[str] = None) -> Adapter:
    """No-dependency stand-in for CONCH/MUSK.

    Produces meaningless but deterministic "embeddings" (per-channel mean of
    an 8x8 downsample) with a configurable artificial delay standing in for
    GPU forward-pass cost. Use this to validate run_embeddings.py or
    serve.py's batching/concurrency behavior — request routing, checkpointing,
    auth — on real infrastructure (a real compute node, real network) before
    gated model access or a GPU allocation is sorted out. Never use it for
    actual results.
    """
    import time

    delay_s = float(os.environ.get("PATHOLOGY_TEST_ADAPTER_DELAY_S", "0.05"))

    def preprocess(image):
        from PIL import Image  # local import: keep this adapter's own footprint minimal

        resized = image.resize((8, 8), Image.BILINEAR)
        arr = np.asarray(resized).astype("float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def encode_images(batch: torch.Tensor) -> np.ndarray:
        time.sleep(delay_s * (1 + 0.1 * batch.shape[0]))  # fixed + per-image cost, like a real batch
        return batch.mean(dim=(2, 3)).numpy()

    def encode_texts(texts: List[str]) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        return rng.random((len(texts), 3)).astype("float32")

    return Adapter(name="test", preprocess=preprocess, encode_images=encode_images, encode_texts=encode_texts)


ADAPTERS = {"conch": load_conch, "musk": load_musk, "test": load_test}
