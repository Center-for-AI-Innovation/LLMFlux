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
        # proj_contrast+normalize puts image embeddings in the same
        # contrastively-trained space as encode_text's output, which is what
        # our cosine-similarity zero-shot classification assumes.
        emb = model.encode_image(batch.to(device), proj_contrast=True, normalize=True)
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
        import musk
        import timm
        from musk import utils as musk_utils
        from musk import modeling  # noqa: F401  (registers musk_large_patch16_384 with timm)
        from transformers import XLMRobertaTokenizer
        from torchvision import transforms
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    except ImportError as e:
        raise ImportError(
            f"MUSK is not installed. Run: pip install git+{MUSK_REPO}.git"
        ) from e

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    # fp16 on GPU matches the model card's example and halves memory; CPU
    # (e.g. local smoke testing) stays fp32 since fp16 ops are slow/unsupported there.
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    tokenizer_path = os.path.join(os.path.dirname(musk.__file__), "models", "tokenizer.spm")
    if not os.path.exists(tokenizer_path):
        raise RuntimeError(
            f"Could not find MUSK's tokenizer.spm at {tokenizer_path}. Check "
            f"{MUSK_REPO} for where it's currently packaged."
        )

    try:
        model = timm.create_model("musk_large_patch16_384")
        musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        model = model.to(device=device, dtype=dtype).eval()
        tokenizer = XLMRobertaTokenizer(tokenizer_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MUSK ({e}). Check {MUSK_REPO} for the current "
            "loading API and confirm your HF token has accepted access to "
            "xiangjx/musk."
        ) from e

    preprocess = transforms.Compose([
        transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    @torch.no_grad()
    def encode_images(batch: torch.Tensor) -> np.ndarray:
        emb, _ = model(
            image=batch.to(device=device, dtype=dtype),
            with_head=False, out_norm=False, ms_aug=True, return_global=True,
        )
        return emb.float().cpu().numpy()

    @torch.no_grad()
    def encode_texts(texts: List[str]) -> np.ndarray:
        ids, padding_mask = musk_utils.xlm_tokenizer(texts, tokenizer, max_len=100)
        _, emb = model(
            text_description=ids.to(device), padding_mask=padding_mask.to(device),
            with_head=False, out_norm=True, ms_aug=False, return_global=True,
        )
        return emb.float().cpu().numpy()

    return Adapter(name="musk", preprocess=preprocess, encode_images=encode_images, encode_texts=encode_texts)


def load_open_clip_hub(spec: str, device: str = "cuda", hf_token: Optional[str] = None) -> Adapter:
    """Generic adapter for any OpenCLIP-compatible model — no per-model code
    needed as long as it's published in one of OpenCLIP's own formats.

    Two spec forms:
      - "<hf-repo>" — a model published in OpenCLIP's own HF-hub format (an
        `open_clip_config.json` + weights file on the repo), loaded directly
        via `hf-hub:<repo>`. Covers e.g. wisdomik/QuiltNet-B-32.
      - "<arch>@<hf-repo>/<filename>" — a bare OpenCLIP checkpoint file on an
        otherwise plain HF repo (no OpenCLIP hub config), naming the
        architecture it was trained with and the file to download, e.g.
        "ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt". Only works if you
        already know the architecture; check the model's own README/card.
    """
    hf_token = hf_token or os.environ.get("HF_TOKEN")

    if "@" in spec:
        arch_name, _, repo_and_file = spec.partition("@")
        repo_id, _, filename = repo_and_file.rpartition("/")
        if not repo_id or not filename:
            raise ValueError(
                f"Invalid openclip spec {spec!r}; expected '<arch>@<hf-repo>/<filename>'"
            )
        model_name = arch_name
    else:
        repo_id = filename = None
        model_name = f"hf-hub:{spec}"

    try:
        import open_clip
    except ImportError as e:
        raise ImportError("OpenCLIP is not installed. Run: pip install open_clip_torch") from e

    if repo_id is not None:
        from huggingface_hub import hf_hub_download

        pretrained = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
    else:
        pretrained = None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load OpenCLIP model {spec!r} ({e}). Check the model's own "
            "Hugging Face page for its current OpenCLIP loading recipe."
        ) from e

    @torch.no_grad()
    def encode_images(batch: torch.Tensor) -> np.ndarray:
        emb = model.encode_image(batch.to(device))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    @torch.no_grad()
    def encode_texts(texts: List[str]) -> np.ndarray:
        tokens = tokenizer(texts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    return Adapter(
        name=f"openclip:{spec}", preprocess=preprocess, encode_images=encode_images, encode_texts=encode_texts
    )


def load_hf_clip(repo_id: str, device: str = "cuda", hf_token: Optional[str] = None) -> Adapter:
    """Generic adapter for any Hugging Face `transformers` CLIPModel repo,
    e.g. "openai/clip-vit-base-patch32" or another HF-native CLIP checkpoint.
    No per-model code needed as long as it loads with CLIPModel/CLIPProcessor.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        raise ImportError("transformers is not installed. Run: pip install transformers") from e

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    try:
        model = CLIPModel.from_pretrained(repo_id, token=hf_token).to(device).eval()
        processor = CLIPProcessor.from_pretrained(repo_id, token=hf_token)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load transformers CLIP model {repo_id!r} ({e}). Check "
            f"https://huggingface.co/{repo_id} for its current loading API."
        ) from e

    def preprocess(image):
        # CLIPProcessor batches internally; run it per-image here so the
        # result matches every other adapter's preprocess contract (one
        # tensor per image, stacked into a batch by the caller).
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def encode_images(batch: torch.Tensor) -> np.ndarray:
        emb = model.get_image_features(pixel_values=batch.to(device))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    @torch.no_grad()
    def encode_texts(texts: List[str]) -> np.ndarray:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()

    return Adapter(
        name=f"hfclip:{repo_id}", preprocess=preprocess, encode_images=encode_images, encode_texts=encode_texts
    )


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


def resolve_adapter(spec: str, device: str = "cuda", hf_token: Optional[str] = None) -> Adapter:
    """Resolve a --model / PATHOLOGY_MODEL spec string to a loaded Adapter.

    This is the extension point for a model nobody has written an adapter
    for: if it's CLIP-shaped and published in one of the two standard formats
    below, a team gets it with a spec string and no new code at all. Only a
    model with genuinely non-standard loading code (a CONCH/MUSK situation)
    needs a bespoke adapter added to ADAPTERS above.

      - "conch" / "musk" / "test" — the bespoke adapters above.
      - "openclip:<...>" — any OpenCLIP-compatible model (see load_open_clip_hub
        for the two spec forms this accepts).
      - "hfclip:<hf-repo>" — any `transformers` CLIPModel repo, e.g.
        "hfclip:openai/clip-vit-base-patch32" (see load_hf_clip).
    """
    if spec in ADAPTERS:
        return ADAPTERS[spec](device=device, hf_token=hf_token)
    if spec.startswith("openclip:"):
        return load_open_clip_hub(spec[len("openclip:"):], device=device, hf_token=hf_token)
    if spec.startswith("hfclip:"):
        return load_hf_clip(spec[len("hfclip:"):], device=device, hf_token=hf_token)
    raise ValueError(
        f"Unknown model spec {spec!r}. Expected one of {sorted(ADAPTERS)}, "
        "'openclip:<hf-repo>' or 'openclip:<arch>@<hf-repo>/<filename>', "
        "or 'hfclip:<hf-repo>'."
    )
