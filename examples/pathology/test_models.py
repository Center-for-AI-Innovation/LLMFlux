"""Tests for models.py's resolve_adapter() dispatch and spec parsing.

This covers only the spec-parsing/dispatch logic and the exact library calls
each generic adapter makes (asserted via mocks) — not real model loading,
which needs a GPU/Delta allocation and gated model access; see loadtest.py for
that kind of end-to-end validation.

Runs without torch/open_clip/transformers/huggingface_hub installed: `torch`
is stubbed in sys.modules before `models` is imported (models.py imports it
unconditionally at module level), and open_clip/transformers/huggingface_hub
are stubbed per-test via mock.patch.dict, since models.py imports those
lazily inside the functions that need them.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _NoGrad:
    """Stands in for torch.no_grad used as a decorator: @torch.no_grad()."""

    def __call__(self, func):
        return func

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if "torch" not in sys.modules:
    _torch_stub = mock.MagicMock(name="torch_stub")
    _torch_stub.no_grad = _NoGrad
    _torch_stub.Tensor = object
    sys.modules["torch"] = _torch_stub

import models  # noqa: E402


def _fake_open_clip():
    module = mock.MagicMock(name="open_clip_stub")
    module.create_model_and_transforms.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
    module.get_tokenizer.return_value = mock.MagicMock()
    return module


def _fake_transformers():
    module = mock.MagicMock(name="transformers_stub")
    module.CLIPModel.from_pretrained.return_value = mock.MagicMock()
    module.CLIPProcessor.from_pretrained.return_value = mock.MagicMock()
    return module


def _fake_huggingface_hub(download_path="/tmp/fake-checkpoint.pt"):
    module = mock.MagicMock(name="huggingface_hub_stub")
    module.hf_hub_download.return_value = download_path
    return module


class TestResolveAdapterDispatch(unittest.TestCase):
    def test_known_name_dispatches_to_ADAPTERS(self):
        fake_loader = mock.MagicMock(return_value="the-adapter")
        with mock.patch.dict(models.ADAPTERS, {"test": fake_loader}):
            result = models.resolve_adapter("test", device="cpu", hf_token="tok")
        fake_loader.assert_called_once_with(device="cpu", hf_token="tok")
        self.assertEqual(result, "the-adapter")

    def test_openclip_prefix_strips_prefix_before_dispatch(self):
        with mock.patch.object(models, "load_open_clip_hub", return_value="oc-adapter") as loader:
            result = models.resolve_adapter("openclip:wisdomik/QuiltNet-B-32", device="cpu", hf_token=None)
        loader.assert_called_once_with("wisdomik/QuiltNet-B-32", device="cpu", hf_token=None)
        self.assertEqual(result, "oc-adapter")

    def test_hfclip_prefix_strips_prefix_before_dispatch(self):
        with mock.patch.object(models, "load_hf_clip", return_value="hf-adapter") as loader:
            result = models.resolve_adapter("hfclip:openai/clip-vit-base-patch32", device="cpu", hf_token=None)
        loader.assert_called_once_with("openai/clip-vit-base-patch32", device="cpu", hf_token=None)
        self.assertEqual(result, "hf-adapter")

    def test_unknown_spec_raises_with_supported_formats(self):
        with self.assertRaises(ValueError) as ctx:
            models.resolve_adapter("not-a-real-spec", device="cpu", hf_token=None)
        message = str(ctx.exception)
        self.assertIn("conch", message)
        self.assertIn("openclip:", message)
        self.assertIn("hfclip:", message)


class TestLoadOpenClipHub(unittest.TestCase):
    def test_hf_hub_form_loads_directly_from_repo(self):
        fake_open_clip = _fake_open_clip()
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip}):
            adapter = models.load_open_clip_hub("wisdomik/QuiltNet-B-32", device="cpu", hf_token="tok")

        fake_open_clip.create_model_and_transforms.assert_called_once_with(
            "hf-hub:wisdomik/QuiltNet-B-32", pretrained=None
        )
        fake_open_clip.get_tokenizer.assert_called_once_with("hf-hub:wisdomik/QuiltNet-B-32")
        self.assertEqual(adapter.name, "openclip:wisdomik/QuiltNet-B-32")
        self.assertIsNotNone(adapter.encode_texts)

    def test_arch_at_repo_form_downloads_checkpoint_then_loads_by_arch(self):
        fake_open_clip = _fake_open_clip()
        fake_hub = _fake_huggingface_hub(download_path="/cache/pathgenclip.pt")
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip, "huggingface_hub": fake_hub}):
            adapter = models.load_open_clip_hub(
                "ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt", device="cpu", hf_token="tok"
            )

        fake_hub.hf_hub_download.assert_called_once_with(
            repo_id="jamessyx/PathGen-CLIP", filename="pathgenclip.pt", token="tok"
        )
        fake_open_clip.create_model_and_transforms.assert_called_once_with(
            "ViT-B-16", pretrained="/cache/pathgenclip.pt"
        )
        self.assertEqual(adapter.name, "openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt")

    def test_arch_at_spec_missing_filename_raises_value_error(self):
        with self.assertRaises(ValueError):
            models.load_open_clip_hub("ViT-B-16@jamessyx/PathGen-CLIP/", device="cpu", hf_token=None)

    def test_arch_at_spec_missing_repo_raises_value_error(self):
        with self.assertRaises(ValueError):
            models.load_open_clip_hub("ViT-B-16@justafilename.pt", device="cpu", hf_token=None)

    def test_missing_open_clip_raises_helpful_import_error(self):
        with mock.patch.dict(sys.modules, {"open_clip": None}):
            with self.assertRaises(ImportError) as ctx:
                models.load_open_clip_hub("wisdomik/QuiltNet-B-32", device="cpu", hf_token=None)
        self.assertIn("open_clip_torch", str(ctx.exception))

    def test_load_failure_is_wrapped_with_model_context(self):
        fake_open_clip = _fake_open_clip()
        fake_open_clip.create_model_and_transforms.side_effect = OSError("no such repo")
        with mock.patch.dict(sys.modules, {"open_clip": fake_open_clip}):
            with self.assertRaises(RuntimeError) as ctx:
                models.load_open_clip_hub("bogus/repo", device="cpu", hf_token=None)
        self.assertIn("bogus/repo", str(ctx.exception))
        self.assertIn("no such repo", str(ctx.exception))


class TestLoadHfClip(unittest.TestCase):
    def test_loads_via_clip_model_and_processor(self):
        fake_transformers = _fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            adapter = models.load_hf_clip("openai/clip-vit-base-patch32", device="cpu", hf_token="tok")

        fake_transformers.CLIPModel.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32", token="tok"
        )
        fake_transformers.CLIPProcessor.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32", token="tok"
        )
        self.assertEqual(adapter.name, "hfclip:openai/clip-vit-base-patch32")
        self.assertIsNotNone(adapter.encode_texts)

    def test_falls_back_to_hf_token_env_var(self):
        fake_transformers = _fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with mock.patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
                models.load_hf_clip("openai/clip-vit-base-patch32", device="cpu", hf_token=None)

        fake_transformers.CLIPModel.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32", token="env-token"
        )

    def test_missing_transformers_raises_helpful_import_error(self):
        with mock.patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                models.load_hf_clip("openai/clip-vit-base-patch32", device="cpu", hf_token=None)
        self.assertIn("transformers", str(ctx.exception))

    def test_load_failure_is_wrapped_with_model_context(self):
        fake_transformers = _fake_transformers()
        fake_transformers.CLIPModel.from_pretrained.side_effect = OSError("gated repo")
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with self.assertRaises(RuntimeError) as ctx:
                models.load_hf_clip("some/gated-repo", device="cpu", hf_token=None)
        self.assertIn("some/gated-repo", str(ctx.exception))
        self.assertIn("gated repo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
