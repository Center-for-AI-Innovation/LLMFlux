"""Tests for vision-to-JSONL conversion utilities."""

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

from llmflux.converters.vision import encode_image, get_image_mime_type, vision_to_jsonl


def _write_tiny_png(path: str) -> None:
    """Write a 1x1 white PNG to path (valid minimal PNG bytes)."""
    PNG_1X1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with open(path, "wb") as f:
        f.write(PNG_1X1)


class TestEncodeImage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_encodes_file_as_base64(self):
        img = os.path.join(self.tmp.name, "img.png")
        _write_tiny_png(img)
        encoded = encode_image(img)
        decoded = base64.b64decode(encoded)
        self.assertTrue(decoded[:4] == b"\x89PNG")

    def test_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            encode_image(os.path.join(self.tmp.name, "nonexistent.png"))


class TestGetImageMimeType(unittest.TestCase):
    def test_png(self):
        self.assertEqual(get_image_mime_type("photo.png"), "image/png")

    def test_jpg(self):
        self.assertEqual(get_image_mime_type("photo.jpg"), "image/jpeg")

    def test_jpeg(self):
        self.assertEqual(get_image_mime_type("photo.jpeg"), "image/jpeg")

    def test_gif(self):
        self.assertEqual(get_image_mime_type("anim.gif"), "image/gif")

    def test_webp(self):
        self.assertEqual(get_image_mime_type("img.webp"), "image/webp")

    def test_unknown_extension(self):
        self.assertEqual(get_image_mime_type("file.tiff"), "application/octet-stream")

    def test_uppercase_extension(self):
        self.assertEqual(get_image_mime_type("PHOTO.PNG"), "image/png")


class TestVisionToJsonl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_image(self, name="test.png") -> Path:
        img = self.test_dir / name
        _write_tiny_png(str(img))
        return img

    def test_single_image_creates_jsonl(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        result_path = vision_to_jsonl(str(img), output_path=out)
        self.assertEqual(result_path, out)
        self.assertTrue(os.path.exists(out))
        with open(out) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["custom_id"], "test")
        self.assertEqual(entry["url"], "/v1/chat/completions")
        user_msg = next(m for m in entry["body"]["messages"] if m["role"] == "user")
        content_types = [c["type"] for c in user_msg["content"]]
        self.assertIn("text", content_types)
        self.assertIn("image_url", content_types)

    def test_default_prompt_applied(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out)
        with open(out) as f:
            entry = json.loads(f.readline())
        user_msg = next(m for m in entry["body"]["messages"] if m["role"] == "user")
        text_part = next(c for c in user_msg["content"] if c["type"] == "text")
        self.assertIn("Describe", text_part["text"])

    def test_custom_prompt_template(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out, prompt_template="What color is this?")
        with open(out) as f:
            entry = json.loads(f.readline())
        user_msg = next(m for m in entry["body"]["messages"] if m["role"] == "user")
        text_part = next(c for c in user_msg["content"] if c["type"] == "text")
        self.assertEqual(text_part["text"], "What color is this?")

    def test_system_prompt_included(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out, system_prompt="You are a vision AI.")
        with open(out) as f:
            entry = json.loads(f.readline())
        messages = entry["body"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a vision AI.")

    def test_prompts_map_by_filename(self):
        img = self._make_image("cat.png")
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out, prompts_map={"cat.png": "Is this a cat?"})
        with open(out) as f:
            entry = json.loads(f.readline())
        user_msg = next(m for m in entry["body"]["messages"] if m["role"] == "user")
        text_part = next(c for c in user_msg["content"] if c["type"] == "text")
        self.assertEqual(text_part["text"], "Is this a cat?")

    def test_metadata_included(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out)
        with open(out) as f:
            entry = json.loads(f.readline())
        self.assertIn("metadata", entry)
        self.assertEqual(entry["metadata"]["filename"], "test.png")

    def test_directory_with_multiple_images(self):
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        for name in ("a.png", "b.png", "c.png"):
            _write_tiny_png(str(img_dir / name))
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img_dir), output_path=out, file_pattern="*.png")
        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 3)

    def test_directory_default_pattern_finds_all_supported_extensions(self):
        """The default must actually match images — it used to be a brace glob,
        which Python's glob does not expand, so directories yielded nothing."""
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        for name in ("a.png", "b.jpg", "c.jpeg", "d.gif", "e.webp", "f.bmp"):
            _write_tiny_png(str(img_dir / name))
        (img_dir / "notes.txt").write_text("not an image")
        out = str(self.test_dir / "out.jsonl")

        vision_to_jsonl(str(img_dir), output_path=out)

        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 6)

    def test_directory_default_pattern_is_case_insensitive(self):
        """Phones and cameras produce IMG_1234.JPG."""
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        for name in ("IMG_1.JPG", "IMG_2.PNG"):
            _write_tiny_png(str(img_dir / name))
        out = str(self.test_dir / "out.jsonl")

        vision_to_jsonl(str(img_dir), output_path=out)

        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_directory_default_pattern_is_not_recursive(self):
        img_dir = self.test_dir / "images"
        (img_dir / "nested").mkdir(parents=True)
        _write_tiny_png(str(img_dir / "top.png"))
        _write_tiny_png(str(img_dir / "nested" / "deep.png"))
        out = str(self.test_dir / "out.jsonl")

        vision_to_jsonl(str(img_dir), output_path=out)

        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 1)

    def test_recursive_pattern_descends_into_subdirectories(self):
        img_dir = self.test_dir / "images"
        (img_dir / "nested").mkdir(parents=True)
        _write_tiny_png(str(img_dir / "top.png"))
        _write_tiny_png(str(img_dir / "nested" / "deep.png"))
        out = str(self.test_dir / "out.jsonl")

        vision_to_jsonl(str(img_dir), output_path=out, file_pattern="**/*.png")

        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_legacy_brace_pattern_is_expanded(self):
        """The old broken default should still work if a caller passes it."""
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        for name in ("a.png", "b.jpg"):
            _write_tiny_png(str(img_dir / name))
        out = str(self.test_dir / "out.jsonl")

        vision_to_jsonl(
            str(img_dir), output_path=out, file_pattern="*.{jpg,jpeg,png,gif,webp,bmp}"
        )

        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_empty_directory_warns(self):
        """An empty result is silent otherwise: a valid but empty JSONL file."""
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        (img_dir / "notes.txt").write_text("not an image")
        out = str(self.test_dir / "out.jsonl")

        with self.assertLogs("llmflux.converters.vision", level="WARNING") as logs:
            vision_to_jsonl(str(img_dir), output_path=out)

        self.assertIn("No images found", "\n".join(logs.output))

    def test_oversized_image_skipped(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out, max_image_size=1)
        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 0)

    def test_report_names_oversized_images(self):
        """A skipped image must be visible to the caller, not only in the log."""
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        for name in ("small.png", "big.png"):
            _write_tiny_png(str(img_dir / name))
        (img_dir / "big.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 5000)
        out = str(self.test_dir / "out.jsonl")

        path, report = vision_to_jsonl(
            str(img_dir), output_path=out, max_image_size=1000, return_report=True
        )

        self.assertEqual(path, out)
        self.assertEqual(report["found"], 2)
        self.assertEqual(report["converted"], 1)
        self.assertEqual(len(report["skipped_too_large"]), 1)
        skipped = report["skipped_too_large"][0]
        self.assertTrue(skipped["path"].endswith("big.png"))
        self.assertEqual(skipped["limit"], 1000)
        self.assertGreater(skipped["size"], 1000)

    def test_report_counts_a_clean_run(self):
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        for name in ("a.png", "b.png"):
            _write_tiny_png(str(img_dir / name))
        out = str(self.test_dir / "out.jsonl")

        _, report = vision_to_jsonl(str(img_dir), output_path=out, return_report=True)

        self.assertEqual(report, {
            "found": 2, "converted": 2, "skipped_too_large": [], "failed": [],
        })

    def test_returns_bare_path_by_default(self):
        """Existing callers must keep getting a string back."""
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        result = vision_to_jsonl(str(img), output_path=out)
        self.assertIsInstance(result, str)

    def test_dropped_images_are_summarized_in_a_warning(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        with self.assertLogs("llmflux.converters.vision", level="WARNING") as logs:
            vision_to_jsonl(str(img), output_path=out, max_image_size=1)
        self.assertIn("Converted 0 of 1 images", "\n".join(logs.output))

    def test_default_max_image_size_accepts_typical_phone_photo(self):
        """A 12MB photo is a normal 48MP phone JPEG and must not be dropped."""
        img_dir = self.test_dir / "images"
        img_dir.mkdir()
        big = img_dir / "phone.jpg"
        big.write_bytes(b"\xff\xd8\xff" + b"0" * (12 * 1024 * 1024))
        out = str(self.test_dir / "out.jsonl")

        _, report = vision_to_jsonl(str(img_dir), output_path=out, return_report=True)

        self.assertEqual(report["converted"], 1)
        self.assertEqual(report["skipped_too_large"], [])

    def test_raises_for_missing_input(self):
        with self.assertRaises(FileNotFoundError):
            vision_to_jsonl(str(self.test_dir / "nope.png"), output_path="/tmp/x.jsonl")

    def test_model_included_in_body(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out, model="gpt-4o")
        with open(out) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["body"]["model"], "gpt-4o")

    def test_auto_output_path_created(self):
        img = self._make_image()
        result_path = vision_to_jsonl(str(img))
        self.assertTrue(os.path.exists(result_path))
        os.unlink(result_path)


if __name__ == "__main__":
    unittest.main()