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

    def test_oversized_image_skipped(self):
        img = self._make_image()
        out = str(self.test_dir / "out.jsonl")
        vision_to_jsonl(str(img), output_path=out, max_image_size=1)
        with open(out) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 0)

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