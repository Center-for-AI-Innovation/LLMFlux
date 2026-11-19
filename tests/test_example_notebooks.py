"""Tests for the example notebooks in examples/notebooks/.

The notebooks are run the way a reader runs them, minus the cells that talk to
Slurm (tagged "slurm"): the real cell code builds requests from the real
sample images, and the parsing cell (tagged "results") reads a results file in
the format BatchProcessor writes.
"""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
VISION_NOTEBOOK = NOTEBOOK_DIR / "vision_batch_analysis.ipynb"
SAMPLE_IMAGES = NOTEBOOK_DIR / "sample_images"


def _code_cells(path):
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            yield index, "".join(cell["source"]), set(cell.get("metadata", {}).get("tags", []))


def _chat_result(request, reply=None, error=None):
    """One result entry the way BatchProcessor saves it."""
    result = {"input": request, "metadata": {"model": "Qwen/Qwen2.5-VL-7B-Instruct"}}
    if error is not None:
        result["error"] = error
    else:
        result["output"] = {
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        }
    return result


class TestNotebookHygiene(unittest.TestCase):
    """Committed notebooks must not carry a run's outputs or someone's setup."""

    def notebooks(self):
        found = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
        self.assertTrue(found, f"no notebooks in {NOTEBOOK_DIR}")
        return found

    def test_outputs_are_cleared(self):
        for path in self.notebooks():
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    with self.subTest(notebook=path.name, cell=cell["source"][:1]):
                        self.assertEqual(cell["outputs"], [])
                        self.assertIsNone(cell["execution_count"])

    def test_no_accounts_home_paths_or_usernames(self):
        # Outputs and defaults leak these: an account default charges whoever
        # runs the notebook to someone else's allocation.
        forbidden = re.compile(r"\b[a-z]{4}-(?:delta|dtai)-(?:gpu|cpu|gh)\b|/u/[a-z]|akumdam")
        for path in self.notebooks():
            with self.subTest(notebook=path.name):
                self.assertIsNone(forbidden.search(path.read_text()))

    def test_code_cells_compile(self):
        for path in self.notebooks():
            for index, source, _ in _code_cells(path):
                with self.subTest(notebook=path.name, cell=index):
                    compile(source, f"{path.name} cell {index}", "exec")


class TestVisionNotebook(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name) / "run"
        env = {"LLMFLUX_IMAGE_DIR": str(SAMPLE_IMAGES), "LLMFLUX_WORKDIR": str(self.workdir)}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SLURM_ACCOUNT", None)

    def run_notebook(self, results_for=None, max_images=None):
        """Execute the non-Slurm cells; return the notebook's namespace.

        results_for(requests) builds the results list written to RESULTS_JSON
        just before the "results" cell, standing in for the batch job.
        """
        namespace = {"__name__": "__notebook__"}
        for index, source, tags in _code_cells(VISION_NOTEBOOK):
            if "slurm" in tags:
                continue
            if "results" in tags:
                if results_for is None:
                    break
                payload = {"results": results_for(namespace["requests"])}
                Path(namespace["RESULTS_JSON"]).write_text(json.dumps(payload))
            exec(compile(source, f"cell {index}", "exec"), namespace)
            if index == 2 and max_images is not None:  # the configuration cell
                namespace["MAX_IMAGES"] = max_images
        return namespace

    def test_one_request_per_sample_image(self):
        ns = self.run_notebook()
        lines = Path(ns["REQUESTS_JSONL"]).read_text().splitlines()
        requests = [json.loads(line) for line in lines]
        expected_ids = sorted(p.stem for p in SAMPLE_IMAGES.glob("*.jpg"))
        self.assertEqual(sorted(r["custom_id"] for r in requests), expected_ids)
        self.assertEqual(len(expected_ids), 4)

        for request in requests:
            with self.subTest(image=request["custom_id"]):
                body = request["body"]
                self.assertEqual(request["url"], "/v1/chat/completions")
                # LLMFlux rejects a body model that differs from the engine's
                # internal name; --model fills it in.
                self.assertNotIn("model", body)
                # The converter writes 0.7/500, which would beat the CLI flags.
                self.assertEqual(body["temperature"], ns["TEMPERATURE"])
                self.assertEqual(body["max_tokens"], ns["MAX_TOKENS"])
                system, user = body["messages"]
                self.assertEqual(system, {"role": "system", "content": ns["SYSTEM_PROMPT"]})
                text, image = user["content"]
                self.assertEqual(text, {"type": "text", "text": ns["USER_PROMPT"]})
                self.assertTrue(image["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_max_images_converts_only_a_subset_and_leaves_the_source_alone(self):
        before = sorted(p.name for p in SAMPLE_IMAGES.iterdir())
        ns = self.run_notebook(max_images=2)
        requests = Path(ns["REQUESTS_JSONL"]).read_text().splitlines()
        self.assertEqual(len(requests), 2)
        self.assertEqual(sorted(p.name for p in SAMPLE_IMAGES.iterdir()), before)

        # Re-running with a different limit replaces the subset, not adds to it.
        ns = self.run_notebook(max_images=3)
        self.assertEqual(len(Path(ns["REQUESTS_JSONL"]).read_text().splitlines()), 3)

    def test_replies_are_parsed_with_a_status_for_each_image(self):
        reply = {"summary": "A stop sign.", "objects": ["stop sign", "pole"], "scene": "street",
                 "visible_text": ["STOP"], "safety_flags": {"violence": False, "adult": False, "medical": False}}

        def fake_results(requests):
            by_id = {r["custom_id"]: r for r in requests}
            return [
                _chat_result(by_id["bar_chart"], reply="```json\n" + json.dumps(reply) + "\n```"),
                _chat_result(by_id["door_sign"], reply=json.dumps(reply)),
                _chat_result(by_id["landscape"], reply="I can't describe this image as JSON."),
                _chat_result(by_id["stop_sign"], error="400 Client Error: Bad Request"),
            ]

        ns = self.run_notebook(results_for=fake_results)

        saved = json.loads(Path(ns["PARSED_JSON"]).read_text())
        self.assertEqual([(r["image"], r["status"]) for r in saved], [
            ("bar_chart", "ok"),
            ("door_sign", "ok"),
            ("landscape", "reply not JSON"),
            ("stop_sign", "request failed"),
        ])
        self.assertEqual(saved[0]["parsed"], reply)  # the fence was tolerated
        self.assertIsNone(saved[2]["parsed"])
        self.assertEqual(saved[3]["error"], "400 Client Error: Bad Request")

        table = ns["table"]
        self.assertEqual(list(table["status"]), ["ok", "ok", "reply not JSON", "request failed"])
        self.assertEqual(table.loc[0, "visible_text"], "STOP")

    def test_parse_json_reply(self):
        parse = self.run_notebook(results_for=lambda requests: [])["parse_json_reply"]
        self.assertEqual(parse('{"a": 1}'), {"a": 1})
        self.assertEqual(parse('Here you go:\n```json\n{"a": {"b": 2}}\n```'), {"a": {"b": 2}})
        self.assertIsNone(parse("no json here"))
        self.assertIsNone(parse('{"a": 1'))
        self.assertIsNone(parse("[1, 2]"))
        self.assertIsNone(parse(None))


if __name__ == "__main__":
    unittest.main()
