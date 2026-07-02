"""Tests for the JSON to JSONL converter."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from llmflux.converters.json import json_to_jsonl

class TestJSONToJSONL(unittest.TestCase):
    """Test suite for the json_to_jsonl function."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create test JSON files
        
        # Simple JSON with an array of objects
        self.simple_array_json = self.test_dir / "simple_array.json"
        simple_data = [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
        ]
        with open(self.simple_array_json, "w") as f:
            json.dump(simple_data, f)
        
        # JSON with nested objects
        self.nested_json = self.test_dir / "nested.json"
        nested_data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
        with open(self.nested_json, "w") as f:
            json.dump(nested_data, f)
        
        # JSON with a single object
        self.single_object_json = self.test_dir / "single_object.json"
        single_data = {"id": 1, "name": "Alice", "email": "alice@example.com"}
        with open(self.single_object_json, "w") as f:
            json.dump(single_data, f)
        
        # Output path
        self.output_path = self.test_dir / "output.jsonl"
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_json_to_jsonl_array(self):
        """Test converting a JSON array to JSONL."""
        # Convert JSON to JSONL
        result = json_to_jsonl(self.simple_array_json, self.output_path)
        
        # Check that output file exists
        self.assertTrue(os.path.exists(self.output_path))
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["total_items"], 3)
        self.assertEqual(result["successful_conversions"], 3)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 3 entries
        self.assertEqual(len(entries), 3)
        
        # Check that each entry has the expected structure
        for entry in entries:
            self.assertIn("custom_id", entry)
            self.assertIn("method", entry)
            self.assertEqual(entry["method"], "POST")
            self.assertIn("url", entry)
            self.assertEqual(entry["url"], "/v1/chat/completions")
            self.assertIn("body", entry)
            self.assertIn("messages", entry["body"])
            self.assertIn("metadata", entry)
            self.assertIn("source_file", entry["metadata"])
            self.assertEqual(entry["metadata"]["source_file"], str(self.simple_array_json))
    
    def test_json_to_jsonl_nested_with_key(self):
        """Test converting a nested JSON to JSONL with a specific key."""
        # Convert JSON to JSONL, targeting the 'users' key
        result = json_to_jsonl(
            self.nested_json, 
            self.output_path,
            json_key="users"
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["total_items"], 2)  # 2 users
        self.assertEqual(result["successful_conversions"], 2)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 2 entries
        self.assertEqual(len(entries), 2)
        
        # Verify that user data is in the entries
        user_names = set()
        for entry in entries:
            content = entry["body"]["messages"][0]["content"]
            user_data = json.loads(content)
            user_names.add(user_data.get("name"))
        
        # Check that we have entries for both users
        self.assertEqual(user_names, {"Alice", "Bob"})
    
    def test_json_to_jsonl_single_object(self):
        """Test converting a single JSON object to JSONL."""
        # Convert JSON to JSONL
        result = json_to_jsonl(self.single_object_json, self.output_path)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["total_items"], 1)
        self.assertEqual(result["successful_conversions"], 1)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there is 1 entry
        self.assertEqual(len(entries), 1)
        
        # Verify that the single object data is in the entry
        content = entries[0]["body"]["messages"][0]["content"]
        data = json.loads(content)
        self.assertEqual(data.get("name"), "Alice")
    
    def test_json_to_jsonl_with_template(self):
        """Test JSON to JSONL conversion with a template."""
        # Define a custom template
        template = "Process this user: {content}"
        
        # Convert JSON to JSONL with template
        result = json_to_jsonl(
            self.simple_array_json, 
            self.output_path,
            prompt_template=template
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["successful_conversions"], 3)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check prompt format
        for entry in entries:
            content = entry["body"]["messages"][0]["content"]
            self.assertTrue(content.startswith("Process this user:"))
    
    def test_json_to_jsonl_with_api_params(self):
        """Test JSON to JSONL conversion with API parameters."""
        # Define custom API parameters
        api_params = {
            "temperature": 0.8,
            "max_tokens": 200,
            "top_p": 0.95
        }
        
        # Convert JSON to JSONL with API parameters
        result = json_to_jsonl(
            self.simple_array_json, 
            self.output_path,
            api_parameters=api_params
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["successful_conversions"], 3)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check API parameters in each entry
        for entry in entries:
            body = entry["body"]
            self.assertEqual(body.get("temperature"), 0.8)
            self.assertEqual(body.get("max_tokens"), 200)
            self.assertEqual(body.get("top_p"), 0.95)
    
    def test_json_to_jsonl_error_handling(self):
        """Test error handling in JSON to JSONL conversion."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            json_to_jsonl(
                self.test_dir / "nonexistent.json", 
                self.output_path
            )
        
        # Test with invalid JSON key
        result = json_to_jsonl(
            self.nested_json, 
            self.output_path,
            json_key="nonexistent_key"
        )
        
        # Should fail with key error
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Key 'nonexistent_key' not found in JSON")

    def test_auto_output_path(self):
        """No output_path → temp file is created and populated."""
        result = json_to_jsonl(self.simple_array_json)
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["output_path"])
        self.assertTrue(os.path.exists(result["output_path"]))
        with open(result["output_path"]) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 3)

    def test_invalid_json_file(self):
        """Malformed JSON file returns success=False with an error message."""
        bad_json = self.test_dir / "bad.json"
        bad_json.write_text("{not valid json}")
        result = json_to_jsonl(bad_json, self.output_path)
        self.assertFalse(result["success"])
        self.assertIn("JSON decode error", result["error"])

    def test_unsupported_root_type(self):
        """A JSON file whose root is a string/number returns success=False."""
        string_root = self.test_dir / "string_root.json"
        string_root.write_text('"just a string"')
        result = json_to_jsonl(string_root, self.output_path)
        self.assertFalse(result["success"])
        self.assertIn("Unsupported JSON format", result["error"])

    def test_failed_item_increments_counter(self):
        """An item that raises during processing increments failed_conversions."""
        from unittest.mock import patch
        array_json = self.test_dir / "arr.json"
        array_json.write_text('[{"id": 1}, {"id": 2}, {"id": 3}]')
        with patch("llmflux.converters.json._process_json_item", side_effect=[None, RuntimeError("boom"), None]):
            result = json_to_jsonl(array_json, self.output_path)
        self.assertEqual(result["failed_conversions"], 1)
        self.assertEqual(result["successful_conversions"], 2)

    def test_batch_format_generates_custom_id(self):
        """Item already in batch format without custom_id gets one generated."""
        data = [{"method": "POST", "url": "/v1/chat/completions", "body": {"messages": []}}]
        input_path = self.test_dir / "batch_no_id.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path)
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        self.assertIn("custom_id", entry)
        self.assertTrue(entry["custom_id"])

    def test_batch_format_uses_id_field(self):
        """Item in batch format uses id_field as custom_id when present."""
        data = [{"method": "POST", "url": "/v1/chat/completions", "body": {}, "my_id": "req-42"}]
        input_path = self.test_dir / "batch_id_field.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path, id_field="my_id")
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["custom_id"], "req-42")

    def test_batch_format_preserves_existing_model(self):
        """A model already in body is not removed or altered."""
        data = [{"method": "POST", "url": "/v1/chat/completions", "body": {"model": "original"}}]
        input_path = self.test_dir / "batch_has_model.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path)
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["body"]["model"], "original")

    def test_batch_format_merges_api_parameters(self):
        """api_parameters are merged into body when keys not already set."""
        data = [{"method": "POST", "url": "/v1/chat/completions", "body": {"messages": []}}]
        input_path = self.test_dir / "batch_api_params.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path, api_parameters={"top_p": 0.9})
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["body"]["top_p"], 0.9)

    def test_messages_format_with_id_field(self):
        """Item with messages key uses id_field for custom_id."""
        data = [{"messages": [{"role": "user", "content": "hi"}], "req_id": "msg-7"}]
        input_path = self.test_dir / "messages_id.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path, id_field="req_id")
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["custom_id"], "msg-7")

    def test_messages_format_with_prompt_template(self):
        """prompt_template wraps user message content."""
        data = [{"messages": [{"role": "user", "content": "cats"}]}]
        input_path = self.test_dir / "messages_tmpl.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path, prompt_template="Tell me about: {content}")
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        user_msg = next(m for m in entry["body"]["messages"] if m["role"] == "user")
        self.assertEqual(user_msg["content"], "Tell me about: cats")

    def test_prompt_format_with_system_key(self):
        """Item with prompt + system keys produces a system message first."""
        data = [{"prompt": "Hello", "system": "You are helpful."}]
        input_path = self.test_dir / "prompt_system.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path)
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        messages = entry["body"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello")

    def test_prompt_format_with_template(self):
        """prompt_template is applied to the prompt value."""
        data = [{"prompt": "dogs"}]
        input_path = self.test_dir / "prompt_tmpl.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path, prompt_template="Describe: {content}")
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        user_msg = next(m for m in entry["body"]["messages"] if m["role"] == "user")
        self.assertEqual(user_msg["content"], "Describe: dogs")

    def test_prompt_format_with_id_field(self):
        """Item with prompt key uses id_field for custom_id."""
        data = [{"prompt": "hi", "task_id": "t-99"}]
        input_path = self.test_dir / "prompt_id.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path, id_field="task_id")
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["custom_id"], "t-99")

    def test_generic_dict_no_recognized_keys(self):
        """Dict with no prompt/messages/batch keys is serialised as user message content."""
        data = [{"foo": "bar", "baz": 42}]
        input_path = self.test_dir / "generic.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path)
        self.assertTrue(result["success"])
        with open(self.output_path) as f:
            entry = json.loads(f.readline())
        content = entry["body"]["messages"][0]["content"]
        parsed = json.loads(content)
        self.assertEqual(parsed["foo"], "bar")
        self.assertIn("original_item", entry["metadata"])

    def test_non_dict_item_string(self):
        """A JSON array of plain strings produces one entry per string."""
        data = ["first prompt", "second prompt"]
        input_path = self.test_dir / "strings.json"
        input_path.write_text(json.dumps(data))
        result = json_to_jsonl(input_path, self.output_path)
        self.assertTrue(result["success"])
        self.assertEqual(result["successful_conversions"], 2)
        with open(self.output_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(lines[0]["body"]["messages"][0]["content"], "first prompt")
        self.assertEqual(lines[1]["body"]["messages"][0]["content"], "second prompt")


if __name__ == "__main__":
    unittest.main()
