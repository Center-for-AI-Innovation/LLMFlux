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

if __name__ == "__main__":
    unittest.main() 
