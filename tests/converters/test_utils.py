"""Tests for JSONL utility functions."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from llmflux.converters.utils import (
    validate_jsonl,
    read_jsonl,
    merge_jsonl_files,
    jsonl_to_json,
    create_jsonl_entry,
    write_jsonl_entry
)

class TestJSONLUtils(unittest.TestCase):
    """Test suite for JSONL utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_validate_jsonl_valid(self):
        """Test validation of valid JSONL file."""
        # Create valid JSONL file
        valid_path = self.test_dir / "valid.jsonl"
        with open(valid_path, "w") as f:
            f.write('{"key": "value1"}\n')
            f.write('{"key": "value2"}\n')
            f.write('{"key": "value3"}\n')
        
        self.assertTrue(validate_jsonl(valid_path))
    
    def test_validate_jsonl_invalid(self):
        """Test validation of invalid JSONL file."""
        # Create invalid JSONL file
        invalid_path = self.test_dir / "invalid.jsonl"
        with open(invalid_path, "w") as f:
            f.write('{"key": "value1"}\n')
            f.write('{"key": "value2\n')  # Missing closing quote and brace
            f.write('{"key": "value3"}\n')
        
        self.assertFalse(validate_jsonl(invalid_path))
    
    def test_validate_jsonl_nonexistent(self):
        """Test validation of nonexistent JSONL file."""
        nonexistent_path = self.test_dir / "nonexistent.jsonl"
        self.assertFalse(validate_jsonl(nonexistent_path))
    
    def test_read_jsonl(self):
        """Test reading JSONL file."""
        # Create JSONL file
        test_path = self.test_dir / "test.jsonl"
        expected_data = [
            {"key": "value1", "num": 1},
            {"key": "value2", "num": 2},
            {"key": "value3", "num": 3}
        ]
        
        with open(test_path, "w") as f:
            for item in expected_data:
                f.write(json.dumps(item) + "\n")
        
        # Read JSONL
        result = list(read_jsonl(test_path))
        self.assertEqual(result, expected_data)
    
    def test_merge_jsonl_files(self):
        """Test merging JSONL files."""
        # Create JSONL files
        file1_path = self.test_dir / "file1.jsonl"
        file2_path = self.test_dir / "file2.jsonl"
        output_path = self.test_dir / "merged.jsonl"
        
        with open(file1_path, "w") as f:
            f.write('{"key": "value1"}\n')
            f.write('{"key": "value2"}\n')
        
        with open(file2_path, "w") as f:
            f.write('{"key": "value3"}\n')
            f.write('{"key": "value4"}\n')
        
        # Merge files
        result_path = merge_jsonl_files([file1_path, file2_path], output_path)
        
        # Verify result
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, "r") as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 4)
        self.assertEqual(json.loads(lines[0]), {"key": "value1"})
        self.assertEqual(json.loads(lines[1]), {"key": "value2"})
        self.assertEqual(json.loads(lines[2]), {"key": "value3"})
        self.assertEqual(json.loads(lines[3]), {"key": "value4"})
    
    def test_jsonl_to_json(self):
        """Test converting JSONL to JSON."""
        # Create JSONL file
        jsonl_path = self.test_dir / "test.jsonl"
        json_path = self.test_dir / "test.json"
        
        expected_data = [
            {"key": "value1"},
            {"key": "value2"},
            {"key": "value3"}
        ]
        
        with open(jsonl_path, "w") as f:
            for item in expected_data:
                f.write(json.dumps(item) + "\n")
        
        # Convert JSONL to JSON
        result_path = jsonl_to_json(jsonl_path, json_path)
        
        # Verify result
        self.assertEqual(result_path, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        with open(json_path, "r") as f:
            result_data = json.load(f)
        
        self.assertEqual(result_data, expected_data)
    
    def test_create_jsonl_entry(self):
        """Test creating JSONL entry."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"}
        ]
        
        # Create entry
        entry = create_jsonl_entry(
            messages=messages,
            custom_id="test-id",
            temperature=0.5,
            max_tokens=100
        )

        # Verify entry
        self.assertEqual(entry["custom_id"], "test-id")
        self.assertEqual(entry["method"], "POST")
        self.assertEqual(entry["url"], "/v1/chat/completions")
        self.assertIsNone(entry["body"]["model"])
        self.assertEqual(entry["body"]["messages"], messages)
        self.assertEqual(entry["body"]["temperature"], 0.5)
        self.assertEqual(entry["body"]["max_tokens"], 100)
    
    def test_write_jsonl_entry(self):
        """Test writing JSONL entry."""
        entry = {
            "custom_id": "test-id",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "Hello"}]
            }
        }
        
        # Write entry
        output_path = self.test_dir / "output.jsonl"
        write_jsonl_entry(entry, output_path)
        
        # Verify result
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, "r") as f:
            content = f.read().strip()
        
        self.assertEqual(json.loads(content), entry)

if __name__ == "__main__":
    unittest.main() 
