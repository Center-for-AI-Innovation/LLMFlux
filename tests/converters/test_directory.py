"""Tests for the directory to JSONL converter."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from llmflux.converters.directory import directory_to_jsonl

class TestDirectoryToJSONL(unittest.TestCase):
    """Test suite for the directory_to_jsonl function."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a directory structure with some test files
        self.input_dir = self.test_dir / "input_directory"
        os.makedirs(self.input_dir, exist_ok=True)
        
        # Create subdirectories
        self.sub_dir1 = self.input_dir / "subdir1"
        self.sub_dir2 = self.input_dir / "subdir2"
        os.makedirs(self.sub_dir1, exist_ok=True)
        os.makedirs(self.sub_dir2, exist_ok=True)
        
        # Create some text files
        with open(self.input_dir / "file1.txt", "w") as f:
            f.write("This is a test file.")
        
        with open(self.sub_dir1 / "file2.txt", "w") as f:
            f.write("This is another test file in a subdirectory.")
        
        with open(self.sub_dir2 / "file3.txt", "w") as f:
            f.write("Yet another test file in a different subdirectory.")
        
        # Create a JSON file
        with open(self.input_dir / "data.json", "w") as f:
            json.dump({"key": "value", "nested": {"data": "example"}}, f)
        
        # Create an image file (dummy binary data)
        with open(self.input_dir / "image.jpg", "wb") as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00')
        
        # Output path
        self.output_path = self.test_dir / "output.jsonl"
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_directory_to_jsonl_basic(self):
        """Test basic directory to JSONL conversion."""
        # Convert directory to JSONL
        result = directory_to_jsonl(self.input_dir, self.output_path)
        
        # Check that output file exists
        self.assertTrue(os.path.exists(self.output_path))
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 5)  # All 5 files
        self.assertEqual(result["successful_conversions"], 5)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 5 entries
        self.assertEqual(len(entries), 5)
        
        # Check that each entry has the expected structure
        for entry in entries:
            self.assertIn("custom_id", entry)
            self.assertIn("method", entry)
            self.assertEqual(entry["method"], "POST")
            self.assertIn("url", entry)
            self.assertEqual(entry["url"], "/v1/chat/completions")
            self.assertIn("body", entry)
            self.assertIn("content", entry["body"].get("messages", [{}])[0])
            self.assertIn("metadata", entry)
            self.assertIn("source_file", entry["metadata"])
    
    def test_directory_to_jsonl_with_extensions(self):
        """Test directory to JSONL conversion with specific extensions."""
        # Convert only text files
        result = directory_to_jsonl(
            self.input_dir, 
            self.output_path, 
            extensions=[".txt"]
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 3)  # Only the 3 text files
        self.assertEqual(result["successful_conversions"], 3)
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 3 entries (one for each txt file)
        self.assertEqual(len(entries), 3)
        
        # Verify that all entries are from txt files
        source_files = [entry["metadata"]["source_file"] for entry in entries]
        for source in source_files:
            self.assertTrue(source.endswith(".txt"))
    
    def test_directory_to_jsonl_with_template(self):
        """Test directory to JSONL conversion with a template."""
        # Define a custom template
        template = "Process this file: {content}"
        
        # Convert directory to JSONL with template
        result = directory_to_jsonl(
            self.input_dir, 
            self.output_path,
            prompt_template=template
        )
        
        # Check result
        self.assertTrue(result["success"])
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check prompt format
        for entry in entries:
            messages = entry["body"].get("messages", [])
            if messages:
                content = messages[0].get("content", "")
                if "This is a test file" in content:  # Check only text files
                    self.assertTrue(content.startswith("Process this file:"))
    
    def test_directory_to_jsonl_recursive_false(self):
        """Test directory to JSONL conversion without recursion."""
        # Convert only files in the top directory (no recursion)
        result = directory_to_jsonl(
            self.input_dir, 
            self.output_path,
            recursive=False
        )
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 3)  # Only top-level files
        
        # Read output file
        entries = []
        with open(self.output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 3 entries (only top-level files)
        self.assertEqual(len(entries), 3)
        
        # Verify that no entries are from subdirectories
        source_files = [entry["metadata"]["source_file"] for entry in entries]
        for source in source_files:
            self.assertNotIn("subdir", source)
    
    def test_directory_to_jsonl_with_api_params(self):
        """Test directory to JSONL conversion with API parameters."""
        # Define custom API parameters
        api_params = {
            "temperature": 0.8,
            "max_tokens": 200,
            "top_p": 0.95
        }
        
        # Convert directory to JSONL with API parameters
        result = directory_to_jsonl(
            self.input_dir, 
            self.output_path,
            api_parameters=api_params
        )
        
        # Check result
        self.assertTrue(result["success"])
        
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
    
    def test_directory_to_jsonl_error_handling(self):
        """Test error handling in directory to JSONL conversion."""
        # Test with non-existent directory
        with self.assertRaises(FileNotFoundError):
            directory_to_jsonl(
                self.test_dir / "nonexistent_dir", 
                self.output_path
            )
        
        # Test with invalid extension filter
        result = directory_to_jsonl(
            self.input_dir, 
            self.output_path,
            extensions=[".nonexistent"]
        )
        
        # Should succeed but with 0 successful conversions
        self.assertTrue(result["success"])
        self.assertEqual(result["successful_conversions"], 0)

if __name__ == "__main__":
    unittest.main() 
