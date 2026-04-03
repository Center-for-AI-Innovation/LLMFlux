
"""Tests for the CSV to JSONL converter."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from llmflux.converters.csv import csv_to_jsonl

class TestCSVToJSONL(unittest.TestCase):
    """Test suite for the csv_to_jsonl function."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_csv_to_jsonl_template_mode(self):
        """Test CSV to JSONL conversion using template mode."""
        # Create a test CSV file
        csv_path = self.test_dir / "test.csv"
        with open(csv_path, "w") as f:
            f.write("name,age,occupation\n")
            f.write("John,30,Engineer\n")
            f.write("Jane,25,Designer\n")
        
        # Define a template
        template = "Generate a bio for {name}, who is {age} years old and works as a {occupation}."
        
        # Convert CSV to JSONL
        output_path = self.test_dir / "output.jsonl"
        result = csv_to_jsonl(
            input_path=csv_path,
            output_path=output_path,
            prompt_template=template,
            system_prompt="You are a professional biography writer."
        )
        
        # Check that the operation was successful
        self.assertTrue(result["success"])
        self.assertEqual(result["total_rows"], 2)
        self.assertEqual(result["successful_conversions"], 2)
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Read output file
        entries = []
        with open(output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 2 entries
        self.assertEqual(len(entries), 2)
        
        # Check that each entry has the expected structure
        for entry in entries:
            self.assertIn("custom_id", entry)
            self.assertIn("method", entry)
            self.assertEqual(entry["method"], "POST")
            self.assertIn("url", entry)
            self.assertEqual(entry["url"], "/v1/chat/completions")
            self.assertIn("body", entry)
            
            # Check messages
            messages = entry["body"]["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["content"], "You are a professional biography writer.")
            self.assertEqual(messages[1]["role"], "user")
            self.assertTrue(messages[1]["content"].startswith("Generate a bio for"))
            
            # Check metadata
            self.assertIn("metadata", entry)
            self.assertIn("csv_row", entry["metadata"])
            self.assertIn("source_file", entry["metadata"])
    
    def test_csv_to_jsonl_column_mode(self):
        """Test CSV to JSONL conversion using column mode."""
        # Create a test CSV file with prompts and contexts
        csv_path = self.test_dir / "prompts.csv"
        with open(csv_path, "w") as f:
            f.write("id,prompt,context\n")
            f.write("1,What is machine learning?,I am a beginner in AI.\n")
            f.write("2,Explain neural networks.,I have some programming experience.\n")
        
        # Convert CSV to JSONL using column mode
        output_path = self.test_dir / "output.jsonl"
        result = csv_to_jsonl(
            input_path=csv_path,
            output_path=output_path,
            prompt_column="prompt",
            id_column="id"
        )
        
        # Check that the operation was successful
        self.assertTrue(result["success"])
        self.assertEqual(result["total_rows"], 2)
        self.assertEqual(result["successful_conversions"], 2)
        
        # Check that output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Read output file
        entries = []
        with open(output_path, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Check that there are 2 entries
        self.assertEqual(len(entries), 2)
        
        # Check that ids are preserved
        ids = [entry["custom_id"] for entry in entries]
        self.assertIn("1", ids)
        self.assertIn("2", ids)
        
        # Find the entry with id=1
        entry_1 = next(entry for entry in entries if entry["custom_id"] == "1")
        
        # Check prompt content
        messages = entry_1["body"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "What is machine learning?")
        
        # Check metadata
        self.assertIn("metadata", entry_1)
        self.assertIn("csv_row", entry_1["metadata"])
        self.assertEqual(entry_1["metadata"]["csv_row"]["context"], "I am a beginner in AI.")
    
    def test_csv_to_jsonl_error_handling(self):
        """Test error handling in CSV to JSONL conversion."""
        output_path = self.test_dir / "output.jsonl"
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            csv_to_jsonl(
                input_path=self.test_dir / "nonexistent.csv",
                output_path=output_path,
                prompt_template="Test template"
            )
        
        # Create a file so the FileNotFoundError check passes and ValueError is reached
        existing_csv = self.test_dir / "test.csv"
        with open(existing_csv, "w") as f:
            f.write("name,age\nJohn,30\n")

        # Test without either template or column
        with self.assertRaises(ValueError):
            csv_to_jsonl(
                input_path=existing_csv,
                output_path=output_path
            )

if __name__ == "__main__":
    unittest.main() 
