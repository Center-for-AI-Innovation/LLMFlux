"""JSONL utility functions for LLMFlux."""

import json
import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Generator

logger = logging.getLogger(__name__)

def validate_jsonl(jsonl_path: str) -> bool:
    """Validate that a file is properly formatted JSONL.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        True if file is valid JSONL, False otherwise
    """
    if not os.path.exists(jsonl_path):
        logger.error(f"File not found: {jsonl_path}")
        return False
    
    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {i}: {line[:100]}... Error: {e}")
                    return False
        return True
    except Exception as e:
        logger.error(f"Error validating JSONL file: {e}")
        return False

def read_jsonl(jsonl_path: str) -> Generator[Dict[str, Any], None, None]:
    """Read and parse JSONL file line by line.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Yields:
        Parsed JSON objects from each line
    """
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                item = json.loads(line)
                yield item
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON line: {line[:100]}... Error: {e}")

def merge_jsonl_files(input_files: List[str], output_path: str) -> str:
    """Merge multiple JSONL files into one.
    
    Args:
        input_files: List of JSONL files to merge
        output_path: Path to save merged file
        
    Returns:
        Path to merged JSONL file
    """
    try:
        with open(output_path, 'w') as out_f:
            for input_file in input_files:
                if not os.path.exists(input_file):
                    logger.warning(f"File not found, skipping: {input_file}")
                    continue
                    
                with open(input_file, 'r') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        # Validate JSON before writing
                        try:
                            json.loads(line)
                            out_f.write(line + '\n')
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {input_file}, skipping line: {line[:100]}...")
        
        logger.info(f"Merged {len(input_files)} JSONL files to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error merging JSONL files: {e}")
        raise

def jsonl_to_json(jsonl_path: str, json_path: str) -> str:
    """Convert JSONL to JSON array.
    
    Args:
        jsonl_path: Path to JSONL file
        json_path: Path to save JSON file
        
    Returns:
        Path to JSON file
    """
    try:
        items = list(read_jsonl(jsonl_path))
        
        with open(json_path, 'w') as f:
            json.dump(items, f, indent=2)
        
        logger.info(f"Converted JSONL to JSON: {jsonl_path} -> {json_path}")
        return json_path
    except Exception as e:
        logger.error(f"Error converting JSONL to JSON: {e}")
        raise

def generate_custom_id() -> str:
    """Generate a custom ID for JSONL entries.
    
    Returns:
        A unique ID string
    """
    return str(uuid.uuid4())

def create_jsonl_entry(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    custom_id: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    **kwargs
) -> Dict[str, Any]:
    """Create a standardized JSONL entry following OpenAI batch API format.
    
    Args:
        messages: List of message objects with role and content
        model: Model name to use for inference
        custom_id: Custom identifier (generated if not provided)
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for the request body
        
    Returns:
        Formatted JSONL entry as dictionary
    """
    return {
        "custom_id": custom_id or generate_custom_id(),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
    }

def write_jsonl_entry(entry: Dict[str, Any], file_path: str, mode: str = 'a'):
    """Write a single JSONL entry to a file.
    
    Args:
        entry: JSONL entry dictionary
        file_path: Path to JSONL file
        mode: File open mode ('a' for append, 'w' for write)
    """
    with open(file_path, mode) as f:
        f.write(json.dumps(entry) + '\n') 
