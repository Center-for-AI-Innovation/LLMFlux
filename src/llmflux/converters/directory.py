"""Directory to JSONL conversion utilities for LLMFlux."""

import os
import json
import logging
import tempfile
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set

from .utils import create_jsonl_entry, generate_custom_id

logger = logging.getLogger(__name__)

def read_file_content(file_path: str, max_file_size: int = 1024 * 1024, encoding: str = 'utf-8') -> str:
    """Read file content with size checking.
    
    Args:
        file_path: Path to the file
        max_file_size: Maximum file size in bytes
        encoding: File encoding
        
    Returns:
        File content as string
        
    Raises:
        ValueError: If file is too large
        UnicodeDecodeError: If file cannot be decoded with the specified encoding
    """
    file_size = os.path.getsize(file_path)
    if file_size > max_file_size:
        raise ValueError(f"File {file_path} exceeds size limit: {file_size} > {max_file_size}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode {file_path} as {encoding}, trying binary mode")
        # Fallback to binary reading with replacement of non-UTF-8 characters
        with open(file_path, 'rb') as f:
            content = f.read()
            return content.decode(encoding, errors='replace')

def directory_to_jsonl(
    input_path: str,
    output_path: str,
    file_pattern: str = "*.*",
    prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_file_size: int = 1024 * 1024,
    encoding: str = 'utf-8',
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    api_parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert a directory of files to JSONL format.
    
    Args:
        input_path: Path to directory
        output_path: Path to save JSONL output
        file_pattern: File pattern to match
        prompt_template: Template string for prompts (e.g., "Process this file: {content}")
        system_prompt: Optional system prompt
        model: Model to use for inference
        exclude_patterns: List of patterns to exclude
        max_file_size: Maximum file size in bytes
        encoding: File encoding
        extensions: List of file extensions to include
        recursive: Whether to process subdirectories
        api_parameters: Additional API parameters
        
    Returns:
        Dictionary with results information:
        {
            "success": bool,
            "total_files": int,
            "successful_conversions": int,
            "failed_conversions": int,
            "output_path": str,
            "error": Optional[str]
        }
    
    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Directory not found: {input_path}")
    
    input_path = Path(input_path)
    exclude_patterns = exclude_patterns or []
    api_parameters = api_parameters or {}
    
    # Keep track of stats
    stats = {
        "success": True,
        "total_files": 0,
        "successful_conversions": 0,
        "failed_conversions": 0,
        "output_path": str(output_path),
        "error": None
    }
    
    # Build file extension filter if needed
    extension_filter = set()
    if extensions:
        extension_filter = {ext if ext.startswith('.') else f'.{ext}' for ext in extensions}
    
    # Find files
    pattern = file_pattern
    if recursive:
        pattern = f"**/{file_pattern}"
    
    files = list(input_path.glob(pattern))
    filtered_files = []
    
    # Apply filters
    for file_path in files:
        if not file_path.is_file():
            continue
            
        # Skip if not in extension filter
        if extension_filter and file_path.suffix not in extension_filter:
            continue
            
        # Skip if matches exclude patterns
        if any(file_path.match(pattern) for pattern in exclude_patterns):
            continue
            
        filtered_files.append(file_path)
    
    stats["total_files"] = len(filtered_files)
    logger.info(f"Found {len(filtered_files)} files to convert")
    
    try:
        # Process files
        with open(output_path, 'w') as f:
            for file_path in filtered_files:
                try:
                    # Read file content
                    content = read_file_content(
                        file_path,
                        max_file_size=max_file_size,
                        encoding=encoding
                    )
                    
                    # Create messages array
                    messages = []
                    
                    # Add system prompt if provided
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    # Format prompt with template if provided
                    if prompt_template:
                        user_content = prompt_template.format(content=content)
                    else:
                        user_content = content
                    
                    messages.append({
                        "role": "user",
                        "content": user_content
                    })
                    
                    # Create JSONL entry
                    entry = create_jsonl_entry(
                        messages=messages,
                        model=model,
                        custom_id=generate_custom_id(),
                        **api_parameters
                    )
                    
                    # Add metadata
                    entry["metadata"] = {
                        "source_file": str(file_path),
                        "file_size": os.path.getsize(file_path),
                        "file_extension": file_path.suffix,
                    }
                    
                    # Write to JSONL file
                    f.write(json.dumps(entry) + '\n')
                    
                    stats["successful_conversions"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    stats["failed_conversions"] += 1
        
        logger.info(f"Created JSONL file at {output_path} with {stats['successful_conversions']} entries")
        return stats
        
    except Exception as e:
        logger.error(f"Error in directory_to_jsonl: {e}")
        stats["success"] = False
        stats["error"] = str(e)
        if os.path.exists(output_path):
            os.unlink(output_path)
        return stats 
