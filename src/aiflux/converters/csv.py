"""CSV to JSONL conversion utilities for AI-Flux."""

import os
import csv
import json
import logging
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from string import Formatter

from .utils import create_jsonl_entry, generate_custom_id

logger = logging.getLogger(__name__)

def _format_with_fallback(template: str, **kwargs) -> str:
    """Format a string with named placeholders, ignoring missing keys.
    
    Args:
        template: String template with named placeholders
        **kwargs: Values for template placeholders
        
    Returns:
        Formatted string
    """
    keys = [fname for _, fname, _, _ in Formatter().parse(template) if fname]
    missing_keys = [k for k in keys if k not in kwargs]
    
    # Add empty strings for missing keys
    for key in missing_keys:
        kwargs[key] = ""
        
    return template.format(**kwargs)

def csv_to_jsonl(
    input_path: str, 
    output_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    prompt_column: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    id_column: Optional[str] = None,
    api_parameters: Optional[Dict[str, Any]] = None,
    **pandas_kwargs
) -> Dict[str, Any]:
    """Convert CSV to JSONL format.
    
    This function supports two modes:
    1. Template mode: Applies a prompt_template to each row
    2. Column mode: Uses a specified column as the prompt
    
    Args:
        input_path: Path to CSV file
        output_path: Path to save JSONL output (or None for temp file)
        prompt_template: Template string with CSV column placeholders (e.g., "Summarize: {text}")
        prompt_column: Column containing prompts (for multi-prompt mode)
        system_prompt: Optional system prompt
        model: Model to use for inference
        id_column: Column to use as custom_id (uses generated UUID if None)
        api_parameters: Additional API parameters
        **pandas_kwargs: Additional pandas.read_csv parameters
        
    Returns:
        Dictionary with results information:
        {
            "success": bool,
            "total_rows": int,
            "successful_conversions": int,
            "failed_conversions": int,
            "output_path": str,
            "error": Optional[str]
        }
    
    Raises:
        ValueError: If neither prompt_template nor prompt_column is provided
        FileNotFoundError: If input CSV file doesn't exist
    """
    # Initialize results dictionary
    results = {
        "success": True,
        "total_rows": 0,
        "successful_conversions": 0,
        "failed_conversions": 0,
        "output_path": None,
        "error": None
    }

    if not prompt_template and not prompt_column:
        raise ValueError("Either prompt_template or prompt_column must be provided")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    
    # Create output path if not provided
    if not output_path:
        output_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.jsonl', dir=tempfile.gettempdir()
        )
        output_path = output_file.name
        output_file.close()
    
    # Convert Path objects to strings
    if isinstance(input_path, Path):
        input_path = str(input_path)
    if isinstance(output_path, Path):
        output_path = str(output_path)
        
    results["output_path"] = output_path
    api_parameters = api_parameters or {}
    
    try:
        # Read CSV file
        df = pd.read_csv(input_path, **pandas_kwargs)
        results["total_rows"] = len(df)
        logger.info(f"Read CSV with {len(df)} rows from {input_path}")
        
        # Create JSONL file
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                try:
                    row_dict = row.to_dict()
                    
                    # Convert any Path objects in row_dict to strings
                    for key, value in row_dict.items():
                        if isinstance(value, Path):
                            row_dict[key] = str(value)
                    
                    # Get custom ID if specified
                    custom_id = None
                    if id_column and id_column in row_dict:
                        custom_id = str(row_dict[id_column])
                    else:
                        custom_id = generate_custom_id()
                    
                    # Create messages list
                    messages = []
                    
                    # Add system prompt if provided
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    # Add user prompt based on mode
                    if prompt_template:
                        # Template mode: Format template with row values
                        user_content = _format_with_fallback(prompt_template, **row_dict)
                    elif prompt_column:
                        # Column mode: Use specified column as prompt
                        if prompt_column not in row_dict:
                            logger.warning(f"Prompt column '{prompt_column}' not found in row, skipping")
                            results["failed_conversions"] += 1
                            continue
                        user_content = str(row_dict[prompt_column])
                    else:
                        # This should never happen due to earlier check
                        continue
                    
                    messages.append({
                        "role": "user",
                        "content": user_content
                    })
                    
                    # Create JSONL entry
                    entry = create_jsonl_entry(
                        messages=messages,
                        model=model,
                        custom_id=custom_id,
                        **api_parameters
                    )
                    
                    # Add row data as metadata
                    entry["metadata"] = {
                        "csv_row": row_dict,
                        "source_file": input_path
                    }
                    
                    # Write to JSONL file
                    f.write(json.dumps(entry) + '\n')
                    results["successful_conversions"] += 1
                except Exception as e:
                    logger.error(f"Error processing CSV row: {e}")
                    results["failed_conversions"] += 1
                
        logger.info(f"Created JSONL file at {output_path} with {results['successful_conversions']} entries")
        return results
        
    except Exception as e:
        logger.error(f"Error converting CSV to JSONL: {e}")
        results["success"] = False
        results["error"] = str(e)
        if os.path.exists(output_path) and not output_path.startswith('/tmp'):
            os.unlink(output_path)
        return results 
