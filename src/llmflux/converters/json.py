"""JSON to JSONL conversion utilities for LLMFlux."""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .utils import create_jsonl_entry, generate_custom_id

logger = logging.getLogger(__name__)

def json_to_jsonl(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model: Optional[str] = None,
    id_field: Optional[str] = None,
    json_key: Optional[str] = None,
    prompt_template: Optional[str] = None,
    api_parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convert JSON to JSONL format.
    
    This function converts JSON in various formats to JSONL format compatible with the batch processor.
    It handles both arrays of JSON objects and single JSON objects.
    
    Args:
        input_path: Path to JSON file
        output_path: Path to save JSONL output
        model: Model to use for inference
        id_field: Field to use as custom_id (uses generated UUID if None)
        json_key: Key to extract from the JSON object (for nested objects)
        prompt_template: Template string for prompts (e.g., "Process this data: {content}")
        api_parameters: Additional API parameters
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results information:
        {
            "success": bool,
            "total_items": int,
            "successful_conversions": int,
            "failed_conversions": int,
            "output_path": str,
            "error": Optional[str]
        }
    
    Raises:
        FileNotFoundError: If input JSON file doesn't exist
        ValueError: If JSON format is not recognized
    """
    # Initialize results dictionary
    results = {
        "success": True,
        "total_items": 0,
        "successful_conversions": 0,
        "failed_conversions": 0,
        "output_path": None,
        "error": None
    }
    
    # Convert Path objects to strings
    if isinstance(input_path, Path):
        input_path = str(input_path)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    
    # Create output path if not provided
    if not output_path:
        output_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.jsonl', dir=tempfile.gettempdir()
        )
        output_path = output_file.name
        output_file.close()
    elif isinstance(output_path, Path):
        output_path = str(output_path)
    
    results["output_path"] = output_path
    api_parameters = api_parameters or {}
    
    try:
        # Read JSON file
        with open(input_path, 'r') as f:
            json_data = json.load(f)
        
        # Extract data using json_key if specified
        if json_key:
            if isinstance(json_data, dict) and json_key in json_data:
                json_data = json_data[json_key]
            else:
                results["success"] = False
                results["error"] = f"Key '{json_key}' not found in JSON"
                return results
        
        # Create JSONL file
        with open(output_path, 'w') as f:
            # Handle array of objects
            if isinstance(json_data, list):
                results["total_items"] = len(json_data)
                for item in json_data:
                    try:
                        _process_json_item(item, f, model, id_field, prompt_template, api_parameters, input_path)
                        results["successful_conversions"] += 1
                    except Exception as e:
                        logger.error(f"Error processing JSON item: {e}")
                        results["failed_conversions"] += 1
            # Handle single object
            elif isinstance(json_data, dict):
                results["total_items"] = 1
                try:
                    _process_json_item(json_data, f, model, id_field, prompt_template, api_parameters, input_path)
                    results["successful_conversions"] += 1
                except Exception as e:
                    logger.error(f"Error processing JSON item: {e}")
                    results["failed_conversions"] += 1
            else:
                results["success"] = False
                results["error"] = f"Unsupported JSON format: {type(json_data)}"
                if os.path.exists(output_path):
                    os.unlink(output_path)
                return results
        
        logger.info(f"Created JSONL file at {output_path} from JSON with {results['successful_conversions']} entries")
        return results
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        results["success"] = False
        results["error"] = f"JSON decode error: {str(e)}"
        if os.path.exists(output_path) and not output_path.startswith('/tmp'):
            os.unlink(output_path)
        return results
    except Exception as e:
        logger.error(f"Error converting JSON to JSONL: {e}")
        results["success"] = False
        results["error"] = str(e)
        if os.path.exists(output_path) and not output_path.startswith('/tmp'):
            os.unlink(output_path)
        return results

def _process_json_item(
    item: Dict[str, Any], 
    file_handle: Any, 
    model: Optional[str] = None, 
    id_field: Optional[str] = None,
    prompt_template: Optional[str] = None,
    api_parameters: Optional[Dict[str, Any]] = None,
    input_path: Optional[Union[str, Path]] = None
):
    """Process a single JSON item and write to JSONL file.
    
    Args:
        item: JSON item to process
        file_handle: File handle to write to
        model: Model to use for inference
        id_field: Field to use as custom_id
        prompt_template: Template for formatting prompt
        api_parameters: Additional API parameters
        input_path: Path to the source file
    """
    api_parameters = api_parameters or {}
    
    # Convert Path objects to strings
    if isinstance(input_path, Path):
        input_path = str(input_path)
    
    # Convert any Path objects in the item to strings
    if isinstance(item, dict):
        for key, value in list(item.items()):
            if isinstance(value, Path):
                item[key] = str(value)
                
    # Check if item already has the OpenAI batch format
    if all(k in item for k in ('method', 'url', 'body')):
        # Item is already in OpenAI batch format
        # Just ensure custom_id exists
        if 'custom_id' not in item:
            if id_field and id_field in item:
                item['custom_id'] = str(item[id_field])
            else:
                item['custom_id'] = generate_custom_id()
        
        # Add API parameters if provided
        if api_parameters and 'body' in item:
            for key, value in api_parameters.items():
                if key not in item['body']:
                    item['body'][key] = value
                
        # Write item as is
        file_handle.write(json.dumps(item) + '\n')
        return
    
    # Check if item has a messages field (direct prompt format)
    if 'messages' in item:
        messages = item['messages']
        custom_id = None
        
        if id_field and id_field in item:
            custom_id = str(item[id_field])
        elif 'id' in item:
            custom_id = str(item['id'])
        else:
            custom_id = generate_custom_id()
        
        # Apply prompt template if specified
        if prompt_template and len(messages) > 0:
            for i, msg in enumerate(messages):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    messages[i]['content'] = prompt_template.format(content=content)
            
        # Create JSONL entry
        entry = create_jsonl_entry(
            messages=messages,
            model=model or item.get('model'),
            custom_id=custom_id,
            temperature=item.get('temperature', 0.7),
            max_tokens=item.get('max_tokens', 500),
            **{k: v for k, v in item.items() if k not in ('messages', 'model', 'id')},
            **api_parameters
        )
        
        # Add metadata
        if input_path:
            entry["metadata"] = {"source_file": input_path}
        
        # Write to JSONL file
        file_handle.write(json.dumps(entry) + '\n')
        return
    
    # Legacy format: Assume content is in 'prompt' field or the item is the prompt itself
    if isinstance(item, dict):
        if 'prompt' in item:
            prompt = item['prompt']
            custom_id = None
            
            if id_field and id_field in item:
                custom_id = str(item[id_field])
            elif 'id' in item:
                custom_id = str(item['id'])
            else:
                custom_id = generate_custom_id()
                
            # Create messages array
            messages = []
            
            # Add system message if provided
            if 'system' in item:
                messages.append({
                    "role": "system",
                    "content": item['system']
                })
                
            # Add user message, applying template if specified
            user_content = prompt
            if prompt_template:
                user_content = prompt_template.format(content=prompt)
                
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Create JSONL entry
            entry = create_jsonl_entry(
                messages=messages,
                model=model or item.get('model'),
                custom_id=custom_id,
                temperature=item.get('temperature', 0.7),
                max_tokens=item.get('max_tokens', 500),
                **{k: v for k, v in item.items() if k not in ('prompt', 'system', 'model', 'id')},
                **api_parameters
            )
            
            # Add metadata
            if input_path:
                entry["metadata"] = {"source_file": input_path}
            
            # Write to JSONL file
            file_handle.write(json.dumps(entry) + '\n')
        else:
            # No recognizable format, create a generic entry with all fields
            custom_id = None
            
            if id_field and id_field in item:
                custom_id = str(item[id_field])
            elif 'id' in item:
                custom_id = str(item['id'])
            else:
                custom_id = generate_custom_id()
            
            # Create content, applying template if specified    
            content = json.dumps(item)
            if prompt_template:
                content = prompt_template.format(content=content)
                
            # Create messages array with item content
            messages = [{
                "role": "user",
                "content": content
            }]
            
            # Create JSONL entry
            entry = create_jsonl_entry(
                messages=messages,
                model=model,
                custom_id=custom_id,
                **api_parameters
            )
            
            # Add original item as metadata
            metadata = {"original_item": item}
            if input_path:
                metadata["source_file"] = input_path
            entry["metadata"] = metadata
            
            # Write to JSONL file
            file_handle.write(json.dumps(entry) + '\n')
    else:
        # Simple string or other value
        custom_id = generate_custom_id()
        
        # Create content, applying template if specified
        content = str(item)
        if prompt_template:
            content = prompt_template.format(content=content)
            
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # Create JSONL entry
        entry = create_jsonl_entry(
            messages=messages,
            model=model,
            custom_id=custom_id,
            **api_parameters
        )
        
        # Add metadata
        if input_path:
            entry["metadata"] = {"source_file": input_path}
        
        # Write to JSONL file
        file_handle.write(json.dumps(entry) + '\n') 
