"""Vision to JSONL conversion utilities for LLMFlux."""

import os
import json
import logging
import tempfile
import base64
import glob
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .utils import create_jsonl_entry, generate_custom_id

logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    """Base64 encode an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded image string
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    return encoded_string

def get_image_mime_type(image_path: str) -> str:
    """Get MIME type for an image based on its extension.
    
    Args:
        image_path: Path to image file
        
    Returns:
        MIME type string
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'application/octet-stream')

def vision_to_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    prompts_map: Optional[Dict[str, str]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_image_size: int = 10 * 1024 * 1024,
    file_pattern: str = "*.{jpg,jpeg,png,gif,webp,bmp}"
) -> str:
    """Convert images to JSONL format.
    
    Args:
        input_path: Path to image directory or single image
        output_path: Path to save JSONL output
        prompt_template: Template string for image analysis
        prompts_map: Dictionary mapping image paths to prompts
        system_prompt: Optional system prompt
        model: Vision-capable model to use
        max_image_size: Maximum image file size in bytes
        file_pattern: Glob pattern for image files
        
    Returns:
        Path to generated JSONL file
    
    Raises:
        FileNotFoundError: If input path doesn't exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Default prompt template if not provided
    if not prompt_template:
        prompt_template = "Describe what you see in this image."
    
    # Create output path if not provided
    if not output_path:
        output_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.jsonl', dir=tempfile.gettempdir()
        )
        output_path = output_file.name
        output_file.close()
    
    try:
        # Determine if input_path is a directory or a single image
        image_paths = []
        if os.path.isdir(input_path):
            # Handle directory of images
            pattern = os.path.join(input_path, file_pattern)
            image_paths = glob.glob(pattern, recursive=True)
            logger.info(f"Found {len(image_paths)} images in {input_path}")
        else:
            # Handle single image
            image_paths = [input_path]
        
        # Create JSONL file
        with open(output_path, 'w') as f:
            for image_path in image_paths:
                # Skip directories
                if os.path.isdir(image_path):
                    continue
                
                # Skip files that are too large
                if os.path.getsize(image_path) > max_image_size:
                    logger.warning(f"Skipping {image_path}: image too large ({os.path.getsize(image_path)} bytes)")
                    continue
                
                try:
                    # Get custom ID from filename
                    filename = os.path.basename(image_path)
                    custom_id = os.path.splitext(filename)[0]
                    
                    # Create messages array
                    messages = []
                    
                    # Add system prompt if provided
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })
                    
                    # Get prompt for this image
                    user_prompt = prompt_template
                    if prompts_map and image_path in prompts_map:
                        user_prompt = prompts_map[image_path]
                    elif prompts_map and filename in prompts_map:
                        user_prompt = prompts_map[filename]
                    
                    # Base64 encode the image
                    encoded_image = encode_image(image_path)
                    mime_type = get_image_mime_type(image_path)
                    
                    # Create content array with text and image
                    content = [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}"
                            }
                        }
                    ]
                    
                    # Add user message with content array
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                    
                    # Create JSONL entry
                    entry = create_jsonl_entry(
                        messages=messages,
                        model=model,
                        custom_id=custom_id
                    )
                    
                    # Add image metadata
                    rel_path = os.path.relpath(image_path, input_path) if os.path.isdir(input_path) else filename
                    entry["metadata"] = {
                        "filepath": rel_path,
                        "filename": filename,
                        "size": os.path.getsize(image_path)
                    }
                    
                    # Write to JSONL file
                    f.write(json.dumps(entry) + '\n')
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
        
        logger.info(f"Created vision JSONL file at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting vision to JSONL: {e}")
        if os.path.exists(output_path) and not output_path.startswith('/tmp'):
            os.unlink(output_path)
        raise 
