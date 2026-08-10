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

# Extensions vLLM/Ollama vision models accept, mapped to the MIME type used in
# the data: URL. Also drives directory discovery when no file_pattern is given.
"""Largest image accepted by default.

Sized for real inputs rather than for the request path: 12MP phone JPEGs land
at 2-6MB, but 48MP phones, DSLR JPEGs and full-screen PNG screenshots
routinely pass 10MB, and dropping those loses legitimate work. Note base64
adds about a third, so this bounds the request body at roughly 33MB per image.
"""
DEFAULT_MAX_IMAGE_SIZE = 25 * 1024 * 1024

IMAGE_MIME_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
}


def _expand_braces(pattern: str) -> List[str]:
    """Expand one {a,b,c} group into separate patterns.

    glob has no brace expansion, so "*.{jpg,png}" silently matches nothing.
    Callers that pass such a pattern (it used to be this function's default)
    get the extensions they meant.
    """
    start = pattern.find('{')
    end = pattern.find('}', start + 1)
    if start == -1 or end == -1:
        return [pattern]
    prefix, options, suffix = pattern[:start], pattern[start + 1:end], pattern[end + 1:]
    return [f"{prefix}{option}{suffix}" for option in options.split(',')]


def _find_images(directory: str, file_pattern: Optional[str]) -> List[str]:
    """List image files in a directory.

    With no file_pattern, every supported extension is matched case-insensitively
    so phone/camera names like IMG_1234.JPG are included. An explicit pattern is
    passed to glob (brace groups expanded), keeping "**/*.jpg" style recursion.
    """
    if file_pattern is None:
        return sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.splitext(name)[1].lower() in IMAGE_MIME_TYPES
            and os.path.isfile(os.path.join(directory, name))
        )

    matches = set()
    for expanded in _expand_braces(file_pattern):
        matches.update(glob.glob(os.path.join(directory, expanded), recursive=True))
    return sorted(matches)


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
    return IMAGE_MIME_TYPES.get(ext, 'application/octet-stream')

def vision_to_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    prompts_map: Optional[Dict[str, str]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    file_pattern: Optional[str] = None,
    return_report: bool = False
):
    """Convert images to JSONL format.

    Args:
        input_path: Path to image directory or single image
        output_path: Path to save JSONL output
        prompt_template: Template string for image analysis
        prompts_map: Dictionary mapping image paths to prompts
        system_prompt: Optional system prompt
        model: Vision-capable model to use
        max_image_size: Maximum image file size in bytes. Base64 encoding adds
            about a third, so this also bounds request body size.
        file_pattern: Optional glob pattern for image files. Defaults to every
            supported extension, matched case-insensitively and non-recursively.
            Pass e.g. "**/*.jpg" to recurse into subdirectories.
        return_report: If True, return a (path, report) tuple instead of a bare
            path. The report names the images that did not make it into the
            JSONL, which is otherwise only visible in the logs.

    Returns:
        Path to generated JSONL file, or (path, report) if return_report is True.
        The report is a dict with 'found', 'converted', 'skipped_too_large' and
        'failed' keys; the latter two hold per-image detail dicts.

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
            image_paths = _find_images(input_path, file_pattern)
            if image_paths:
                logger.info(f"Found {len(image_paths)} images in {input_path}")
            else:
                # Loud, because the result is a valid-but-empty JSONL file.
                logger.warning(
                    f"No images found in {input_path}"
                    + (f" matching pattern '{file_pattern}'" if file_pattern else "")
                    + f". Supported extensions: {', '.join(sorted(IMAGE_MIME_TYPES))}"
                )
        else:
            # Handle single image
            image_paths = [input_path]
        
        report: Dict[str, Any] = {
            "found": len(image_paths),
            "converted": 0,
            "skipped_too_large": [],
            "failed": [],
        }

        # Create JSONL file
        with open(output_path, 'w') as f:
            for image_path in image_paths:
                # Skip directories
                if os.path.isdir(image_path):
                    continue

                # Skip files that are too large
                size = os.path.getsize(image_path)
                if size > max_image_size:
                    logger.warning(
                        f"Skipping {image_path}: image too large "
                        f"({size} bytes > max_image_size {max_image_size})"
                    )
                    report["skipped_too_large"].append({
                        "path": image_path,
                        "size": size,
                        "limit": max_image_size,
                    })
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
                    report["converted"] += 1

                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    report["failed"].append({"path": image_path, "error": str(e)})

        # Summarize losses once, so a short JSONL is explained in one place
        # rather than only in per-image lines scattered through the log.
        dropped = len(report["skipped_too_large"]) + len(report["failed"])
        if dropped:
            logger.warning(
                f"Converted {report['converted']} of {report['found']} images: "
                f"{len(report['skipped_too_large'])} too large, "
                f"{len(report['failed'])} failed. Pass return_report=True for details."
            )

        logger.info(f"Created vision JSONL file at {output_path}")
        return (output_path, report) if return_report else output_path

    except Exception as e:
        logger.error(f"Error converting vision to JSONL: {e}")
        if os.path.exists(output_path) and not output_path.startswith('/tmp'):
            os.unlink(output_path)
        raise 
