"""
Converters module for LLMFlux.

This module provides utility functions for converting various input formats
to the standardized JSONL format used by the batch processor.
"""

from .csv import csv_to_jsonl
from .directory import directory_to_jsonl
from .vision import vision_to_jsonl
from .json import json_to_jsonl
from .utils import (
    validate_jsonl,
    merge_jsonl_files,
    jsonl_to_json,
    read_jsonl,
    create_jsonl_entry,
    write_jsonl_entry,
    generate_custom_id
)

__all__ = [
    # CSV converter
    'csv_to_jsonl',
    
    # Directory converter
    'directory_to_jsonl',
    
    # Vision converter
    'vision_to_jsonl',
    
    # JSON converter
    'json_to_jsonl',
    
    # JSONL utilities
    'validate_jsonl',
    'merge_jsonl_files',
    'jsonl_to_json',
    'read_jsonl',
    'create_jsonl_entry',
    'write_jsonl_entry',
    'generate_custom_id'
] 
