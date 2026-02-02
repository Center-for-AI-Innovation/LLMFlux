#!/usr/bin/env python3
"""Utility functions for LLMFlux examples."""

import datetime
from pathlib import Path

def get_timestamped_filename(base_name):
    """Add a timestamp to a filename.
    
    Args:
        base_name (str): The original filename
        
    Returns:
        str: Filename with timestamp inserted before the extension
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = base_name.rsplit('.', 1)
    if len(name_parts) > 1:
        return f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
    return f"{base_name}_{timestamp}"

def ensure_results_dir():
    """Ensure the results directory exists."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    return results_dir 
