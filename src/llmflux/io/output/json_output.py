#!/usr/bin/env python3
"""JSON output handler for LLMFlux."""

import json
import logging
from pathlib import Path
from typing import List

from ..base import OutputHandler, OutputResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JSONOutputHandler(OutputHandler):
    """Handler for saving results as JSON."""
    
    def save(self, results: List[OutputResult], output_path: str, indent: int = 2) -> None:
        """Save results to JSON file.
        
        Args:
            results: List of results to save
            output_path: Path to save results to
            indent: JSON indentation level (default: 2)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                results_data = [result.to_dict() for result in results]
                json.dump(results_data, f, indent=indent)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise 
