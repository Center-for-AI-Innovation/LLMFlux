#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

class BaseProcessor(ABC):
    """Base class for all processors."""
    
    def __init__(self, model: str):
        """Initialize processor with model name."""
        self.model = model
        
    @abstractmethod
    def setup(self) -> None:
        """Setup processor (e.g., start server, load model)."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    @abstractmethod
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inputs.
        
        Args:
            batch: List of input items to process
            
        Returns:
            List of processed results
        """
        pass
    
    def validate_input(self, item: Dict[str, Any]) -> bool:
        """Validate input item.
        
        Args:
            item: Input item to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True  # Base implementation accepts all inputs
    
    def format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output result.
        
        Args:
            result: Raw result to format
            
        Returns:
            Formatted result
        """
        return result  # Base implementation returns as-is 