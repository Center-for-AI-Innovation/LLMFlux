#!/usr/bin/env python3
"""Base classes for input and output handling in LLMFlux."""

import abc
from typing import Dict, Any, List, Iterator, Optional, Union
from pathlib import Path


class InputHandler(abc.ABC):
    """Base class for input handlers."""
    
    @abc.abstractmethod
    def process(self, input_source: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Process input source and yield items in OpenAI-compatible format.
        
        Args:
            input_source: Source of input data
            **kwargs: Additional parameters for processing
            
        Yields:
            Dict items with 'messages' in OpenAI format
        """
        pass


class OutputResult:
    """Container for input, output, and metadata."""
    
    def __init__(
        self,
        input: Dict[str, Any],
        output: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize output result.
        
        Args:
            input: Input data with messages in OpenAI format
            output: Generated output
            error: Optional error message if processing failed
            metadata: Additional metadata
        """
        self.input = input
        self.output = output
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict representation of result
        """
        result = {
            "input": self.input,
            "metadata": self.metadata
        }
        
        if self.output is not None:
            result["output"] = self.output
            
        if self.error is not None:
            result["error"] = self.error
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputResult':
        """Create from dictionary.
        
        Args:
            data: Dict representation of result
            
        Returns:
            OutputResult instance
        """
        return cls(
            input=data.get("input", {}),
            output=data.get("output"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


class OutputHandler(abc.ABC):
    """Base class for output handlers."""
    
    @abc.abstractmethod
    def save(self, results: List[OutputResult], output_path: str) -> None:
        """Save results to output path.
        
        Args:
            results: List of results to save
            output_path: Path to save results to
        """
        pass 
