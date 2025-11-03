#!/usr/bin/env python3
"""Minimal benchmark utilities for generating test datasets."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def generate_synthetic_prompts(
    num_prompts: int = 50,
    seed: int = 42,
    model: str = "llama3.2:3b"
) -> List[Dict[str, Any]]:
    """Generate simple synthetic benchmark prompts.
    
    Args:
        num_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        model: Model name to use in prompts
        
    Returns:
        List of JSONL-compatible prompt dictionaries
    """
    random.seed(seed)
    
    # Simple prompt templates
    templates = [
        "What is {}?",
        "Explain {} in simple terms.",
        "How does {} work?",
        "Describe the key features of {}.",
        "What are the benefits of {}?",
    ]
    
    topics = [
        "machine learning",
        "cloud computing",
        "data science",
        "artificial intelligence",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "distributed systems",
    ]
    
    prompts = []
    for i in range(num_prompts):
        template = random.choice(templates)
        topic = random.choice(topics)
        content = template.format(topic)
        
        entry = {
            "custom_id": f"bench-{i:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": 0.7,
                "max_tokens": 512,
            }
        }
        prompts.append(entry)
    
    return prompts


def save_prompts_to_jsonl(prompts: List[Dict[str, Any]], filepath: Path) -> None:
    """Save prompts to JSONL file.
    
    Args:
        prompts: List of prompt dictionaries
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
