#!/usr/bin/env python3
"""Minimal benchmark utilities for generating test datasets."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


BENCHMARK_DATA_DIR = Path(__file__).resolve().parent / "benchmark_data"


def ensure_benchmark_data_dir() -> Path:
    """Create the benchmark data directory if it does not exist."""
    BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return BENCHMARK_DATA_DIR


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


def extract_prompts_from_jsonl(
    filepath: Path,
    num_prompts: int = 20
) -> List[Dict[str, Any]]:
    """Extract a shuffled subset of prompts from a JSONL file.

    Args:
        filepath: Input file path
        num_prompts: Maximum number of prompts to return. Use ``0`` to return an
            empty list and negative values to return all prompts.
    """
    if num_prompts == 0:
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    if num_prompts < 0 or num_prompts >= len(prompts):
        return prompts

    random.shuffle(prompts)

    return prompts[:num_prompts]

def create_test_prompts_file(num_prompts: int = 120, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Get test prompts for a given model from 6 LiveBench categories: data_analysis, language, math, reasoning, instruction_following, and coding.
    Args:
        num_prompts: Total number of prompts to generate.
        temperature: Sampling temperature for the prompts.
        max_tokens: Maximum number of tokens for each prompt.
    Returns:
        Path to the created prompts file.
    """
    file_names = ["data_analysis", "language", "math", "reasoning", "instruction_following", "coding"]
    # Check if all the files exist else download the prompts data
    benchmark_dir = ensure_benchmark_data_dir()
    if not all((benchmark_dir / f"{file_name}.jsonl").exists() for file_name in file_names):
        print("Downloading prompts data from the LiveBench HuggingFace dataset...")
        download_prompts_data()
        print("Prompts data downloaded successfully")
    
    all_prompts = []
    prompts_file = benchmark_dir / "benchmark_prompts.jsonl"
    for i in range(len(file_names)):
        file_name = benchmark_dir / f"{file_names[i]}.jsonl"
        prompts = extract_prompts_from_jsonl(file_name, num_prompts=num_prompts//len(file_names))
        for prompt in prompts:
            all_prompts.append({"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"messages":prompt,"temperature":temperature,"max_tokens":max_tokens}})
    save_prompts_to_jsonl(all_prompts, prompts_file)

    return str(prompts_file)



def save_prompts_to_jsonl(prompts: List[Dict[str, Any]], filepath: Path) -> None:
    """Save prompts to JSONL file.
    
    Args:
        prompts: List of prompt dictionaries
        filepath: Output file path
    """

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')

def download_prompts_data() -> None:
    """
    Download prompts data from the LiveBench HuggingFace dataset.
    Categories:  "coding", "data_analysis", "instruction_following","math","reasoning","language"
    """

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "download_prompts_data requires the optional dependencies 'datasets' and 'pandas'. "
            "Install them to enable downloading prompts."
        ) from exc

    # LiveBench categories
    CATEGORIES = [
        "coding",
        "data_analysis", 
        "instruction_following",
        "math",
        "reasoning",
        "language",
    ]

    # System prompts by category
    SYSTEM_PROMPTS = {
        "coding": "You are an expert programmer. Provide accurate, well-structured code solutions.",
        "math": "You are a mathematics expert. Think step by step and provide precise solutions.",
        "data_analysis": "You are a data analysis expert. Provide accurate and detailed analysis.",
        "reasoning": "You are an expert at logical reasoning. Think through problems systematically.",
        "instruction_following": "You are a helpful assistant that follows instructions precisely.",
        "language": "You are an expert in language and communication.",
    }

    def download_and_convert_category(category: str) -> None:
        """
        Download and convert a category from the LiveBench HuggingFace dataset.
        """
        print(f"Loading {category} from HuggingFace...")
        dataset = load_dataset(f"livebench/{category}", split="test")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Get system prompt for category
        system_prompt = SYSTEM_PROMPTS[category]

        # Create output directories
        benchmark_dir = ensure_benchmark_data_dir()

        # Function to convert to OpenAI batch format
        def create_openai_batch_row(row: pd.Series, system_prompt: str) -> Dict[str, Any]:
            """
            Create an OpenAI batch row from a LiveBench row.
            """
            messages = [{"role": "system", "content": system_prompt}]
            for turn in row["turns"]:
                messages.append({
                "role": "user",
                "content": turn
            })
    
            return messages
        
        prompts = []
        # Convert to OpenAI batch format
        for index, row in df.iterrows():
            openai_row = create_openai_batch_row(row, system_prompt)
            prompts.append(openai_row)
        
        # Save OpenAI batch format to JSONL file
        openai_path = benchmark_dir / f"{category}.jsonl"
        save_prompts_to_jsonl(prompts, openai_path)
        print(f"Saved {category} prompts to {openai_path}")
        return None
    
    # Download and convert all categories
    for category in CATEGORIES:
        try:
            download_and_convert_category(category)
        except Exception as e:
            print(f"Error downloading and converting {category}: {e}")
    
    return None
