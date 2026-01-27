#!/usr/bin/env python3
"""Example script for CSV to JSONL conversion and processing with LLMFlux."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llmflux import BatchProcessor
from llmflux.core.config import Config
from llmflux.converters import csv_to_jsonl
from llmflux.slurm import SlurmRunner
from examples.utils import get_timestamped_filename, ensure_results_dir

def process_csv_with_jsonl():
    """Example of processing a CSV file using the JSONL-first approach."""
    # Load model configuration
    config = Config()
    model_config = config.load_model_config("phi3", "mini")
    
    # Define a system prompt for context
    system_prompt = "You are a research assistant specializing in summarizing scientific papers."
    
    # Create timestamped output path
    jsonl_path = get_timestamped_filename('data/papers.jsonl')
    output_path = get_timestamped_filename('results/paper_summaries.json')
    
    # Convert CSV to JSONL format first
    conversion_result = csv_to_jsonl(
        input_path='data/papers.csv',
        output_path=jsonl_path,
        prompt_template=(
            "Please summarize the following research paper:\n\n"
            "Title: {title}\n"
            "Abstract: {abstract}\n\n"
            "Provide a concise summary focusing on:\n"
            "1. Main research question\n"
            "2. Key methodology\n"
            "3. Main findings\n"
            "4. Significance of results"
        ),
        system_prompt=system_prompt,
        model="llama3.2:3b"
    )
    
    print(f"Converted CSV to JSONL: {jsonl_path}")
    print(f"Total rows: {conversion_result['total_rows']}")
    print(f"Successful conversions: {conversion_result['successful_conversions']}")
    
    # Initialize processor for JSONL processing
    processor = BatchProcessor(
        model_config=model_config,
        batch_size=4
    )
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "02:00:00"
    
    # Run on SLURM with the JSONL file directly
    runner = SlurmRunner(config=slurm_config, workspace=os.getcwd())
    runner.run(
        processor=processor,  # processor is now optional and can be None
        input_path=jsonl_path,
        output_path=output_path
    )
    
    print(f"Results saved to: {output_path}")

def create_example_csv_data():
    """Create example CSV data for batch processing."""
    # Create example data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create example CSV input
    csv_input = """title,abstract
Quantum Supremacy Using a Programmable Superconducting Processor,"The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space."
Large Language Models in Machine Learning,"Recent advances in transformer architectures and pre-training techniques have led to significant improvements in natural language processing tasks. This paper surveys the current state of large language models and their applications."
"""
    
    with open(data_dir / 'papers.csv', 'w') as f:
        f.write(csv_input)

if __name__ == '__main__':
    # Ensure results directory exists
    ensure_results_dir()
    
    create_example_csv_data()
    print("Processing CSV data with JSONL-first approach...")
    process_csv_with_jsonl() 
