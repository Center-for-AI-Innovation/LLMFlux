#!/usr/bin/env python3
"""
SLURM-based JSONL Processing Example for LLMFlux

This example demonstrates the core functionality of the JSONL-first approach:
1. Creating a properly formatted JSONL file with OpenAI API-compatible requests
2. Submitting the JSONL file directly to SLURM for processing
3. No converters or input handlers required

This is the primary workflow for LLMFlux on HPC systems.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llmflux.core.config import Config
from llmflux.slurm.runner import SlurmRunner
from examples.utils import get_timestamped_filename, ensure_results_dir

def run_slurm_jsonl_job():
    """Launch a SLURM job that processes JSONL directly.
    
    This is the core workflow for the JSONL-first architecture:
    1. Create or obtain a JSONL file in the correct format
    2. Submit it directly to SLURM for processing
    3. Results are written to the specified output path
    """
    # Load configuration
    config = Config()
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:00:00"  # 1 hour
    slurm_config.mem = "16G"        # 16GB memory
    slurm_config.gpus_per_node = 1  # 1 GPU
    slurm_config.partition = "a100"  # A100 GPU partition
    
    print(f"Using SLURM account: {slurm_config.account}")
    
    # Define paths
    jsonl_path = 'data/slurm_batch.jsonl'
    output_path = get_timestamped_filename('results/slurm_direct_results.json')
    
    # Create test JSONL file with proper formatting
    create_test_jsonl(jsonl_path)
    
    print("\n===== LLMFLUX SLURM JSONL PROCESSING DEMO =====")
    print(f"Input JSONL: {jsonl_path}")
    print(f"Output path: {output_path}")
    print(f"SLURM settings: {slurm_config.partition}, {slurm_config.time}, {slurm_config.mem}, {slurm_config.gpus_per_node} GPU(s)")
    
    # Create runner with standard configuration
    slurm_runner = SlurmRunner(
        config=slurm_config,
        workspace=os.getcwd()
    )
    
    # Submit job - THE CORE FUNCTIONALITY
    # This demonstrates direct JSONL processing with no converters or handlers
    print("\nSubmitting job to SLURM...")
    job_id = slurm_runner.run(
        input_path=jsonl_path,     # Direct JSONL input
        output_path=output_path,   # Path for results
        
        # Optional parameters - these follow the priority system:
        # CLI/code params > environment vars > defaults
        model="llama3.2:3b",       # Model name from available templates
        max_retries=3,             # Retry failed requests
        retry_delay=1.0,           # Delay between retries (seconds)
        batch_size=4               # Items to process in parallel
    )
    
    print(f"\n✅ Job submitted successfully to SLURM with ID: {job_id}")
    print("Use the following commands to monitor your job:")
    print(f"  squeue -j {job_id}           # Check job status")
    print(f"  scontrol show job {job_id}   # View detailed job info")
    print(f"  scancel {job_id}             # Cancel the job if needed")
    print(f"\nResults will be written to: {output_path}")
    print(f"You can monitor results as they arrive: tail -f {output_path}")

def create_test_jsonl(output_path, num_entries=10):
    """Create a test JSONL file with multiple API-compatible entries.
    
    This function creates a properly formatted JSONL file for testing.
    In production, you'd typically create this file using your own data
    or use a converter utility if converting from another format.
    
    The JSONL format follows the OpenAI Batch API specification.
    
    Args:
        output_path: Path to save the JSONL file
        num_entries: Number of JSONL entries to create
    """
    # Create data directory if needed
    data_dir = Path(output_path).parent
    data_dir.mkdir(exist_ok=True)
    
    # Questions to use in the test
    questions = [
        "Explain the concept of quantum entanglement.",
        "What are the ethical implications of artificial intelligence?",
        "How does nuclear fusion work?",
        "What caused the 2008 financial crisis?",
        "Explain the theory of relativity in simple terms.",
        "What are the most promising renewable energy sources?",
        "How does the human memory work?",
        "What is the current state of climate change research?",
        "Explain how CRISPR gene editing technology works.",
        "What are the major challenges in cybersecurity today?"
    ]
    
    # Create JSONL entries
    jsonl_entries = []
    for i in range(min(num_entries, len(questions))):
        entry = {
            "custom_id": f"batch-question-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "llama3.2:3b",  # Match model name from templates
                "messages": [
                    {"role": "system", "content": "You are a knowledgeable assistant providing clear, accurate information."},
                    {"role": "user", "content": questions[i]}
                ],
                "temperature": 0.5,
                "max_tokens": 500,
                "top_p": 0.9
            }
        }
        jsonl_entries.append(entry)
    
    # Write JSONL file (one JSON object per line)
    with open(output_path, 'w') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created JSONL file at {output_path}")
    print(f"- Contains {len(jsonl_entries)} entries in OpenAI API batch format")
    
    # Show example of the JSONL format
    print("\nExample JSONL entry format:")
    print(json.dumps(jsonl_entries[0], indent=2)[:300] + "...\n")

if __name__ == "__main__":
    # Ensure results directory exists
    ensure_results_dir()
    
    # Run SLURM job with direct JSONL processing
    run_slurm_jsonl_job() 
