#!/usr/bin/env python3
"""Example for processing JSON files with the JSONL-first approach in LLMFlux."""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llmflux.core.config import Config
from llmflux.processors.batch import BatchProcessor
from llmflux.converters import json_to_jsonl
from llmflux.slurm.runner import SlurmRunner
from examples.utils import get_timestamped_filename, ensure_results_dir

def process_json_with_jsonl(input_path: str, output_path: str):
    """Process JSON files using the JSONL-first approach.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save results
    """
    # Load configuration
    config = Config()
    
    # Setup SLURM configuration
    slurm_config = config.get_slurm_config()
    slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
    slurm_config.time = "01:00:00"  # 1 hour
    slurm_config.memory = "16G"        # 16GB memory
    slurm_config.gpus_per_node = 1  # 1 GPU
    slurm_config.partition = "a100"  # A100 GPU partition
    
    print(f"Using SLURM account: {slurm_config.account}")
    
    print("\n===== LLMFLUX JSON TO JSONL PROCESSING DEMO =====")
    print(f"Input JSON: {input_path}")
    print(f"Output path: {output_path}")
    print(f"SLURM settings: {slurm_config.partition}, {slurm_config.time}, {slurm_config.memory}, {slurm_config.gpus_per_node} GPU(s)")
    
    # Convert JSON to JSONL first
    print("\nConverting JSON to JSONL...")
    jsonl_path = str(Path(input_path).with_suffix('.jsonl'))
    conversion_result = json_to_jsonl(
        input_path=input_path,
        output_path=jsonl_path,
        model="llama3.2:3b",
        # Don't pass additional API parameters that might conflict with the JSON content
        api_parameters={}
    )
    
    print(f"\n✅ Converted JSON to JSONL: {jsonl_path}")
    print(f"Total items: {conversion_result.get('total_items', 0)}")
    print(f"Successful conversions: {conversion_result.get('successful_conversions', 0)}")
    if conversion_result.get('failed_conversions'):
        print(f"Failed conversions: {conversion_result.get('failed_conversions', 0)}")
    
    # Show example of the JSONL format
    try:
        with open(jsonl_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                first_entry = json.loads(first_line)
                print("\nExample JSONL entry format:")
                print(json.dumps(first_entry, indent=2)[:300] + "...\n")
            else:
                print("\nWarning: JSONL file is empty. No entries were converted successfully.")
    except (json.JSONDecodeError, FileNotFoundError, IndexError) as e:
        print(f"\nError reading JSONL file: {e}")
    
    # Submit job to SLURM only if we have successful conversions
    if conversion_result.get('successful_conversions', 0) > 0:
        print("\nSubmitting job to SLURM...")
        job_id = SlurmRunner(
            config=slurm_config,
            workspace=os.getcwd()
        ).run(
            input_path=jsonl_path,
            output_path=output_path,
            model="llama3.2:3b",       # Model name from available templates
            max_retries=3,             # Retry failed requests
            retry_delay=1.0,           # Delay between retries (seconds)
            batch_size=8               # Items to process in parallel
        )
        
        print(f"\n✅ Job submitted successfully to SLURM with ID: {job_id}")
        print("Use the following commands to monitor your job:")
        print(f"  squeue -j {job_id}           # Check job status")
        print(f"  scontrol show job {job_id}   # View detailed job info")
        print(f"  scancel {job_id}             # Cancel the job if needed")
        print(f"\nResults will be written to: {output_path}")
        print(f"You can monitor results as they arrive: tail -f {output_path}")
    else:
        print("\n❌ No successful conversions. SLURM job not submitted.")

def create_example_json_data():
    """Create example JSON data for processing."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create example data - using a format that won't conflict with the converter
    prompts = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantum computing in simple terms."
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with expertise in healthcare."
                },
                {
                    "role": "user",
                    "content": "How is machine learning being used in healthcare today?"
                }
            ]
        }
    ]
    
    # Write to file
    json_path = data_dir / "prompts.json"
    with open(json_path, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Created example data at {json_path}")
    print("\nExample JSON content:")
    print(json.dumps(prompts, indent=2)[:300] + "...\n")
    
    return str(json_path)

if __name__ == "__main__":
    # Ensure results directory exists
    ensure_results_dir()
    
    # Create example data
    print("\n===== CREATING EXAMPLE JSON DATA =====")
    input_file = create_example_json_data()
    
    # Process JSON with JSONL-first approach
    output_file = get_timestamped_filename("results/jsonl_batch_results.json")
    process_json_with_jsonl(input_file, output_file) 
