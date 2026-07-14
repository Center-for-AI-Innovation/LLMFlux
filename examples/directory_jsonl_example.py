#!/usr/bin/env python3
"""Example for processing a directory of files with the JSONL-first approach in LLMFlux."""

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
from llmflux.converters import directory_to_jsonl
from llmflux.slurm.runner import SlurmRunner
from examples.utils import get_timestamped_filename, ensure_results_dir

def process_directory_with_jsonl():
    """Process a directory of text files using the JSONL-first approach."""
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
    
    # Define input and output paths
    input_dir = Path('data/text_files')
    jsonl_path = get_timestamped_filename('data/directory_content.jsonl')
    output_path = get_timestamped_filename('results/directory_results.json')
    
    print("\n===== LLMFLUX DIRECTORY TO JSONL PROCESSING DEMO =====")
    print(f"Input directory: {input_dir}")
    print(f"Intermediate JSONL: {jsonl_path}")
    print(f"Output path: {output_path}")
    print(f"SLURM settings: {slurm_config.partition}, {slurm_config.time}, {slurm_config.memory}, {slurm_config.gpus_per_node} GPU(s)")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\n❌ Error: Input directory {input_dir} does not exist.")
        print("Please run the script again to create example files first.")
        return
    
    # Convert directory contents to JSONL first
    print("\nConverting directory contents to JSONL...")
    conversion_result = directory_to_jsonl(
        input_path=input_dir,
        output_path=jsonl_path,
        file_pattern="*.txt",  # Only process .txt files
        recursive=True,  # Include subdirectories
        prompt_template=(
            "Please analyze the following text file:\n\n"
            "Content:\n{content}\n\n"
            "Provide a summary and extract key information."
        ),
        system_prompt="You are a document analysis assistant specialized in extracting information from text files.",
        model="llama3.2:3b"
    )
    
    print(f"\n✅ Converted directory contents to JSONL: {jsonl_path}")
    print(f"Total files: {conversion_result.get('total_files', 0)}")
    print(f"Successful conversions: {conversion_result.get('successful_conversions', 0)}")
    if conversion_result.get('failed_conversions'):
        print(f"Failed conversions: {conversion_result.get('failed_conversions', 0)}")
    if conversion_result.get('skipped_files'):
        print(f"Skipped files: {len(conversion_result.get('skipped_files', []))}")
    
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
            batch_size=4               # Items to process in parallel
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

def create_example_files():
    """Create example text files for directory processing."""
    # Create directory structure
    data_dir = Path('data/text_files')
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectory
    reports_dir = data_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Create example files
    files = [
        (data_dir / 'sample1.txt', "This is a sample document with important information. The project deadline is March 15, 2025."),
        (data_dir / 'sample2.txt', "Meeting notes: Discussed the new JSONL-first architecture and its benefits for scalability."),
        (reports_dir / 'quarterly_report.txt', "Q1 2025 Report: Revenue increased by 15% compared to previous quarter. New initiatives are showing promising results.")
    ]
    
    # Write content to files
    for file_path, content in files:
        with open(file_path, 'w') as f:
            f.write(content)
    
    print(f"Created {len(files)} example files in {data_dir}")
    # Show example of the files created
    for file_path, content in files:
        print(f"  - {file_path.relative_to(Path('data'))}: {content[:50]}...")

if __name__ == "__main__":
    # Ensure results directory exists
    ensure_results_dir()
    
    # Create example files
    print("\n===== CREATING EXAMPLE FILES =====")
    create_example_files()
    
    # Process directory with JSONL-first approach
    process_directory_with_jsonl() 
