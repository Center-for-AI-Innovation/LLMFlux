# LLMFlux Examples

This directory contains example scripts demonstrating how to use LLMFlux for batch processing with LLMs on HPC systems.

## JSONL-First Examples

The following examples demonstrate the JSONL-first architecture that simplifies batch processing by:
1. Working directly with JSONL files in OpenAI-compatible format
2. Separating format conversion as an optional pre-processing step
3. Processing JSONL files directly with no input handlers needed

### 1. Direct JSONL Processing (`jsonl_slurm_direct.py`)

Demonstrates the core workflow - submitting JSONL files directly to SLURM:

```python
# Setup SLURM configuration
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
slurm_config.time = "01:00:00"
slurm_config.memory = "16G"
slurm_config.gpus_per_node = 1

# Create runner with configuration
slurm_runner = SlurmRunner(
    config=slurm_config,
    workspace=os.getcwd()
)

# Submit job - this is the core functionality
job_id = slurm_runner.run(
    input_path="data/prompts.jsonl",     # Direct JSONL input
    output_path="results/output.json",   # Path for results
    model="llama3.2:3b",                 # Model to use
    batch_size=4                         # Items to process in parallel
)
```

### 2. CSV to JSONL Processing (`csv_jsonl_example.py`)

Demonstrates converting a CSV file to JSONL format and then processing it:

```python
# Convert CSV to JSONL format
conversion_result = csv_to_jsonl(
    input_path='data/papers.csv',
    output_path=jsonl_path,
    prompt_template="Please summarize this paper: Title: {title}, Abstract: {abstract}",
    system_prompt="You are a research assistant specialized in summarizing papers.",
    model="llama3.2:3b"
)

# Setup SLURM configuration
slurm_config = config.get_slurm_config()
slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
slurm_config.time = "02:00:00"

# Run on SLURM with the JSONL file directly
runner = SlurmRunner(config=slurm_config, workspace=os.getcwd())
runner.run(
    input_path=jsonl_path,
    output_path=output_path,
    model="llama3.2:3b",
    batch_size=4
)
```

### 3. JSON to JSONL Processing (`json_jsonl_example.py`)

Demonstrates converting a JSON file to JSONL format and then processing it:

```python
# Convert JSON to JSONL first
conversion_result = json_to_jsonl(
    input_path=input_path,
    output_path=jsonl_path,
    model="llama3.2:3b"
)

# Setup SLURM configuration and runner
slurm_config = config.get_slurm_config()
slurm_config.account = os.getenv('SLURM_ACCOUNT', '')
runner = SlurmRunner(config=slurm_config, workspace=os.getcwd())

# Run with JSONL file directly
runner.run(
    input_path=jsonl_path,
    output_path=output_path,
    model="llama3.2:3b",
    batch_size=8
)
```

### 4. Directory to JSONL Processing (`directory_jsonl_example.py`)

Demonstrates converting a directory of files to JSONL format and then processing them:

```python
# Convert directory contents to JSONL
conversion_result = directory_to_jsonl(
    input_path=input_dir,
    output_path=jsonl_path,
    file_pattern="*.txt",  # Only process .txt files
    recursive=True,  # Include subdirectories
    prompt_template="Please analyze this file: {filename}\nContent: {content}",
    system_prompt="You are a document analysis assistant.",
    model="llama3.2:3b"
)

# Process with JSONL-first approach
runner.run(
    input_path=jsonl_path,
    output_path=output_path,
    model="llama3.2:3b",
    batch_size=4
)
```

## JSONL Format

LLMFlux uses the OpenAI Batch API specification for its JSONL files:

```jsonl
{"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"model":"llama3.2:3b","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is quantum computing?"}],"temperature":0.7,"max_tokens":500}}
{"custom_id":"request2","method":"POST","url":"/v1/chat/completions","body":{"model":"llama3.2:3b","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is machine learning?"}],"temperature":0.7,"max_tokens":500}}
```

## Running the Examples

1. Ensure you have set up your environment variables in `.env`
2. Run an example:
   ```bash
   # Run the core JSONL workflow
   python examples/jsonl_slurm_direct.py
   
   # Run a converter example
   python examples/csv_jsonl_example.py
   ```

Each example script will create sample data files if they don't exist and then submit a SLURM job for processing. 

## Further Documentation

For complete documentation of all configuration options, see:
- [Configuration Guide](../docs/CONFIGURATION.md)
- [Models Guide](../docs/MODELS.md) 
