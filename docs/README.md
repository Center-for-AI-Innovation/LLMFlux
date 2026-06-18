# LLMFlux: LLM Batch Processing Pipeline for HPC Systems

A streamlined solution for running Large Language Models (LLMs) in batch mode on HPC systems powered by Slurm. LLMFlux uses the OpenAI-compatible API format with a JSONL-first architecture, enabling your prompts to flow efficiently through LLM engines at scale.

[![PyPI version](https://badge.fury.io/py/llmflux.svg)](https://pypi.org/project/llmflux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Architecture

```
      JSONL Input                    Batch Processing                    Results
   (OpenAI Format)                 (Ollama/vLLM + Model)               (JSON Output)
         │                                 │                                 │
         │                                 │                                 │
         ▼                                 ▼                                 ▼
    ┌──────────┐                   ┌──────────────┐                   ┌──────────┐
    │  Batch   │                   │              │                   │  Output  │
    │ Requests │─────────────────▶ │   Model on   │─────────────────▶ │  Results │
    │  (JSONL) │                   │    GPU(s)    │                   │  (JSON)  │
    └──────────┘                   │              │                   └──────────┘
                                   └──────────────┘                    
```

LLMFlux processes JSONL files in a standardized OpenAI-compatible batch API format, enabling efficient processing of thousands of prompts on HPC systems with minimal overhead.

## Documentation

- [Configuration Guide](CONFIGURATION.md) - How to configure LLMFlux
- [Models Guide](MODELS.md) - Supported models and requirements
- [Repository Structure](REPOSITORY_STRUCTURE.md) - Codebase organization
- [Testing Guide](TESTING.md) - How to run tests

## Installation

> **Prerequisites:** LLMFlux runs models inside [Apptainer](https://apptainer.org/) (formerly Singularity) containers on the compute nodes. Apptainer must be available on your HPC cluster — contact your system administrator if you are unsure. No local GPU is required; everything runs on the SLURM nodes.

```bash
pip install llmflux
```

Or for development:

1. **Create and Activate Conda Environment:**
   ```bash
   conda create -n llmflux python=3.11 -y
   conda activate llmflux
   ```

2. **Install Package:**
   ```bash
   pip install -e .
   ```

3. **Environment Setup:**
   ```bash
   cp .env.example .env
   # Edit .env with your SLURM account and model details
   ```
   
Confirm the installation by running a base command and ensuring your system gives the correct output:

```bash
$llmflux -h
usage: llmflux [-h] [--version] {run,benchmark,show-models,jobs,status,logs,cancel} ...

LLMFlux CLI

positional arguments:
  {run,benchmark,show-models,jobs,status,logs,cancel}
    run                 Submit a batch processing job
    benchmark           Run a benchmark job
    show-models         List all available model keys from models.yaml
    jobs                List LLMFlux tracked Slurm jobs
    status              Show detailed status for a job
    logs                Show last lines of stdout and stderr for a tracked job
    cancel              Cancel a tracked running/pending job

options:
  -h, --help            show this help message and exit
  --version, -V         Show llmflux version and exit
```

## Quick Start

### Core Batch Processing on SLURM

The primary workflow for LLMFlux is submitting JSONL files for batch processing on SLURM:

```python
from llmflux.slurm import SlurmRunner
from llmflux.core.config import Config

# Setup SLURM configuration
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = "myaccount"

# Initialize runner
runner = SlurmRunner(config=slurm_config)

# Submit JSONL file directly for processing
job_id = runner.run(
    input_path="prompts.jsonl",
    output_path="results.json",
    model="Llama-3.2-3B-Instruct",
    batch_size=4
)
print(f"Job submitted with ID: {job_id}")
```

### JSONL Input Format

JSONL input format follows the OpenAI Batch API specification:

```jsonl
{"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Explain quantum computing"}],"temperature":0.7,"max_tokens":500}}
{"custom_id":"request2","method":"POST","url":"/v1/chat/completions","body":{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is machine learning?"}],"temperature":0.7,"max_tokens":500}}
```

For advanced options like custom batch sizes, processing settings, or SLURM configuration, see the [Configuration Guide](CONFIGURATION.md).

For advanced model configuration, see the [Models Guide](MODELS.md).

## Command-Line Interface

LLMFlux uses **model keys** (e.g. `Llama-3.2-3B-Instruct`) to identify models. Run `llmflux show-models` to see every available key and which engines each supports. The `--model` argument always takes a model key, not an Ollama tag or HuggingFace repo name.

```bash
# Process a JSONL file with the default vLLM engine
llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl --output results/output.json

# Use the Ollama engine instead
llmflux run --model Llama-3.2-3B-Instruct --engine ollama --input data/prompts.jsonl --output results/output.json

# Specify a SLURM account and partition
llmflux run \
   --account your-account \
   --partition gpu \
   --model Llama-3.2-3B-Instruct \
   --input data/prompts.jsonl \
   --output results/output.json
```

`--output` is optional. When omitted, results are written to a timestamped file in your configured workspace output directory.

The model key determines which HuggingFace repository vLLM downloads (e.g. `MistralLite` maps to `amazon/MistralLite`) and which Ollama tag is pulled. Use `llmflux show-models` to look up the key for any model.

**HuggingFace token:** Some models (e.g. Llama) are gated and require a token. Once you have one, add it to your `.env` file:

```bash
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXX
```

Visit the model's HuggingFace page and accept the terms to activate access. You may also need to grant your token read access to gated repos in your HuggingFace account settings. Models are cached by default at `~/.cache/huggingface/hub`; set `HF_HOME` in `.env` to change this location.

LLMFlux downloads models automatically for both vLLM and Ollama on first use.

For detailed command options:
```bash
llmflux --help
```

### Job Control Commands

LLMFlux tracks submitted jobs in a local registry (`~/.llmflux/jobs.json`) and
combines that metadata with Slurm state.

```bash
# List tracked jobs (default: active states only)
llmflux jobs

# Include historical states
llmflux jobs --all

# Filter by one or more states
llmflux jobs --state RUNNING --state FAILED

# Show detailed merged status for one job
llmflux status <job-id>

# Tail logs (default: 100 lines)
llmflux logs <job-id>
llmflux logs <job-id> --tail 200
llmflux logs <job-id> -f

# Cancel a tracked job
llmflux cancel <job-id>
llmflux cancel <job-id> --force
```

Notes:
- `jobs` and `status` derive live state from Slurm JSON output.
- `logs` and `cancel` only operate on jobs present in the LLMFlux registry.

## Output Format

Results are saved in the user's workspace:

```json
[
  {
    "input": {
      "custom_id": "request1",
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
        "messages": [
          {"role": "system", "content": "You are a helpful assistant"},
          {"role": "user", "content": "Original prompt text"}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
      },
      "metadata": {
        "source_file": "example.txt"
      }
    },
    "output": {
      "id": "chat-cmpl-123",
      "object": "chat.completion",
      "created": 1699123456,
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Generated response text"
          },
          "finish_reason": "stop"
        }
      ]
    },
    "metadata": {
      "model": "llama3.2:3b",
      "timestamp": "2023-11-04T12:34:56.789Z",
      "processing_time": 1.23
    }
  }
]
```

## Utility Converters

LLMFlux provides utility converters to help prepare JSONL input files from common formats.

### CSV converter

```bash
llmflux convert csv \
    --input data/papers.csv \
    --output data/papers.jsonl \
    --template "Summarize the following abstract: {abstract}"
```

- `--template` is a Python format string whose placeholders (`{column_name}`) are filled from each CSV row. Every column in the CSV can be used as a placeholder.
- `--output` is optional; if omitted, the output is written alongside the input file with a `.jsonl` extension.

### Directory converter

```bash
llmflux convert dir \
    --input data/documents/ \
    --output data/docs.jsonl \
    --recursive
```

- Each file in the directory becomes one JSONL entry. The file contents are placed in the `user` message.
- `--recursive` descends into subdirectories.
- Supported file types: `.txt`, `.md`, `.rst`, and plain text files without an extension.

For additional code examples, see the [examples directory](examples/).

## Benchmarking

LLMFlux ships with a benchmarking workflow that can source prompts, submit the SLURM job, and collect results/metrics for you.

```bash
llmflux benchmark \
    --model Llama-3.2-3B-Instruct \
    --name nightly \
    --num-prompts 60 \
    --account ACCOUNT_NAME \
    --partition PARTITION_NAME \
    --nodes 1
```

- **Prompt sources**: omit `--input` to automatically download and cache LiveBench categories (``benchmark_data/``). Provide `--input path/to/prompts.jsonl` to reuse an existing JSONL file instead. Use `--num-prompts`, `--temperature`, and `--max-tokens` to control synthetic dataset generation.
- **Outputs**: results default to `results/benchmarks/<name>_results.json` and a metrics summary (`<name>_metrics.txt`) containing elapsed SLURM runtime and number of prompts processed.
- **Batch tuning**: adjust `--batch-size` for throughput. Pass model arguments such as `--temperature` and `--max-tokens` to forward them to the runner.
- **SLURM overrides**: forward scheduler settings with `--account`, `--partition`, `--nodes`, `--gpus-per-node`, `--time`, `--mem`, and `--cpus-per-task`.
- **Job controls**: add `--rebuild` to force an Apptainer image rebuild or `--debug` to keep the generated job script for inspection.

For the complete option reference:

```bash
llmflux benchmark --help
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](../LICENSE)
