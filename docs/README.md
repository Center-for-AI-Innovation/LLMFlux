---
layout: default
title: LLMFlux Documentation
---

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

## Installation

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
usage: llmflux [-h] {run,benchmark} ...

LLMFlux CLI

positional arguments:
  {run,benchmark}
    run            Submit a batch processing job
    benchmark      Run a benchmark job

options:
  -h, --help       show this help message and exit
$llmflux -v
llmflux 0.1.2
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
    model="llama3.2:3b",
    batch_size=4
)
print(f"Job submitted with ID: {job_id}")
```

### JSONL Input Format

JSONL input format follows the OpenAI Batch API specification:

```jsonl
{"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"model":"llama3.2:3b","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Explain quantum computing"}],"temperature":0.7,"max_tokens":500}}
{"custom_id":"request2","method":"POST","url":"/v1/chat/completions","body":{"model":"llama3.2:3b","messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is machine learning?"}],"temperature":0.7,"max_tokens":500}}
```

For advanced options like custom batch sizes, processing settings, or SLURM configuration, see the [Configuration Guide](CONFIGURATION.md).

For advanced model configuration, see the [Models Guide](MODELS.md).

## Command-Line Interface

LLMFlux includes a command-line interface for submitting batch processing jobs. It uses Ollama as it's default engine, and model configurations rely on the Ollama naming scheme. To process your prompts.jsonl file using the Ollama engine running the llama3.2 model with 3b parameters, you would run the command:

```bash
# Process JSONL file directly (core functionality)
llmflux run --model llama3.2:3b --input data/prompts.jsonl --output results/output.json
```

In addition to the default OLLAMA engine, LLMFlux can also be run using vLLM, to take advantage of HuggingFace models. In order to use a model that requires a HuggingFace key, you will first need to update the default .env parameter to use your personal token. You then can call using the names as established in the templates dir:

```bash
# Process JSONL file using VLLM backend
llmflux run --model llama3.2:3b --input data/prompts.jsonl --output results/output.json --engine=vllm
```

This will run the same as above, using VLLM as the backend interface. If you wanted to run mistral-lite, for example, checking the file mistral-lite/7b.yaml reveals the name: "mistrallite:7b". Update to the appropriate HuggingFace key and run 
```bash
# Process JSONL file using VLLM backend
llmflux run --model mistral-lite:7b --input data/prompts.jsonl --output results/output.json --engine=vllm
```
this will run the model, as noted in the config, by searching HuggingFace for `hf_name: "amazon/MistralLite"`. You will
need to check an existing model file from the folder src/llmflux/templates to find a configuration that matches what you want
and use the name as the argument for the --model argument.

Note that in order to use some HuggingFace models, you will need a key from HF. Once you have a token, update your
local copy of the .env file and add or change this line:

```bash
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXX
```
to use the token, replace the hf_XXXX piece with your token. For some gated repos, you will have to visit the huggingface repository directly and activate access (often by accepting a terms and conditions agreement). You may also need to adjust settings on your HF token to ensure that LLMFlux has proper rights to access the model. In addition, the model will by default be stored in your base directory: `~/.cache/huggingfacel/hub`. To change this, you can add the following parameter to your `.env` file:
```bash
HF_HOME=/path/to/dir
```
llmflux will automatically download the appropriate models for both OLLAMA and vLLM.

For detailed command options:
```bash
llmflux --help
```

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
        "model": "llama3.2:3b",
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
      "model": "llama3.2:3b",
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

LLMFlux provides utility converters to help prepare JSONL files from various input formats:

```bash
# Convert CSV to JSONL
llmflux convert csv --input data/papers.csv --output data/papers.jsonl --template "Summarize: {text}"

# Convert directory to JSONL
llmflux convert dir --input data/documents/ --output data/docs.jsonl --recursive
```

For code examples of converters, see the [examples directory](examples/).

## Benchmarking

LLMFlux ships with a benchmarking workflow that can source prompts, submit the SLURM job, and collect results/metrics for you.

```bash
llmflux benchmark --model llama3.2:3b --name nightly --num-prompts 60 \
  --account ACCOUNT_NAME --partition PARTITION_NAME --nodes 1
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
