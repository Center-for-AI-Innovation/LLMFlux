---
layout: default
title: AI-Flux Documentation
---

# AI-Flux: LLM Batch Processing Pipeline for HPC Systems

A streamlined solution for running Large Language Models (LLMs) in batch mode on HPC systems powered by Slurm. AI-Flux uses the OpenAI-compatible API format with a JSONL-first architecture for all interactions.

## Architecture

```
      JSONL Input                    Batch Processing                    Results
   (OpenAI Format)                  (Ollama + Model)                   (JSON Output)
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

AI-Flux processes JSONL files in a standardized OpenAI-compatible batch API format, enabling efficient processing of thousands of prompts on HPC systems with minimal overhead.

## Documentation

- [Configuration Guide](CONFIGURATION.md) - How to configure AI-Flux
- [Models Guide](MODELS.md) - Supported models and requirements
- [Repository Structure](REPOSITORY_STRUCTURE.md) - Codebase organization

## Installation

1. **Create and Activate Conda Environment:**
   ```bash
   conda create -n aiflux python=3.11 -y
   conda activate aiflux
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

## Quick Start

### Core Batch Processing on SLURM

The primary workflow for AI-Flux is submitting JSONL files for batch processing on SLURM:

```python
from aiflux.slurm import SlurmRunner
from aiflux.core.config import Config

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

AI-Flux includes a command-line interface for submitting batch processing jobs:

```bash
# Process JSONL file directly (core functionality)
aiflux run --model llama3.2:3b --input data/prompts.jsonl --output results/output.json
```

For detailed command options:
```bash
aiflux --help
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

AI-Flux provides utility converters to help prepare JSONL files from various input formats:

```bash
# Convert CSV to JSONL
aiflux convert csv --input data/papers.csv --output data/papers.jsonl --template "Summarize: {text}"

# Convert directory to JSONL
aiflux convert dir --input data/documents/ --output data/docs.jsonl --recursive
```

For code examples of converters, see the [examples directory](https://github.com/UIUC-Chatbot/ai-flux/tree/main/examples/).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/UIUC-Chatbot/ai-flux/blob/main/CONTRIBUTING.md) for guidelines.

## License

[MIT License](https://github.com/UIUC-Chatbot/ai-flux/blob/main/LICENSE) 