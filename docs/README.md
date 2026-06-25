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
usage: llmflux [-h] [--version] {run,serve,connect,benchmark,show-models,jobs,status,logs,cancel} ...

LLMFlux CLI

positional arguments:
  {run,serve,connect,benchmark,show-models,jobs,status,logs,cancel}
    run                 Submit a batch processing job
    serve               Start a model as a long-running service on a compute node
    connect             Show connection info for a running serve job
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
    model="llama3.2:3b",
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

LLMFlux includes a command-line interface for submitting batch processing jobs. It uses vLLM as it's default engine, and model configurations rely on the HuggingFace naming scheme. To process your prompts.jsonl file using the Ollama engine running the llama3.2 model with 3b parameters, you would run the command:

```bash
# Process JSONL file directly (core functionality)
llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl --output results/output.json
```

In addition to the default vLLM engine, LLMFlux can also be run using Ollama. You then can call using the names as established in the models.yaml file in the templates dir:

# With SLURM account and partition
```bash
llmflux run \
   --account your-account \
   --partition gpu \
   --model Llama-3.2-3B-Instruct \
   --input data/prompts.jsonl \
   --output results/output.json
```

```bash
# Process JSONL file using VLLM backend
llmflux run --model MistralLite --input data/prompts.jsonl --output results/output.json
```

This will run the same as above, using VLLM as the backend interface. If you wanted to run mistral-lite, for example, checking the file mistral-lite/7b.yaml reveals the name: "mistrallite:7b". Update to the appropriate HuggingFace key and run 
```bash
# Process JSONL file using VLLM backend
llmflux run --model MistralLite --input data/prompts.jsonl --output results/output.json
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

LLMFlux provides utility converters to help prepare JSONL files from various input formats:

```bash
# Convert CSV to JSONL
llmflux convert csv --input data/papers.csv --output data/papers.jsonl --template "Summarize: {text}"

# Convert directory to JSONL
llmflux convert dir --input data/documents/ --output data/docs.jsonl --recursive
```

For code examples of converters, see the [examples directory](examples/).

## Interactive Serving

In addition to batch processing, LLMFlux can start a model as a long-running
OpenAI-compatible service on a compute node.

### Start a serve job

```bash
llmflux serve \
    --model Llama-3.2-3B-Instruct \
    --email you@example.com \
    --time 02:00:00 \
    --engine vllm
```

- `--model` — model key from `models.yaml` (same as `llmflux run`)
- `--email` — you will receive an email **when the model is ready** (not just when the job starts)
- `--time` — how long to keep the service alive (e.g. `02:00:00`, `08:00:00`)
- `--engine` — `vllm` (default) or `ollama`

LLMFlux generates a unique API key for the session and prints the job ID:

```
Serve job submitted: 2301062
  Model:  Llama-3.2-3B-Instruct
  Engine: vllm
  Time:   02:00:00
  Email:  you@example.com

You will receive an email at you@example.com when the service is ready.
Then run: llmflux connect 2301062
```

### Connect once the model is ready

After receiving the ready email, run:

```bash
llmflux connect 2301062
```

This pings the endpoint and prints everything you need:

```
Service is ready.

  Endpoint:  http://gpu-node-04:8031/v1
  API Key:   llmflux-57de4141f7d9b52a24481f05438c166c
  Model:     meta-llama/Llama-3.2-3B-Instruct
  Engine:    vllm

Example usage:

from openai import OpenAI
client = OpenAI(base_url="http://gpu-node-04:8031/v1", api_key="llmflux-57de4141f7d9b52a24481f05438c166c")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Check status and shut down

```bash
# See endpoint, API key, and email for a serve job
llmflux status 2301062

# Shut the service down early
llmflux cancel 2301062
```

`llmflux jobs` also shows a `TYPE` column (`serve` / `batch`) so you can
distinguish long-running services from batch jobs at a glance.

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
