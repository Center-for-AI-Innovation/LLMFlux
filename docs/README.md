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

**Prerequisites:** LLMFlux runs models inside [Apptainer](https://apptainer.org/) (formerly Singularity) containers on HPC systems. Apptainer must be available on your cluster — confirm with `apptainer --version` or contact your cluster admin.

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

LLMFlux uses vLLM as its default engine. Model keys come from `models.yaml` (see `llmflux show-models`). `--output` is optional — if omitted, results are written to a timestamped file in your configured output directory.

```bash
# Minimal: engine defaults to vllm, output path is auto-generated
llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl

# Explicit engine and output path
llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl --output results/output.json --engine vllm

# Use the Ollama engine instead
llmflux run --model Llama-3.2-3B-Instruct --input data/prompts.jsonl --engine ollama

# With SLURM account and partition
llmflux run \
   --account your-account \
   --partition gpu \
   --model Llama-3.2-3B-Instruct \
   --input data/prompts.jsonl \
   --output results/output.json
```

The `--model` value must match a key from `models.yaml` (e.g. `MistralLite`, `Llama-3.2-3B-Instruct`). LLMFlux resolves the correct HuggingFace repo name (`hf_name`) and Ollama tag (`name`) automatically from that key.

Note that in order to use some HuggingFace models, you will need a token from HF. Once you have a token, update your
local copy of the `.env` file:

```bash
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXX
```

For some gated repos, you will also need to visit the HuggingFace repository directly and accept the terms of access. The model cache defaults to `~/.cache/huggingface/hub`. To change this, add to your `.env`:

```bash
HF_HOME=/path/to/dir
```

LLMFlux will automatically download the appropriate models for both Ollama and vLLM.

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

LLMFlux provides utility converters to help prepare JSONL files from various input formats.
These are Python functions in `llmflux.converters` — there is no `llmflux convert` CLI
command. Call them from a script, then pass the resulting JSONL file to `llmflux run`.

**Supported source types:**
- `csv_to_jsonl` — CSV files; use `prompt_template` with `{column_name}` placeholders
- `directory_to_jsonl` — directories of text files (`.txt`, `.md`, `.json`, `.py`, `.yaml`, `.yml`); use `recursive=True` to traverse subdirectories
- `json_to_jsonl` — existing JSON files already in batch or messages format

**`prompt_template` syntax:** Wrap column or field names in `{braces}` to interpolate values into the prompt. For example, `"Summarize the following paper: {abstract}"` pulls the `abstract` column from each CSV row.

```python
from llmflux.converters import csv_to_jsonl, directory_to_jsonl, json_to_jsonl

# Convert CSV to JSONL using a prompt template
csv_to_jsonl(
    input_path="data/papers.csv",
    output_path="data/papers.jsonl",
    prompt_template="Summarize: {text}",
)

# Convert a directory of text files to JSONL
directory_to_jsonl(
    input_path="data/documents/",
    output_path="data/docs.jsonl",
    recursive=True,
)

# Convert an existing JSON file (batch or messages format) to JSONL
json_to_jsonl(
    input_path="data/prompts.json",
    output_path="data/prompts.jsonl",
)
```

See `examples/csv_jsonl_example.py`, `examples/directory_jsonl_example.py`, and
`examples/json_jsonl_example.py` for complete, runnable versions of the above.

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

### Test how many concurrent users an endpoint can take

`llmflux loadtest` ramps simulated users against a serve job and reports where
latency stops being interactive. Unlike `llmflux benchmark`, which submits its
own job and sends one request at a time, this drives many requests concurrently
against an endpoint that already exists — which is what exercises vLLM's
continuous batching.

```bash
llmflux loadtest --job-id 2301062 \
    --levels 1,2,4,8,16,32 \
    --phase-seconds 90 \
    --image sample.jpg \
    --slo-ms 20000
```

`--job-id` reads the endpoint, key and model from the serve job's connection
file (owner only); otherwise pass `--endpoint`, `--api-key` and `--model`.

```
 conc   done   err    req/s     tok/s   ttft p95    e2e p95   running   waiting   kv max%
    1     18     0    2.179      43.6       53.9      460.6       1.0       0.0      25.0
    4     72     0    8.683     173.7       56.2      463.0      3.88       0.0     100.0
   16     84     0    8.751     175.0     1419.7     1827.3       4.0      12.0     100.0  <-- over SLO

First level over the 20000ms p95 budget: 16 concurrent users.
Peak output throughput: 175.0 tok/s, first reached at concurrency 4.
```

Read it as: throughput plateaus at the knee, and past that point extra users
only add latency and queue depth (`waiting`). Useful options:

- `--think-time 20` — model users who pause between prompts, instead of the
  default worst case where every user always has a request in flight
- `--image` — repeat it to test multi-image prompts, which are prefill-heavy
  and saturate a GPU much earlier than text
- `--output ramp.json` — keep the raw per-level rows

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
