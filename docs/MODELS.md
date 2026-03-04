# LLMFlux Supported Models

This document lists all models supported by LLMFlux, along with their hardware requirements and configuration details.

## Listing Available Models

To see all available models at any time, run:

```bash
llmflux show-models
```

This will print every model key along with which engines (`ollama`, `vllm`, or both) it supports.

## Model Naming Convention

Models in LLMFlux use the HuggingFace name, for example:
- `Llama-3.2-3B-Instruct` - Llama 3.2 model with 3 billion parameters
- `gemma-3-27b-it` - Gemma 3 model with 27 billion parameters
- `Qwen2.5-32B-Instruct` - Qwen 2.5 model with 32 billion parameters

## Supported Models

### Llama 3.2

Advanced general-purpose model from Meta.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 1b | 8GB | Any CUDA GPU | Lightweight, good for basic tasks |
| 3b | 16GB | A40/A100 | Best balance of performance/resource usage |

### Llama 3.2 Vision

Vision-capable variant of Llama 3.2.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 11b | 24GB | A100 | Vision capabilities require more memory |
| 90b | 40GB | A100 80GB | Handles complex images and reasoning |

### Llama 3.3

Latest generation of Llama optimized for reasoning.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 70b | 80GB | A100 80GB | State-of-the-art reasoning capabilities |

### Gemma 3

Google's efficient and high-quality models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 1b | 2GB | Any CUDA GPU | Extremely efficient, good for basic tasks |
| 4b | 8GB | Any CUDA GPU | Good performance/resource balance |
| 12b | 16GB | A40/A100 | High quality mid-range option |
| 27b | 24GB | A100 | High performance, vision-capable |

### Qwen 2.5

Production-quality models from Alibaba.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 7b | 16GB | A40/A100 | Default setup, good general model |
| 72b | 40GB | A100 80GB | High performance, high resource usage |

### Phi 3

Microsoft's efficient models with strong reasoning capabilities.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| mini | 8GB | Any CUDA GPU | Extremely efficient 3.8B model |
| small | 12GB | Any CUDA GPU | 7B parameters, good performance |
| medium | 24GB | A100 | 14B parameters, balanced option |
| vision | 32GB | A100 | Vision-capable 14B parameter model |

### Mistral Models

Family of high-quality open source models.

| Model | Size | Min GPU Memory | Notes |
|-------|------|----------------|-------|
| Mistral | 7b | 16GB | Original Mistral model |
| Mistral-Small | 22b | 16GB | Optimized for inference speed |
| Mistral-Small | 24b | 16GB | Latest small model |
| Mistral-Large | 123b | 80GB | Large capacity model |
| Mistral-Lite | 7b | 16GB | Small footprint model |
| Mistral-NeMo | 12b | 16GB | NVIDIA optimized model |
| Mistral-OpenOrca | 7b | 16GB | Research tuned version |

### Mixtral

Mixture-of-experts models with strong performance.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 8x7b | 24GB | A100 | Original MoE model |
| 8x22b | 48GB | A100 80GB | Higher parameter version |

### Gemma 2

Google's second generation efficient models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 2b | 2GB | Any CUDA GPU | Lightweight model |
| 9b | 24GB | A100 | Mid-range option |
| 27b | 24GB | A100 | High performance |

### Llama 2

Meta's second generation LLM.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 7b | 16GB | A40/A100 | Entry-level Llama 2 |
| 13b | 16GB | A40/A100 | Good general use |
| 70b | 40GB | A100 80GB | High performance |

### Llama 3

Meta's third generation LLM.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 8b | 16GB | A40/A100 | Good general-purpose model |
| 70b | 40GB | A100 80GB | High performance |

### Llama 3.1

Meta's Llama 3.1 family with extended context.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 8b | 16GB | A40/A100 | Good general-purpose model |
| 70b | 40GB | A100 80GB | High performance |
| 405b | 80GB | Multi-GPU A100 | Largest open model |

### Llama 4

Meta's latest generation model.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 17b-128e | 24GB | A100 | Maverick mixture-of-experts |

### Qwen 2.5 Coder

Alibaba's code-specialized models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 3b | 8GB | Any CUDA GPU | Lightweight coder |
| 7b | 16GB | A40/A100 | Good code generation |

### Qwen 2.5 Math

Alibaba's math-specialized models (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 1.5b | 2GB | Any CUDA GPU | Lightweight math model |
| 7b | 8GB | Any CUDA GPU | Good math reasoning |
| 72b | 40GB | A100 80GB | High performance math |

### Qwen 2.5 Vision

Alibaba's vision-language model (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 7b | 8GB | Any CUDA GPU | Vision-language model |

### Qwen 3

Alibaba's latest generation models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 0.6b | 2GB | Any CUDA GPU | Ultra-lightweight |
| 8b | 24GB | A100 | Good general model |
| 14b | 16GB | A40/A100 | Mid-range option |
| 32b | 24GB | A100 | High performance |

### QwQ

Alibaba's reasoning model (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 32b | 16GB | A40/A100 | Strong reasoning capabilities |

### DeepSeek R1

DeepSeek's reasoning-focused distilled models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 1.5b (Qwen) | 2GB | Any CUDA GPU | Ultra-lightweight reasoning |
| 7b (Qwen) | 16GB | A40/A100 | Good reasoning model |
| 8b (Llama) | 16GB | A40/A100 | Llama-based distillation |
| 14b (Qwen) | 16GB | A40/A100 | Mid-range reasoning |
| 32b (Qwen) | 24GB | A100 | High quality reasoning |
| 70b (Llama) | 40GB | A100 80GB | Highest quality distillation |

### DeepSeek Vision

DeepSeek's vision-language models (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| vl2-small | 8GB | Any CUDA GPU | Smaller vision model |
| vl2 | 24GB | A100 | Full vision-language model |

### CodeLlama

Meta's code-specialized Llama models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 7b | 16GB | A40/A100 | Lightweight code model |
| 13b | 16GB | A40/A100 | Good code generation |
| 34b | 24GB | A100 | Strong code capabilities |
| 70b | 40GB | A100 80GB | Highest quality code model |

### LLaVA

Vision-language models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 7b (v1.6-mistral) | 16GB | A40/A100 | Mistral-based vision |
| 13b (v1.5) | 16GB | A40/A100 | Original LLaVA |
| 34b (v1.6) | 24GB | A100 | Large vision model |

### InternVL 2.5

OpenGVLab's vision-language models (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 8b | 16GB | A40/A100 | Lightweight vision model |
| 26b | 24GB | A100 | Mid-range vision model |
| 38b | 24GB | A100 | Large vision model |

### Phi 3.5

Microsoft's updated Phi model (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| vision | 8GB | Any CUDA GPU | Vision-capable model |

### Cohere Command R

Cohere's instruction-following models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 35b (command-r) | 24GB | A100 | Strong instruction following |
| 104b (command-r-plus) | 80GB | A100 80GB | Highest quality |
| 32b (aya-expanse) | 24GB | A100 | Multilingual model |

### MedGemma

Google's medical domain models (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 4b | 8GB | Any CUDA GPU | Lightweight medical model |
| 27b | 24GB | A100 | Full medical model |

### Kimi

Moonshot's large-scale models.

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| K2 (1T) | 24GB | A100 | Trillion-parameter cloud model |
| K2.5 | 80GB | A100 80GB | Latest generation |

### Pixtral

Mistral AI's vision model (vLLM only).

| Size | Min GPU Memory | Recommended | Notes |
|------|----------------|-------------|-------|
| 12b | 16GB | A40/A100 | Vision-capable Mistral model |

### Other Models

| Model | Min GPU Memory | Recommended | Notes |
|-------|----------------|-------------|-------|
| GLM-4-9B | 16GB | A40/A100 | ZAI chatbot model |
| GPT-OSS-120B | 80GB | A100 80GB | OpenAI open-source model |
| Molmo-7B (vLLM only) | 16GB | A40/A100 | Allen AI vision model |
| all-MiniLM-L6-v2 | 2GB | Any CUDA GPU | Sentence embedding model |
| bge-base-en-v1.5 (vLLM only) | 2GB | Any CUDA GPU | Embedding model |
| whisper-large-v3 (vLLM only) | 24GB | A100 | Speech-to-text model |

## GPU Memory Requirements

Each model specifies a minimum GPU memory requirement based on practical use cases. For optimal performance:

- **A100 (80GB)** can run all models, including the largest 70B+ models
- **A100 (40GB)** can run models up to approximately 40B parameters
- **A40 (48GB)** can run most models except the largest 70B+ models
- **Mid-range GPUs (24GB)** can run models up to approximately 27B parameters
- **Consumer GPUs (8-16GB)** can run smaller models like Phi-3 Mini and Mistral-Lite

## Batch Size and Memory Trade-offs

For each model, you can adjust the batch size based on your memory constraints. Higher batch sizes require more memory but enable much faster throughput, while lower batch sizes use less memory but process requests more slowly.

### Memory Requirement Calculation

As a rule of thumb, for every increase of 1 in batch size, you'll need approximately:
- **Small models (1-7B)**: +3-4GB memory
- **Medium models (7-20B)**: +4-8GB memory
- **Large models (20B+)**: +8-16GB memory

### Example: Limited Memory Configuration

When using a smaller GPU or a large model:

```python
# Using code arguments
runner.run(
    input_path="prompts.jsonl",
    output_path="results.json",
    model="Llama-3.2-3B-Instruct",
    batch_size=2  # Reduced from default 4 to use less memory
)

# Or using environment variables
# In .env file:
# MODEL_NAME=Llama-3.2-3B-Instruct
# BATCH_SIZE=2
```

### Example: High Memory Configuration

When using a high-end GPU like A100 (80GB), you can significantly increase batch size for better throughput:

```python
# Using code arguments
# First, configure SLURM to use the right resources
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = "myaccount"
slurm_config.partition = "a100"  # A100 partition
slurm_config.mem = "80G"         # Request full node memory
slurm_config.gpus_per_node = 1   # 1 GPU (A100 80GB)

# Then run with high batch size
runner = SlurmRunner(config=slurm_config)
job_id = runner.run(
    input_path="prompts.jsonl",
    output_path="results.json",
    model="Llama-3.2-3B-Instruct",
    batch_size=16     # 4x default for much higher throughput
)

# Or using environment variables
# In .env file:
# MODEL_NAME=Llama-3.2-3B-Instruct
# BATCH_SIZE=16
# SLURM_MEM=80G
# SLURM_PARTITION=a100
```

### Batch Size Recommendations

| Model Size | GPU Type | Recommended Batch Size | SLURM Memory Setting |
|------------|----------|------------------------|----------------------|
| 3-7B | A100 (40/80GB) | 16-24 | 40G-80G |
| 3-7B | A40 (48GB) | 12-16 | 48G |
| 3-7B | Mid-range (24GB) | 6-8 | 24G |
| 7-20B | A100 (80GB) | 8-12 | 80G |
| 7-20B | A100 (40GB) | 4-6 | 40G |
| 20-40B | A100 (80GB) | 4-6 | 80G |
| 40-70B | A100 (80GB) | 2-3 | 80G |

### Large Model Configuration on Single GPU

When running large models (70B+) on a single A100 (80GB) GPU:

```python
# Using code arguments
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = "myaccount"
slurm_config.partition = "a100"   # A100 partition
slurm_config.mem = "80G"          # Request full node memory
slurm_config.gpus_per_node = 1    # 1 A100 80GB GPU
slurm_config.cpus_per_task = 16   # Increase CPU cores for preprocessing

runner = SlurmRunner(config=slurm_config)
job_id = runner.run(
    input_path="prompts.jsonl",
    output_path="results.json",
    model="Llama-3.3-70B-Instruct",
    batch_size=2               # Even a batch size of 2 is significant for 70B models
)

# Or using environment variables
# In .env file:
# MODEL_NAME=Llama-3.3-70B-Instruct
# BATCH_SIZE=2
# SLURM_MEM=80G
# SLURM_CPUS_PER_TASK=16
# SLURM_PARTITION=a100
```

### Multi-GPU Configuration for Increased Throughput

For large models with higher throughput, you can leverage multiple GPUs:

```python
# Multi-GPU setup for maximum performance
config = Config()
slurm_config = config.get_slurm_config()
slurm_config.account = "myaccount"
slurm_config.partition = "a100"    # A100 partition
slurm_config.mem = "160G"          # Request memory for multiple GPUs
slurm_config.gpus_per_node = 2     # Request 2 A100 GPUs
slurm_config.cpus_per_task = 32    # Increase CPU cores for multi-GPU processing

runner = SlurmRunner(config=slurm_config)
job_id = runner.run(
    input_path="prompts.jsonl",
    output_path="results.jsonl",
    model="Llama-3.3-70B-Instruct",
    batch_size=4                # Higher batch size possible with multiple GPUs
)

# Or using environment variables
# In .env file:
# MODEL_NAME=Llama-3.3-70B-Instruct
# BATCH_SIZE=4
# SLURM_MEM=160G
# SLURM_CPUS_PER_TASK=32
# SLURM_PARTITION=a100
# SLURM_GPUS_PER_NODE=2
```

## Custom Model Configuration

You can customize model parameters by creating your own YAML configuration files:

```yaml
# custom/my-model.yaml
name: "my-custom-model"

resources:
  gpu_layers: 32
  gpu_memory: "16GB"
  batch_size: 8
  max_concurrent: 2

parameters:
  temperature: 0.5
  top_p: 0.9
  max_tokens: 4096
```

Then load it in your code:
```python
from llmflux.core.config import Config

config = Config()
model_config = config.load_model_config(
    model_key="my-custom-model",
    custom_config_path="path/to/custom/my-model.yaml"
) 
