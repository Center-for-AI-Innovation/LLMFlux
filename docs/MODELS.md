# LLMFlux Supported Models

This document lists all models supported by LLMFlux, along with their hardware requirements and configuration details.

## Model Naming Convention

Models in LLMFlux follow the naming convention `family:size`, for example:
- `llama3.2:3b` - Llama 3.2 model with 3 billion parameters
- `gemma3:27b` - Gemma 3 model with 27 billion parameters
- `phi3:small` - Phi-3 Small model

## Supported Models

### Llama 3.2

Advanced general-purpose model from Meta.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| 3b | llama3.2/3b.yaml | 8GB | Any CUDA GPU | Best balance of performance/resource usage |
| 7b | llama3.2/7b.yaml | 13GB | A40/A100 | Good for general use |
| 70b | llama3.2/70b.yaml | 35GB | A100 80GB | High-performance, requires high-end GPU |

### Llama 3.2 Vision

Vision-capable variant of Llama 3.2.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| 8b | llama3.2-vision/8b.yaml | 16GB | A40/A100 | Vision capabilities require more memory |
| 70b | llama3.2-vision/70b.yaml | 40GB | A100 80GB | Handles complex images and reasoning |

### Llama 3.3

Latest generation of Llama optimized for reasoning.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| 8b | llama3.3/8b.yaml | 14GB | A40/A100 | Good general-purpose reasoner |
| 70b | llama3.3/70b.yaml | 38GB | A100 80GB | State-of-the-art reasoning capabilities |

### Gemma 3

Google's efficient and high-quality models.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| 1b | gemma3/1b.yaml | 6GB | Any CUDA GPU | Extremely efficient, good for basic tasks |
| 4b | gemma3/4b.yaml | 10GB | Any CUDA GPU | Good performance/resource balance |
| 12b | gemma3/12b.yaml | 16GB | A40/A100 | High quality mid-range option |
| 27b | gemma3/27b.yaml | 24GB | A100 | High performance, vision-capable |

### Qwen 2.5

Production-quality models from Alibaba.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| 7b | qwen2.5/7b.yaml | 15GB | A40/A100 | Default setup, good general model |
| 72b | qwen2.5/72b.yaml | 35GB | A100 80GB | High performance, high resource usage |

### Phi 3

Microsoft's efficient models with strong reasoning capabilities.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| mini | phi3/mini.yaml | 6GB | Any CUDA GPU | Extremely efficient 3.8B model |
| small | phi3/small.yaml | 12GB | Any CUDA GPU | 7B parameters, good performance |
| medium | phi3/medium.yaml | 16GB | A40/A100 | 14B parameters, balanced option |
| vision | phi3/vision.yaml | 18GB | A40/A100 | Vision-capable 14B parameter model |

### Mistral Models

Family of high-quality open source models.

| Model | Size | Config File | Min GPU Memory | Notes |
|-------|------|-------------|----------------|-------|
| Mistral | 7b | mistral/7b.yaml | 13GB | Original Mistral model |
| Mistral | 8x7b | mistral/8x7b.yaml | 36GB | Mixture of experts model |
| Mistral-Small | 7b | mistral-small/7b.yaml | 12GB | Optimized for inference speed |
| Mistral-Large | 33b | mistral-large/33b.yaml | 28GB | Large capacity model |
| Mistral-Lite | 3b | mistral-lite/3b.yaml | 8GB | Small footprint model |
| Mistral-NeMo | 12b | mistral-nemo/12b.yaml | 16GB | NVIDIA optimized model |
| Mistral-OpenOrca | 7b | mistral-openorca/7b.yaml | 14GB | Research tuned version |

### Mixtral

Mixture-of-experts models with strong performance.

| Size | Config File | Min GPU Memory | Recommended | Notes |
|------|-------------|----------------|-------------|-------|
| 8x7b | mixtral/8x7b.yaml | 24GB | A100 | Original MoE model |
| 8x22b | mixtral/8x22b.yaml | 40GB | A100 80GB | Higher parameter version |

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
    model="llama3.2:7b",
    batch_size=2  # Reduced from default 4 to use less memory
)

# Or using environment variables
# In .env file:
# MODEL_NAME=llama3.2:7b
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
    model="llama3.2:7b",
    batch_size=16     # 4x default for much higher throughput
)

# Or using environment variables
# In .env file:
# MODEL_NAME=llama3.2:7b
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
    model="llama3.3:70b",
    batch_size=2               # Even a batch size of 2 is significant for 70B models
)

# Or using environment variables
# In .env file:
# MODEL_NAME=llama3.3:70b
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
    model="llama3.3:70b",
    batch_size=4                # Higher batch size possible with multiple GPUs
)

# Or using environment variables
# In .env file:
# MODEL_NAME=llama3.3:70b
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
name: "custom:my-model"

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
    model_type="custom",
    model_size="my-model",
    custom_config_path="path/to/custom/my-model.yaml"
) 
