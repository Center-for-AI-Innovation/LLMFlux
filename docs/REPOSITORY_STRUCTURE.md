# LLMFlux Repository Structure

This document explains the organization of the LLMFlux codebase to help you understand and navigate the project.

## Directory Structure

```
llmflux/
├── src/
│   └── llmflux/                 
│       ├── core/              
│       │   ├── processor.py   # Base processor interface
│       │   ├── config.py      # Configuration management
│       │   ├── config_manager.py # Configuration priority system
│       │   └── client.py      # LLM client interface
│       ├── processors/        # Built-in processors
│       │   └── batch.py       # JSONL batch processor
│       ├── slurm/             # SLURM integration
│       │   ├── runner.py      # SLURM job management
│       │   └── scripts/       # SLURM scripts
│       ├── converters/        # Format converters (utilities)
│       │   ├── csv.py         # CSV to JSONL converter
│       │   ├── json.py        # JSON to JSONL converter
│       │   ├── directory.py   # Directory to JSONL converter
│       │   ├── vision.py      # Vision to JSONL converter
│       │   └── utils.py       # JSONL utilities
│       ├── io/                # Input/Output handling
│       │   ├── base.py        # Base output classes
│       │   └── output/        # Output handlers
│       │       └── json_output.py # JSON output handler
│       ├── templates/         # Model templates
│       │   ├── llama3.2/
│       │   ├── llama3.3/
│       │   └── qwen2.5/
│       └── utils/            
│           └── env.py         # Environment utilities
├── examples/                  # Example implementations
├── tests/                    
└── pyproject.toml
```

## Key Components

### Core Module

The `core` module contains the foundational components of the system:

- `processor.py`: Base class for all processors
- `config.py`: Configuration management and models
- `config_manager.py`: Manages configuration priority
- `client.py`: Interface for communicating with language models

### Processors Module

The `processors` module contains implementations of batch processors:

- `batch.py`: The main JSONL batch processor implementation

### SLURM Module

The `slurm` module handles integration with SLURM for HPC systems:

- `runner.py`: SLURM job submission and management
- `scripts/`: SLURM batch scripts for job execution

### Converters Module

The `converters` module contains utilities for converting data to JSONL format:

- `csv.py`: Convert CSV files to JSONL
- `json.py`: Convert JSON files to JSONL
- `directory.py`: Convert directory contents to JSONL
- `vision.py`: Prepare vision data for JSONL processing
- `utils.py`: Utility functions for JSONL handling

### IO Module

The `io` module handles input and output operations:

- `base.py`: Base classes for input/output handling
- `output/json_output.py`: JSON output formatter

### Templates Module

The `templates` module contains YAML configuration files for supported models:

- Organization is by model family (e.g., `llama3.2/`) then size (e.g., `7b.yaml`)

### Utils Module

The `utils` module contains utility functions used throughout the codebase:

- `env.py`: Environment variable utilities

## Other Directories

- `examples/`: Example scripts demonstrating usage of the library
- `tests/`: Unit and integration tests
- `docs/`: Documentation files
- `data/`: Default directory for input and output data
- `models/`: Default directory for model cache
- `logs/`: Default directory for log files
- `containers/`: Container definitions and scripts

## Important Files

- `pyproject.toml`: Package configuration and dependencies
- `.env.example`: Example environment configuration
- `README.md`: Main project documentation 
