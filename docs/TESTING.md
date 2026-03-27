# Testing Guide for AI-Flux

This guide explains how to run the test suite for AI-Flux.

## Quick Start

### Install Test Dependencies

```bash
# Install package with test dependencies
pip install -e ".[test]"

# Or install individually
pip install pytest pytest-cov pytest-mock pytest-xdist
```

### Run All Tests

#### With Python (Recommended)

Using `python -m pytest` is recommended because it:
- Uses the correct Python interpreter and environment
- Works even if `pytest` is not in your PATH
- Ensures you're using the Python where dependencies are installed
- Works better in virtual environments and on HPC clusters

```bash
# Basic test run
python -m pytest

# With verbose output
python -m pytest -v

# With coverage report
python -m pytest --cov=aiflux --cov-report=html --cov-report=term-missing

# Using specific Python version
python3.11 -m pytest -v
python3.12 -m pytest -v

# Verify which Python is being used
which python
python --version
```

#### Direct pytest (Alternative)

You can also use `pytest` directly if it's installed and in your PATH:

```bash
# Basic test run
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=llmflux --cov-report=html --cov-report=term-missing
```

## Test Classification

The test suite consists of:
- **Unit Tests** (16 tests): Test individual functions in isolation
- **Integration Tests** (9 tests): Test component interactions with mocked external dependencies

All tests can run **locally** without requiring a SLURM cluster.

## Running Tests

### Running Tests with Python

You can run tests using Python directly with `python -m pytest`. This is useful when:
- `pytest` is not in your PATH
- You want to use a specific Python interpreter
- Running on systems where pytest executable is not available
- You want to ensure you're using the correct Python environment

#### Basic Usage with Python

```bash
# Using default Python
python -m pytest

# Using Python 3 explicitly
python3 -m pytest

# Using specific Python version
python3.11 -m pytest
python3.12 -m pytest

# With verbose output
python -m pytest -v

# Extra verbose (show print statements)
python -m pytest -vv -s
```

#### Run All Tests with Python

```bash
# Simple run
python -m pytest

# Verbose output
python -m pytest -v

# Extra verbose (show print statements)
python -m pytest -vv -s

# With coverage
python -m pytest --cov=llmflux --cov-report=term-missing
```

### Run All Tests (Direct pytest)

You can also run pytest directly if it's installed and in your PATH:

```bash
# Simple run
pytest

# Verbose output
pytest -v

# Extra verbose (show print statements)
pytest -vv -s
```

### Run Specific Test Files

#### With Python

```bash
# Run only CLI tests
python -m pytest tests/test_cli.py -v

# Run only SLURM runner tests
python -m pytest tests/slurm/test_runner.py -v

# Run only converter tests
python -m pytest tests/converters/ -v

# Run only processor tests
python -m pytest tests/processors/ -v

# Using specific Python version
python3.11 -m pytest tests/test_cli.py -v
python3.12 -m pytest tests/test_cli.py -v
```

#### Direct pytest

```bash
# Run only CLI tests
pytest tests/test_cli.py -v

# Run only SLURM runner tests
pytest tests/slurm/test_runner.py -v

# Run only converter tests
pytest tests/converters/ -v

# Run only processor tests
pytest tests/processors/ -v
```

### Run Specific Test Classes

#### With Python

```bash
# Run specific test class
python -m pytest tests/test_cli.py::TestRunCommand -v

# Run multiple test classes
python -m pytest tests/test_cli.py::TestRunCommand tests/test_cli.py::TestBenchmarkCommand -v

# Using specific Python version
python3.11 -m pytest tests/test_cli.py::TestRunCommand -v
```

#### Direct pytest

```bash
# Run specific test class
pytest tests/test_cli.py::TestRunCommand -v

# Run multiple test classes
pytest tests/test_cli.py::TestRunCommand tests/test_cli.py::TestBenchmarkCommand -v
```

### Run Specific Test Functions

#### With Python

```bash
# Run specific test function
python -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -v

# Run multiple test functions
python -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic tests/test_cli.py::TestRunCommand::test_run_command_with_slurm_config -v

# Using specific Python version
python3.11 -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -v
```

#### Direct pytest

```bash
# Run specific test function
pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -v

# Run multiple test functions
pytest tests/test_cli.py::TestRunCommand::test_run_command_basic tests/test_cli.py::TestRunCommand::test_run_command_with_slurm_config -v
```

### Run Tests Matching a Pattern

#### With Python

```bash
# Run tests matching "run_command" in name
python -m pytest -k "run_command" -v

# Run tests matching "benchmark"
python -m pytest -k "benchmark" -v

# Run tests matching "environment"
python -m pytest -k "environment" -v

# Run tests NOT matching a pattern
python -m pytest -k "not integration" -v

# Using specific Python version
python3.11 -m pytest -k "run_command" -v
```

#### Direct pytest

```bash
# Run tests matching "run_command" in name
pytest -k "run_command" -v

# Run tests matching "benchmark"
pytest -k "benchmark" -v

# Run tests matching "environment"
pytest -k "environment" -v

# Run tests NOT matching a pattern
pytest -k "not integration" -v
```

### Run Tests in Parallel (Faster)

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run with automatic worker count
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

## Test Coverage

### Generate Coverage Report

```bash
# Terminal output
pytest --cov=aiflux --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=llmflux --cov-report=html
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Coverage with Specific Files

```bash
# Coverage for specific module
pytest --cov=llmflux.cli --cov-report=term-missing

# Coverage for multiple modules
pytest --cov=llmflux.cli --cov=llmflux.slurm --cov-report=term-missing
```

## Running Tests on Server/HPC Cluster

### Basic Setup

```bash
# SSH into server
ssh username@server-hostname

# Navigate to project
cd ~/projects/ai-flux

# Option 1: Load conda module (most common)
module load anaconda3
# OR try these alternatives if anaconda3 is not available:
# module load anaconda
# module load conda
# module load python/3.11
# module load python/3.12

# Check available modules
module avail python
module avail conda
module avail anaconda

# Activate environment
conda activate llmflux

# Install test dependencies if not already installed
pip install -e ".[test]"
```

### Alternatives if anaconda3 Module is Not Found

If `module load anaconda3` fails, try these alternatives:

#### Option 1: Try Different Module Names
```bash
# Try different module names
module load anaconda
module load conda
module load python/3.11
module load python/3.12
module load python

# Check what's available
module avail python
module avail conda
module avail | grep -i python
module avail | grep -i conda
```

#### Option 2: Use System Python with venv
```bash
# Check if Python 3.11+ is available
python3 --version
python3.11 --version
python3.12 --version

# Create virtual environment (if conda not available)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[test]"
```

#### Option 3: Use Miniconda/Anaconda Installation
```bash
# If conda is installed but not as module
# Find conda installation
which conda
# or
~/.conda/bin/conda

# Initialize conda for shell
source ~/.conda/etc/profile.d/conda.sh
# or
eval "$(conda shell.bash hook)"

# Then create and activate environment
conda create -n llmflux python=3.11 -y
conda activate llmflux
pip install -e ".[test]"
```

### Run Tests on Server

#### With Python (Recommended)

```bash
# One-line command (with anaconda3 module)
module load anaconda3 && conda activate llmflux && python -m pytest tests/test_cli.py -v

# Or step by step
module load anaconda3  # or alternative module name
conda activate llmflux
python -m pytest tests/test_cli.py -v

# Run with coverage
python -m pytest --cov=llmflux --cov-report=term-missing

# If using venv instead of conda
source venv/bin/activate
python -m pytest tests/test_cli.py -v

# Using specific Python version
python3.11 -m pytest tests/test_cli.py -v
python3.12 -m pytest tests/test_cli.py -v
```

#### Direct pytest

```bash
# One-line command (with anaconda3 module)
module load anaconda3 && conda activate llmflux && pytest tests/test_cli.py -v

# Or step by step
module load anaconda3  # or alternative module name
conda activate llmflux
pytest tests/test_cli.py -v

# Run with coverage
pytest --cov=llmflux --cov-report=term-missing

# If using venv instead of conda
source venv/bin/activate
pytest tests/test_cli.py -v
```

### Using Different Python Versions

When using `python -m pytest`, you can specify the Python interpreter:

```bash
# With Python 3.11
python3.11 -m pytest tests/test_cli.py -v

# With Python 3.12
python3.12 -m pytest tests/test_cli.py -v

# Using system Python 3
python3 -m pytest tests/test_cli.py -v

# Load specific Python module (if available)
module load python/3.11
python -m pytest tests/test_cli.py -v

# Verify which Python is being used
which python
python --version

# Run tests with specific Python
/usr/bin/python3.11 -m pytest tests/test_cli.py -v
```

## Test Categories

### Unit Tests Only

```bash
# Run only unit tests (parser, commands, main)
pytest tests/test_cli.py::TestCLIParser \
       tests/test_cli.py::TestRunCommand \
       tests/test_cli.py::TestBenchmarkCommand \
       tests/test_cli.py::TestMainFunction \
       -v
```

### Integration Tests Only

```bash
# Run only integration tests
pytest tests/test_cli.py::TestRunnerEnvironmentVariables \
       tests/test_cli.py::TestEnvironmentVariablePrefixes \
       tests/test_cli.py::TestCommandLineIntegration \
       -v
```

## Debugging Tests

### Run with Print Statements

#### With Python

```bash
# Show print statements (use -s flag)
python -m pytest -s

# Show print statements with verbose
python -m pytest -v -s

# Show print for specific test
python -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -s

# Using specific Python version
python3.11 -m pytest -v -s
```

#### Direct pytest

```bash
# Show print statements (use -s flag)
pytest -s

# Show print statements with verbose
pytest -v -s

# Show print for specific test
pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -s
```

### Stop on First Failure

#### With Python

```bash
# Stop on first failure
python -m pytest -x

# Stop after N failures
python -m pytest --maxfail=3

# With verbose
python -m pytest -x -v
```

#### Direct pytest

```bash
# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3
```

### Show Local Variables on Failure

#### With Python

```bash
# Show local variables in traceback
python -m pytest -l

# Show locals with verbose
python -m pytest -v -l
```

#### Direct pytest

```bash
# Show local variables in traceback
pytest -l

# Show locals with verbose
pytest -v -l
```

### Run Last Failed Tests

#### With Python

```bash
# Run only tests that failed last time
python -m pytest --lf

# Run failed tests first, then others
python -m pytest --ff

# With verbose
python -m pytest --lf -v
```

#### Direct pytest

```bash
# Run only tests that failed last time
pytest --lf

# Run failed tests first, then others
pytest --ff
```

### Debug Specific Test

#### With Python

```bash
# Run with pdb debugger on failure
python -m pytest --pdb

# Drop into debugger immediately
python -m pytest --trace

# Debug specific test
python -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic --pdb
```

#### Direct pytest

```bash
# Run with pdb debugger on failure
pytest --pdb

# Drop into debugger immediately
pytest --trace
```

## Test Output Options

### Minimal Output

```bash
# Quiet mode
pytest -q

# Show only failures
pytest -q --tb=short
```

### Detailed Output

```bash
# Verbose with detailed traceback
pytest -v --tb=long

# Show all assertions, not just failures
pytest -v -l

# Show extra summary
pytest -v --tb=short -ra
```

## Common Test Commands

### Most Common

```bash
# Run all tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_cli.py -v

# Run with coverage
pytest --cov=llmflux --cov-report=term

# Run tests matching pattern
pytest -k "test_run" -v
```

### For Development

```bash
# Run tests and stop on first failure
pytest -x -v

# Run only failed tests from last run
pytest --lf -v

# Run with print statements
pytest -s -v

# Run in parallel for speed
pytest -n auto -v
```

## Test File Structure

```
tests/
├── __init__.py
├── test_cli.py              # CLI command tests (25 tests)
├── converters/
│   ├── test_csv.py
│   ├── test_json.py
│   ├── test_directory.py
│   └── test_utils.py
├── processors/
│   └── test_batch.py
└── slurm/
    └── test_runner.py
```

## Test Requirements

### Dependencies

Test dependencies are defined in `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-xdist>=3.0.0",
]
```

### Installation

```bash
# Install with test dependencies
pip install -e ".[test]"

# Or manually
pip install pytest pytest-cov pytest-mock pytest-xdist
```

## Troubleshooting

### Import Errors

```bash
# If tests can't find llmflux module
# Make sure package is installed in editable mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m pytest

# Using specific Python with PYTHONPATH
PYTHONPATH="${PYTHONPATH}:$(pwd)/src" python3.11 -m pytest -v
```

### Module Not Found

```bash
# Verify installation
pip list | grep llmflux

# Reinstall if needed
pip install -e ".[test]"
```

### Test Failures

```bash
# Run with more details
python -m pytest -vv --tb=long

# Run with print statements
python -m pytest -s -v

# Check if it's a specific test
python -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -vv

# Using specific Python version
python3.11 -m pytest -vv --tb=long
```

### Permission Errors

```bash
# If tests can't create temp files
# Check directory permissions
ls -la tests/

# Run with verbose to see what's happening
pytest -vv -s
```

## Continuous Integration

Tests run automatically on:
- Push to main/develop/master branches
- Pull requests
- GitHub Actions CI workflow

See `.github/workflows/ci.yml` for CI configuration.

## Best Practices

1. **Run tests before committing**:
   ```bash
   pytest -v
   ```

2. **Run specific tests you're working on**:
   ```bash
   pytest -k "your_test_pattern" -v
   ```

3. **Check coverage**:
   ```bash
   pytest --cov=llmflux --cov-report=term-missing
   ```

4. **Run tests in parallel for speed**:
   ```bash
   pytest -n auto
   ```

5. **Stop on first failure during development**:
   ```bash
   pytest -x -v
   ```

## Running AI-Flux Commands on Server

### Specify Account in Command

The account can be specified in multiple ways:

#### Method 1: CLI Argument (Recommended)
```bash
llmflux run \
  --account your-account-name \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json

# With partition
llmflux run \
  --account your-account-name \
  --partition gpu \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json
```

#### Method 2: Environment Variable
```bash
# Set for current session
export SLURM_ACCOUNT=your-account-name

# Then run (account will be picked up automatically)
llmflux run \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json
```

#### Method 3: .env File
```bash
# Edit .env file
nano .env

# Add:
SLURM_ACCOUNT=your-account-name

# Save and run
llmflux run --model llama3.2:3b --input data/prompts.jsonl --output results.json
```

### Find Your Account Name

```bash
# Check your SLURM accounts
sacctmgr show user $USER withassoc

# Or check account info
scontrol show user $USER

# Example output shows account names like:
# Account=my-account-name
```

### Complete Example with Account and Partition

```bash
# Check available partitions first
sinfo | grep gpu

# Run with account and correct partition
llmflux run \
  --account your-account-name \
  --partition gpu \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json \
  --time 02:00:00 \
  --mem 64G \
  --gpus-per-node 1
```

### Module Loading Alternatives for AI-Flux

If `anaconda3` module is not found when running AI-Flux:

```bash
# Try alternatives (in order):
module load anaconda
module load conda
module load python/3.11
module load python/3.12
module load python

# Check what's available
module avail | grep -i python
module avail | grep -i conda

# If no module available, use system Python
python3 --version  # Should be 3.11+
python3 -m venv venv
source venv/bin/activate
pip install -e ".[test]"
```

## Quick Reference

### Test Commands with Python (Recommended)

```bash
# Run all tests
python -m pytest

# Run with verbose
python -m pytest -v

# Run specific file
python -m pytest tests/test_cli.py -v

# Run specific test
python -m pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -v

# Run with coverage
python -m pytest --cov=llmflux --cov-report=term

# Run in parallel
python -m pytest -n auto

# Run only failed tests
python -m pytest --lf

# Run with print statements
python -m pytest -s

# Stop on first failure
python -m pytest -x

# Using specific Python version
python3.11 -m pytest -v
python3.12 -m pytest -v

# Run on server with module load
module load anaconda3 && conda activate llmflux && python -m pytest -v
```

### Test Commands (Direct pytest)

```bash
# Run all tests
pytest

# Run with verbose
pytest -v

# Run specific file
pytest tests/test_cli.py -v

# Run specific test
pytest tests/test_cli.py::TestRunCommand::test_run_command_basic -v

# Run with coverage
pytest --cov=llmflux --cov-report=term

# Run in parallel
pytest -n auto

# Run only failed tests
pytest --lf

# Run with print statements
pytest -s

# Stop on first failure
pytest -x

# Run on server with module load
module load anaconda3 && conda activate llmflux && pytest -v
```

### AI-Flux Commands

```bash
# Basic command with account
llmflux run \
  --account your-account \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json

# With partition (if needed)
llmflux run \
  --account your-account \
  --partition gpu \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json

# Benchmark with account
llmflux benchmark \
  --account your-account \
  --model llama3.2:3b

# On server with module load
module load anaconda3 && conda activate llmflux && \
llmflux run \
  --account your-account \
  --model llama3.2:3b \
  --input data/prompts.jsonl \
  --output results.json
```
