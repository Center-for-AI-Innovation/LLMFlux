#!/usr/bin/env python3
"""Command-line interface for AI-Flux.

Provides the `aiflux` executable with subcommands.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List

from .slurm.runner import SlurmRunner
from .processors import BatchProcessor
from .core.config import Config, SlurmConfig
from .benchmark_utils import generate_synthetic_prompts, save_prompts_to_jsonl


def _parse_sbatch_args(sbatch_arg_list: Optional[List[str]]) -> Optional[Dict[str, str]]:
    """Parse --sbatch-arg arguments into a dictionary.
    
    Args:
        sbatch_arg_list: List of "key=value" strings from CLI
        
    Returns:
        Dictionary of extra SBATCH arguments or None
    """
    if not sbatch_arg_list:
        return None
    
    result = {}
    for arg in sbatch_arg_list:
        if '=' not in arg:
            print(f"Warning: Ignoring invalid --sbatch-arg format: {arg} (expected KEY=VALUE)")
            continue
        key, value = arg.split('=', 1)
        result[key.strip()] = value.strip()
    
    return result if result else None


def _benchmark_command(args: argparse.Namespace) -> int:
    """Handle the `benchmark` subcommand.
    
    Args:
        args: Parsed CLI arguments
        
    Returns:
        Process exit code
    """
    # Generate or use provided dataset
    if args.input:
        input_path = Path(args.input)
    else:
        # Generate synthetic prompts
        name = args.name or f"benchmark_{args.model.replace(':', '_')}"
        output_dir = Path("data/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_prompts = getattr(args, "num_prompts", 50)
        prompts = generate_synthetic_prompts(num_prompts=num_prompts, model=args.model)
        
        dataset_path = output_dir / f"{name}_prompts.jsonl"
        save_prompts_to_jsonl(prompts, dataset_path)
        print(f"Generated {num_prompts} prompts: {dataset_path}")
        input_path = dataset_path
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        name = args.name or f"benchmark_{args.model.replace(':', '_')}"
        output_path = f"results/benchmarks/{name}_results.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Collect SLURM config from CLI args (filter out None values)
    config = Config()
    slurm_overrides = {
        key: value for key, value in {
            "account": args.account,
            "partition": args.partition,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "time": args.time,
            "mem": args.mem,
            "cpus_per_task": args.cpus_per_task,
        }.items() if value is not None
    }
    
    # Parse and add extra SBATCH args if provided
    extra_args = _parse_sbatch_args(getattr(args, 'sbatch_arg', None))
    if extra_args:
        slurm_overrides['extra_sbatch_args'] = extra_args
    
    # Merge CLI overrides with config from .env
    slurm_config = config.get_slurm_config(slurm_overrides)
    runner = SlurmRunner(config=slurm_config)
    
    kwargs = {
        "model": args.model,
        "batch_size": getattr(args, "batch_size", 4),
    }
    
    if getattr(args, "temperature", None) is not None:
        kwargs["temperature"] = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        kwargs["max_tokens"] = args.max_tokens
    if getattr(args, "rebuild", False):
        kwargs["rebuild"] = True
    if getattr(args, "debug", False):
        kwargs["debug"] = True
    
    print(f"Submitting benchmark job...")
    print(f"  Model: {args.model}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    job_id = runner.run(input_path=str(input_path), output_path=output_path, **kwargs)
    print(f"Job ID: {job_id}")
    return 0


def _run_command(args: argparse.Namespace) -> int:
    """Handle the `run` subcommand.

    Args:
        args: Parsed CLI arguments

    Returns:
        Process exit code
    """
    input_path = args.input
    output_path = args.output
    model = args.model

    # Basic validation
    if not input_path:
        print("--input is required", file=sys.stderr)
        return 2

    # Ensure output directory exists if provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Local mode: run BatchProcessor directly (no SLURM)
    if getattr(args, "local", False):
        config = Config()
        model_type = model.split(":")[0] if ":" in model else model
        model_size = model.split(":")[1] if ":" in model else "3b"
        model_config = config.load_model_config(model_type, model_size)

        processor = BatchProcessor(
            model_config=model_config,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        # Additional kwargs that map to processor.run or underlying client params
        run_kwargs = {}
        if args.max_tokens is not None:
            run_kwargs["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            run_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            run_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            run_kwargs["top_k"] = args.top_k

        processor.run(input_path=input_path, output_path=output_path or str(Path("results") / "output.json"), **run_kwargs)
        return 0

    config = Config()
    # Collect Slurm config from args
    slurm_config = {
        key: value for key, value in {
            "account": args.account,
            "partition": args.partition,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "time": args.time,
            "mem": args.mem,
            "cpus_per_task": args.cpus_per_task,
        }.items() if value is not None
    }
    
    # Parse and add extra SBATCH args if provided
    extra_args = _parse_sbatch_args(getattr(args, 'sbatch_arg', None))
    if extra_args:
        slurm_config['extra_sbatch_args'] = extra_args
    
    # Update Slurm config with args
    slurm_config = config.get_slurm_config(slurm_config)
    # SLURM mode
    runner = SlurmRunner(config=slurm_config)
    # Collect kwargs accepted by SlurmRunner.run to set env for the job script
    kwargs = {
        "model": model,
        "batch_size": args.batch_size,
        "save_frequency": args.save_frequency,
        "max_retries": args.max_retries,
        "retry_delay": args.retry_delay,
    }
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k

    # Pass rebuild flag through to runner
    if getattr(args, "rebuild", False):
        kwargs["rebuild"] = True
    
    # Pass debug flag through to runner
    if getattr(args, "debug", False):
        kwargs["debug"] = True

    job_id = runner.run(input_path=input_path, output_path=output_path, **kwargs)
    print(job_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiflux", description="AI-Flux CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Submit a batch processing job")
    run_parser.add_argument("--model", required=True, help="Model name, e.g., llama3.2:3b")
    run_parser.add_argument("--input", required=True, help="Path to input JSONL file")
    run_parser.add_argument("--output", required=False, help="Path to output JSON file")

    # Common tuning options
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument("--save-frequency", type=int, default=50)
    run_parser.add_argument("--max-retries", type=int, default=3)
    run_parser.add_argument("--retry-delay", type=float, default=1.0)
    run_parser.add_argument("--max-tokens", type=int)
    run_parser.add_argument("--temperature", type=float)
    run_parser.add_argument("--top-p", type=float)
    run_parser.add_argument("--top-k", type=int)

    # SLURM configuration
    run_parser.add_argument("--account", type=str)
    run_parser.add_argument("--partition", type=str)
    run_parser.add_argument("--nodes", type=int)
    run_parser.add_argument("--gpus-per-node", type=int)
    run_parser.add_argument("--time", type=str)
    run_parser.add_argument("--mem", type=str)
    run_parser.add_argument("--cpus-per-task", type=int)
    run_parser.add_argument(
        "--sbatch-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional SBATCH directive (e.g., --sbatch-arg reservation=my_res). Can be used multiple times."
    )

    # Container rebuild control
    run_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the Apptainer/Singularity image before running",
    )
    
    # Debug mode
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Preserve generated SLURM job script (job.sh) for debugging",
    )


    # Local execution toggle
    # Add support for this in the future - Can be directly used on the compute node
    # run_parser.add_argument("--local", action="store_true", help="Run locally without SLURM")

    # LLM Engine
    # Todo add vllm, ollama is default
    run_parser.add_argument("--engine", type=str)

    run_parser.set_defaults(func=_run_command)
    
    # benchmark subcommand
    benchmark_parser = subparsers.add_parser("benchmark", help="Run a benchmark job")
    benchmark_parser.add_argument("--model", required=True, help="Model name, e.g., llama3.2:3b")
    benchmark_parser.add_argument("--name", type=str, help="Benchmark run name (default: benchmark_{model})")
    benchmark_parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts to generate (default: 50)")
    benchmark_parser.add_argument("--input", type=str, help="Use existing prompts file instead of generating")
    benchmark_parser.add_argument("--output", type=str, help="Path to output JSON file")
    
    # Common tuning options
    benchmark_parser.add_argument("--batch-size", type=int, default=4)
    benchmark_parser.add_argument("--max-tokens", type=int)
    benchmark_parser.add_argument("--temperature", type=float)
    
    # SLURM configuration
    benchmark_parser.add_argument("--account", type=str)
    benchmark_parser.add_argument("--partition", type=str)
    benchmark_parser.add_argument("--nodes", type=int)
    benchmark_parser.add_argument("--gpus-per-node", type=int)
    benchmark_parser.add_argument("--time", type=str)
    benchmark_parser.add_argument("--mem", type=str)
    benchmark_parser.add_argument("--cpus-per-task", type=int)
    benchmark_parser.add_argument(
        "--sbatch-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional SBATCH directive (e.g., --sbatch-arg reservation=my_res). Can be used multiple times."
    )
    
    # Container rebuild control
    benchmark_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the Apptainer/Singularity image before running",
    )
    
    # Debug mode
    benchmark_parser.add_argument(
        "--debug",
        action="store_true",
        help="Preserve generated SLURM job script (job.sh) for debugging",
    )
    
    benchmark_parser.set_defaults(func=_benchmark_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

