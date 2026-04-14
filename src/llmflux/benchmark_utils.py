#!/usr/bin/env python3
"""Benchmark utilities for generating test datasets and computing metrics."""

import json
import random
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


BENCHMARK_DATA_DIR = Path(__file__).resolve().parent / "benchmark_data"


def ensure_benchmark_data_dir() -> Path:
    """Create the benchmark data directory if it does not exist."""
    BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return BENCHMARK_DATA_DIR


def generate_synthetic_prompts(
    num_prompts: int = 50,
    seed: int = 42,
    model: str = "llama3.2:3b"
) -> List[Dict[str, Any]]:
    """Generate simple synthetic benchmark prompts.
    
    Args:
        num_prompts: Number of prompts to generate
        seed: Random seed for reproducibility
        model: Model name to use in prompts
        
    Returns:
        List of JSONL-compatible prompt dictionaries
    """
    random.seed(seed)
    
    # Simple prompt templates
    templates = [
        "What is {}?",
        "Explain {} in simple terms.",
        "How does {} work?",
        "Describe the key features of {}.",
        "What are the benefits of {}?",
    ]
    
    topics = [
        "machine learning",
        "cloud computing",
        "data science",
        "artificial intelligence",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "distributed systems",
    ]
    
    prompts = []
    for i in range(num_prompts):
        template = random.choice(templates)
        topic = random.choice(topics)
        content = template.format(topic)
        
        entry = {
            "custom_id": f"bench-{i:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": 0.7,
                "max_tokens": 512,
            }
        }
        prompts.append(entry)
    
    return prompts


def extract_prompts_from_jsonl(
    filepath: Path,
    num_prompts: int = 20
) -> List[Dict[str, Any]]:
    """Extract a shuffled subset of prompts from a JSONL file.

    Args:
        filepath: Input file path
        num_prompts: Maximum number of prompts to return. Use ``0`` to return an
            empty list and negative values to return all prompts.
    """
    if num_prompts == 0:
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    if num_prompts < 0 or num_prompts >= len(prompts):
        return prompts

    random.shuffle(prompts)

    return prompts[:num_prompts]

def create_test_prompts_file(num_prompts: int = 120, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Get test prompts for a given model from 6 LiveBench categories: data_analysis, language, math, reasoning, instruction_following, and coding.
    Args:
        num_prompts: Total number of prompts to generate.
        temperature: Sampling temperature for the prompts.
        max_tokens: Maximum number of tokens for each prompt.
    Returns:
        Path to the created prompts file.
    """
    file_names = ["data_analysis", "language", "math", "reasoning", "instruction_following", "coding"]
    # Check if all the files exist else download the prompts data
    benchmark_dir = ensure_benchmark_data_dir()
    if not all((benchmark_dir / f"{file_name}.jsonl").exists() for file_name in file_names):
        print("Downloading prompts data from the LiveBench HuggingFace dataset...")
        download_prompts_data()
        print("Prompts data downloaded successfully")
    
    all_prompts = []
    prompts_file = benchmark_dir / "benchmark_prompts.jsonl"
    for i in range(len(file_names)):
        file_name = benchmark_dir / f"{file_names[i]}.jsonl"
        prompts = extract_prompts_from_jsonl(file_name, num_prompts=num_prompts//len(file_names))
        for prompt in prompts:
            all_prompts.append({"custom_id":"request1","method":"POST","url":"/v1/chat/completions","body":{"messages":prompt,"temperature":temperature,"max_tokens":max_tokens}})
    save_prompts_to_jsonl(all_prompts, prompts_file)

    return str(prompts_file)



def save_prompts_to_jsonl(prompts: List[Dict[str, Any]], filepath: Path) -> None:
    """Save prompts to JSONL file.
    
    Args:
        prompts: List of prompt dictionaries
        filepath: Output file path
    """

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')

class VllmMetricsScraper:
    """Background thread that scrapes vLLM's /metrics endpoint during a benchmark run.

    Takes a snapshot before and after the run to compute rates, and polls at a
    fixed interval to average gauge metrics over the full run duration.
    """

    def __init__(self, base_url: str, interval_seconds: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.interval = interval_seconds
        self._samples: List[Dict] = []
        self._snapshot_start: Optional[Dict] = None
        self._snapshot_end: Optional[Dict] = None
        self._start_time: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._snapshot_start = self._scrape()
        self._start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True, name="vllm-scraper")
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop polling, take a final snapshot, and return a summary dict."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._snapshot_end = self._scrape()
        return self._summarize()

    # ------------------------------------------------------------------
    # Scraping / parsing
    # ------------------------------------------------------------------

    def _scrape(self) -> Dict:
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=5)
            resp.raise_for_status()
            return self._parse(resp.text)
        except Exception:
            return {}

    def _parse(self, text: str) -> Dict[str, Dict[str, float]]:
        """Parse Prometheus text format into {metric_name: {labels_str: value}}."""
        result: Dict[str, Dict[str, float]] = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = re.match(r"^(\S+)\s+([-\d.eE+]+)", line)
            if not m:
                continue
            full_name, raw_val = m.group(1), m.group(2)
            try:
                value = float(raw_val)
            except ValueError:
                continue
            if "{" in full_name:
                name, labels_str = full_name.split("{", 1)
                labels_str = labels_str.rstrip("}")
            else:
                name, labels_str = full_name, ""
            result.setdefault(name, {})[labels_str] = value
        return result

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            sample = self._scrape()
            if sample:
                self._samples.append(sample)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _sum_metric(self, snapshot: Dict, name: str) -> float:
        return sum(snapshot.get(name, {}).values())

    def _avg_gauge(self, name: str, snapshots: Optional[List[Dict]] = None) -> Optional[float]:
        """Average a gauge metric over samples, falling back to start/end snapshots."""
        sources = list(self._samples)
        if not sources and snapshots:
            sources = [s for s in snapshots if s]
        vals = [self._sum_metric(s, name) for s in sources if name in s]
        return statistics.mean(vals) if vals else None

    def _histogram_percentile(self, snapshot: Dict, base_name: str, pct: float) -> Optional[float]:
        """Estimate a percentile (ms) from a cumulative Prometheus histogram."""
        bucket_name = f"{base_name}_bucket"
        if bucket_name not in snapshot:
            return None
        buckets: List[tuple] = []
        for labels_str, value in snapshot[bucket_name].items():
            le_m = re.search(r'le="([^"]+)"', labels_str)
            if le_m and le_m.group(1) != "+Inf":
                try:
                    buckets.append((float(le_m.group(1)), value))
                except ValueError:
                    pass
        if not buckets:
            return None
        buckets.sort()
        total = self._sum_metric(snapshot, f"{base_name}_count")
        if not total:
            return None
        target = pct / 100.0 * total
        prev_le, prev_count = 0.0, 0.0
        for le, count in buckets:
            if count >= target:
                if count == prev_count:
                    return round(le * 1000, 1)
                interp = prev_le + (le - prev_le) * (target - prev_count) / (count - prev_count)
                return round(interp * 1000, 1)
            prev_le, prev_count = le, count
        return round(buckets[-1][0] * 1000, 1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summarize(self) -> Dict[str, Any]:
        start = self._snapshot_start or {}
        end = self._snapshot_end or {}
        elapsed = time.time() - self._start_time if self._start_time else 0

        if not end or not elapsed:
            return {}

        # Rates: delta between end and start snapshots divided by elapsed
        req_delta = self._sum_metric(end, "vllm:request_success_total") - self._sum_metric(start, "vllm:request_success_total")
        tok_delta = self._sum_metric(end, "vllm:generation_tokens_total") - self._sum_metric(start, "vllm:generation_tokens_total")

        # KV cache: fraction (0–1) → percentage; fall back to snapshots if no polling samples
        kv_raw = self._avg_gauge("vllm:kv_cache_usage_perc", [start, end])
        kv_cache_pct = round(kv_raw * 100, 4) if kv_raw is not None else None

        # Effective batch size: average number of requests actively running
        eff_batch: Optional[float] = None
        if self._samples:
            counts = [self._sum_metric(s, "vllm:num_requests_running") for s in self._samples]
            eff_batch = round(statistics.mean(counts), 2) if counts else None

        return {
            "vllm_request_throughput_req_per_sec": round(req_delta / elapsed, 3) if elapsed else None,
            "vllm_token_throughput_tok_per_sec": round(tok_delta / elapsed, 1) if elapsed else None,
            "vllm_ttft_p50_ms": self._histogram_percentile(end, "vllm:time_to_first_token_seconds", 50),
            "vllm_ttft_p95_ms": self._histogram_percentile(end, "vllm:time_to_first_token_seconds", 95),
            "vllm_itl_p50_ms": self._histogram_percentile(end, "vllm:request_time_per_output_token_seconds", 50),
            "vllm_itl_p95_ms": self._histogram_percentile(end, "vllm:request_time_per_output_token_seconds", 95),
            "vllm_kv_cache_usage_avg_pct": kv_cache_pct,
            "vllm_effective_batch_size_avg": eff_batch,
            "vllm_scrape_count": len(self._samples),
        }


class GpuUtilScraper:
    """Background thread that samples GPU utilization via nvidia-smi.

    Polls all visible GPUs at a fixed interval. Reports average and peak
    utilization across all GPUs and all samples.
    """

    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True, name="gpu-util-scraper")
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop polling and return a summary dict."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self._summarize()

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            self._samples.extend(self._sample())

    # Common nvidia-smi locations on HPC systems where it may not be in PATH
    _NVIDIA_SMI_CANDIDATES = [
        "nvidia-smi",
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
    ]

    @classmethod
    def _find_nvidia_smi(cls) -> Optional[str]:
        """Return the first usable nvidia-smi path, or None."""
        for candidate in cls._NVIDIA_SMI_CANDIDATES:
            path = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
            if path:
                return path
        return None

    def _sample(self) -> List[float]:
        """Query utilization.gpu for all GPUs; return list of floats (one per GPU)."""
        smi = self._find_nvidia_smi()
        if not smi:
            return []
        try:
            result = subprocess.run(
                [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return [float(v.strip()) for v in result.stdout.splitlines() if v.strip()]
        except Exception:
            pass
        return []

    def _summarize(self) -> Dict[str, Any]:
        if not self._samples:
            return {"gpu_util_avg_pct": None, "gpu_util_peak_pct": None}
        return {
            "gpu_util_avg_pct": round(statistics.mean(self._samples), 1),
            "gpu_util_peak_pct": round(max(self._samples), 1),
        }


def compute_benchmark_metrics(output_path: str) -> Dict[str, Any]:
    """Read a benchmark results file and return a flat metrics dict.

    Supports both the legacy list format and the current envelope format:
    {"results": [...], "vllm_metrics": {...}}
    """
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        results = data
        vllm_metrics: Dict[str, Any] = {}
        gpu_metrics: Dict[str, Any] = {}
    else:
        results = data.get("results", [])
        vllm_metrics = data.get("vllm_metrics", {})
        gpu_metrics = data.get("gpu_metrics", {})

    total = len(results)
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    metrics: Dict[str, Any] = {
        "total_requests": total,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "success_rate_pct": round(100 * len(successful) / total, 2) if total else 0.0,
        **vllm_metrics,
        **gpu_metrics,
    }
    return metrics


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format a metrics dict as a human-readable table string."""
    labels = [
        ("Total requests",          "total_requests"),
        ("Successful",              "successful_requests"),
        ("Failed",                  "failed_requests"),
        ("Success rate",            "success_rate_pct",                    "%"),
        ("Total time",              "elapsed"),
        ("Request throughput",      "vllm_request_throughput_req_per_sec", "req/s"),
        ("Token throughput",        "vllm_token_throughput_tok_per_sec",   "tok/s"),
        ("p50 TTFT",                "vllm_ttft_p50_ms",                    "ms"),
        ("p95 TTFT",                "vllm_ttft_p95_ms",                    "ms"),
        ("p50 inter-token latency", "vllm_itl_p50_ms",                     "ms"),
        ("p95 inter-token latency", "vllm_itl_p95_ms",                     "ms"),
        ("KV cache usage avg",      "vllm_kv_cache_usage_avg_pct",         "%"),
        ("Effective batch size avg","vllm_effective_batch_size_avg"),
        ("GPU utilization avg",     "gpu_util_avg_pct",                    "%"),
        ("GPU utilization peak",    "gpu_util_peak_pct",                   "%"),
    ]
    lines = ["\n=== Benchmark Metrics ==="]
    for row in labels:
        label, key = row[0], row[1]
        unit = row[2] if len(row) > 2 else ""
        val = metrics.get(key)
        display = "N/A" if val is None else (f"{val} {unit}" if unit else str(val))
        lines.append(f"  {label:<28} {display}")
    lines.append("=" * 32)
    return "\n".join(lines)


def download_prompts_data() -> None:
    """
    Download prompts data from the LiveBench HuggingFace dataset.
    Categories:  "coding", "data_analysis", "instruction_following","math","reasoning","language"
    """

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "download_prompts_data requires the optional dependencies 'datasets' and 'pandas'. "
            "Install them to enable downloading prompts."
        ) from exc

    # LiveBench categories
    CATEGORIES = [
        "coding",
        "data_analysis", 
        "instruction_following",
        "math",
        "reasoning",
        "language",
    ]

    # System prompts by category
    SYSTEM_PROMPTS = {
        "coding": "You are an expert programmer. Provide accurate, well-structured code solutions.",
        "math": "You are a mathematics expert. Think step by step and provide precise solutions.",
        "data_analysis": "You are a data analysis expert. Provide accurate and detailed analysis.",
        "reasoning": "You are an expert at logical reasoning. Think through problems systematically.",
        "instruction_following": "You are a helpful assistant that follows instructions precisely.",
        "language": "You are an expert in language and communication.",
    }

    def download_and_convert_category(category: str) -> None:
        """
        Download and convert a category from the LiveBench HuggingFace dataset.
        """
        print(f"Loading {category} from HuggingFace...")
        dataset = load_dataset(f"livebench/{category}", split="test")

        # Convert to pandas DataFrame
        df = dataset.to_pandas()

        # Get system prompt for category
        system_prompt = SYSTEM_PROMPTS[category]

        # Create output directories
        benchmark_dir = ensure_benchmark_data_dir()

        # Function to convert to OpenAI batch format
        def create_openai_batch_row(row: pd.Series, system_prompt: str) -> Dict[str, Any]:
            """
            Create an OpenAI batch row from a LiveBench row.
            """
            messages = [{"role": "system", "content": system_prompt}]
            for turn in row["turns"]:
                messages.append({
                "role": "user",
                "content": turn
            })
    
            return messages
        
        prompts = []
        # Convert to OpenAI batch format
        for index, row in df.iterrows():
            openai_row = create_openai_batch_row(row, system_prompt)
            prompts.append(openai_row)
        
        # Save OpenAI batch format to JSONL file
        openai_path = benchmark_dir / f"{category}.jsonl"
        save_prompts_to_jsonl(prompts, openai_path)
        print(f"Saved {category} prompts to {openai_path}")
        return None
    
    # Download and convert all categories
    for category in CATEGORIES:
        try:
            download_and_convert_category(category)
        except Exception as e:
            print(f"Error downloading and converting {category}: {e}")
    
    return None
