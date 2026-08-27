# CONCH / MUSK Pathology Embeddings (standalone)

Tooling for running the CONCH and MUSK computational-pathology foundation
models on Delta, for the pathology track of the hackathon.

## Why this isn't part of the LLMFlux package

LLMFlux's entire pipeline — `llmflux serve`/`llmflux run`, `LLMClient`,
`BatchProcessor`, the JSONL schema — assumes an OpenAI-compatible
`/v1/chat/completions` server backed by vLLM or Ollama. CONCH and MUSK are not
chat models: they're dual image/text encoders (CLIP-style) used to produce
embeddings for zero-shot classification and retrieval, and neither is served
by vLLM or Ollama. So this directory is a plain, self-contained script run
directly on a compute node — no LLMFlux engine, converter, or output-handler
code involved. It's kept in this repo only for discoverability alongside the
other hackathon prep (see #129, #131, #134), not because it depends on the
`llmflux` package.

## Setup

1. **Request model access.** Both checkpoints are gated on Hugging Face:
   - CONCH: https://huggingface.co/MahmoodLab/CONCH
   - MUSK: https://huggingface.co/xiangjx/musk

   Accept each model's license on its Hugging Face page, then create a token
   at https://huggingface.co/settings/tokens and export it:

   ```bash
   export HF_TOKEN=hf_...
   ```

2. **Install dependencies.** These models ship as small research packages
   installed straight from their GitHub repos, not PyPI releases — install
   inside a venv/conda env on a Delta compute node (or GPU login node):

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` pins the generic dependencies (torch, timm,
   huggingface_hub, Pillow, pandas). **Before relying on `models.py`**, check
   each project's own README for its current install command and loading API
   — these are small academic repos that change without a stable public API,
   and the adapters below reflect what was documented as of this writing, not
   a guaranteed-stable interface:
   - CONCH: https://github.com/mahmoodlab/CONCH
   - MUSK: https://github.com/lilab-stanford/MUSK

3. **Verify** by running the script on a handful of test tiles before
   pointing it at a full dataset (step 4 below).

## Two ways to run this

- **`run_embeddings.py`** — a one-shot batch job: point it at a directory of
  tiles, it writes a CSV and exits. Good for a single team processing their
  own dataset.
- **`serve.py`** — a persistent server: load the model once, keep it warm on
  a GPU, and let many teams send it requests concurrently over HTTP for the
  whole event. **This is the one for staging a model for a hackathon cohort**
  — submitting 10-20 separate batch jobs would mean 10-20 redundant model
  loads and would compete for GPU allocations, whereas one served instance
  handles concurrent requests from all of them.

### Batch mode

```bash
python run_embeddings.py \
  --model conch \
  --input-dir /path/to/image/tiles \
  --output embeddings.csv
```

Add `--text-prompts prompts.json` to also run zero-shot classification
against a fixed label set, where `prompts.json` is:

```json
{
  "tumor": "a histopathology image of tumor tissue",
  "stroma": "a histopathology image of stroma",
  "necrosis": "a histopathology image of necrotic tissue"
}
```

Output is a CSV with one row per image (`image_path`, and either
`embedding` as a JSON-encoded list, or `predicted_label`/`score` per label
when `--text-prompts` is given). Results are flushed every
`--checkpoint-every` images (default 50) so a timed-out or killed job doesn't
lose completed work — rerun with the same `--output` path to resume; already
processed images are skipped.

Run `python run_embeddings.py --help` for all options.

```bash
sbatch --account=$SLURM_ACCOUNT submit.sbatch /path/to/image/tiles embeddings.csv conch
```

`submit.sbatch` targets Delta's `gpuA100x4` partition by default; adjust the
`#SBATCH` directives and the environment activation step for your account and
however you installed the dependencies (conda env vs. venv vs. container).

### Serve mode (for the hackathon)

Stage one model for the whole event:

```bash
sbatch --account=$SLURM_ACCOUNT submit_serve.sbatch conch
```

This starts `serve.py` on a GPU node, waits for it to come up, and writes
`~/.pathology/serve/<job_id>/connection.json` (host, port, a generated API
key) — read that file (or `tail logs/<job_id>.out`) to get the endpoint to
hand out to teams. It runs for the `#SBATCH --time` you set (default 8h);
`scancel` it when the event is over.

Teams call it with `client.py`:

```bash
python client.py --endpoint http://<node>:<port> --api-key <key> tile.png
python client.py --endpoint http://<node>:<port> --api-key <key> --prompts prompts.json tile.png
```

or directly with any HTTP client — `POST /embed {"images": [<base64>, ...]}`
or `POST /classify {"images": [...], "prompts": {label: prompt, ...}}`, both
under `Authorization: Bearer <key>`. See `serve.py`'s docstring for the exact
schema.

**How concurrency is handled:** `serve.py` loads the model once and runs all
GPU forward passes through a single background thread that micro-batches
whatever concurrent requests arrived in the last `PATHOLOGY_MAX_WAIT_MS`
(default 50ms, up to `PATHOLOGY_MAX_BATCH_SIZE` images, default 32) into one
inference call, then returns each request its own slice of the result. This
means N teams hitting it at once cost roughly one batched forward pass, not N
serialized ones, and a GPU forward pass is never called from more than one
thread at a time (required for correctness).

**What's been tested vs. what hasn't:** `loadtest.py` exercises the
concurrency logic itself — request batching, response routing back to the
right caller, and that concurrent requests never get mixed up with each
other — end-to-end against a real running server, but using a fake in-process
adapter (no GPU or model weights needed), verified for up to 30 simultaneous
callers. It does **not** tell you real CONCH/MUSK throughput or GPU memory
headroom for `MAX_BATCH_SIZE` images at once — run it again against the real
`serve.py` (`--model conch` or `--model musk`) on an actual Delta allocation,
at your expected team count, before the event:

```bash
# on the compute node / login node with the server reachable
python loadtest.py --endpoint http://<node>:<port> --api-key <key> \
    --teams 20 --images-per-team 3
```

It reports success/failure per simulated team, latency percentiles, and
whether batching actually happened server-side. If it reports OOM errors or
very high p95 latency at your target team count, lower
`PATHOLOGY_MAX_BATCH_SIZE` (less GPU memory per batch) or raise
`PATHOLOGY_MAX_WAIT_MS` (more batching, more per-request latency) via the
server's environment, or split teams across a second `submit_serve.sbatch`
instance.
