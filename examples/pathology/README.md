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

## Never run this on the login node

Delta's login node enforces per-process CPU/memory limits and kills anything
that looks like sustained compute. Concretely:

- **Never run `serve.py` or `run_embeddings.py` on the login node** — these
  load the model and do the actual inference. It won't error like a crash;
  the process just gets silently `Terminated` out from under you. Always run
  them from an interactive allocation or as a batch job (below).
- **`client.py` and `loadtest.py` are lightweight HTTP clients** — once a
  server is actually running on a compute node (via `srun` or
  `submit_serve.sbatch`), calling it from the login node with these is fine,
  the same as running `curl`.
- **`pip install -r requirements.txt`** is also fine on the login node — it
  needs internet access, which compute nodes on Delta don't have.

To get an interactive allocation for `serve.py` / `run_embeddings.py`:

```bash
srun --account=<your_account> --partition=gpuA100x4 --gpus-per-node=1 \
    --mem=16G --cpus-per-task=4 --time=00:30:00 --pty bash
```

That blocks until you're granted a node and drops you into a shell running on
it (`squeue` will show the job as RUNNING) — run the server from there, or
submit it as a batch job with `submit.sbatch` / `submit_serve.sbatch` instead.

## Setup

1. **Request model access.** Both checkpoints are gated on Hugging Face:
   - CONCH: https://huggingface.co/MahmoodLab/CONCH
   - MUSK: https://huggingface.co/xiangjx/musk

   Accept each model's license on its Hugging Face page, then create a token
   at https://huggingface.co/settings/tokens and export it:

   ```bash
   export HF_TOKEN=hf_...
   ```

2. **Install dependencies.** Verified working on Delta as of 2026-09-17:
   `module load pytorch-conda/2.8` already provides everything in
   `requirements.txt` except `open_clip_torch` (torch, torchvision, timm,
   transformers, huggingface_hub, Pillow, pandas, tqdm, fastapi, uvicorn,
   requests all included) — just add that one package:

   ```bash
   module load pytorch-conda/2.8
   pip install --user "open_clip_torch>=2.24"
   ```

   `submit.sbatch`/`submit_serve.sbatch` already `module load` this. Without
   that module (a different cluster, or you want an isolated env), fall back
   to a plain venv/conda env with `pip install -r requirements.txt` — this
   step (only) is fine on the login node, since it needs internet access.
   Either way, CONCH and MUSK themselves still ship as small research
   packages installed straight from their GitHub repos, not through
   `requirements.txt`. **Before relying on `models.py`**, check each
   project's own README for its current install command and loading API
   — these are small academic repos that change without a stable public API,
   and the adapters below reflect what was documented as of this writing, not
   a guaranteed-stable interface:
   - CONCH: https://github.com/mahmoodlab/CONCH
   - MUSK: https://github.com/lilab-stanford/MUSK

3. **Verify** by running the script on a handful of test tiles before
   pointing it at a full dataset — from an interactive allocation or a batch
   job (see above), never the login node.

## Running a model other than CONCH/MUSK

CONCH and MUSK needed bespoke adapters in `models.py` because they ship as
small research repos with non-standard loading code. Most other
pathology/CLIP models don't have that problem — they're published in one of
two standard formats, and `models.py`'s `resolve_adapter()` can load either
with **no new code**, just a `--model` / `PATHOLOGY_MODEL` spec string:

- **`openclip:<hf-repo>`** — a model published in OpenCLIP's own Hugging Face
  Hub format (an `open_clip_config.json` + weights on the repo), e.g.:

  ```bash
  python run_embeddings.py --model openclip:wisdomik/QuiltNet-B-32 ...
  ```

- **`openclip:<arch>@<hf-repo>/<filename>`** — a bare OpenCLIP checkpoint
  file on an otherwise plain HF repo (no OpenCLIP hub config), naming the
  architecture it was trained with and the file to download, e.g.:

  ```bash
  python run_embeddings.py --model openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt ...
  ```

- **`hfclip:<hf-repo>`** — any `transformers` `CLIPModel` repo, e.g.:

  ```bash
  python run_embeddings.py --model hfclip:openai/clip-vit-base-patch32 ...
  ```

Same specs work for `serve.py` via `PATHOLOGY_MODEL` (and `submit_serve.sbatch
<spec>` / `submit.sbatch <input_dir> <output_csv> <spec>`). This covers any
team that wants to try a pathology or general-purpose CLIP model we haven't
looked at ahead of time, as long as it's CLIP-shaped and published in one of
these two formats — which most are. A model that isn't (another CONCH/MUSK
situation: custom research code, not a standard checkpoint) needs a bespoke
adapter added to `ADAPTERS` in `models.py`, following the `load_conch`/
`load_musk` pattern; that's a small, contained addition, not a redesign. A
team whose model doesn't fit either path is in the same position as any team
bringing entirely their own tooling — they should get their own allocation
and run it directly, same as normal HPC usage.

Gated models (like CONCH/MUSK) still need `HF_TOKEN` set to a token that has
accepted the model's license, same as above.

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

From an interactive allocation (see above) or inside `submit.sbatch`:

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
sbatch --account=bfqi-delta-gpu submit.sbatch /path/to/image/tiles embeddings.csv conch
```

`submit.sbatch` targets Delta's `gpuA100x4` partition by default; adjust the
`#SBATCH` directives and the environment activation step for your account and
however you installed the dependencies (conda env vs. venv vs. container).

### Serve mode (for the hackathon)

Stage one model for the whole event:

```bash
sbatch --account=bfqi-delta-gpu --reservation=<name> submit_serve.sbatch conch
```

This starts `serve.py` on a GPU node, waits for it to come up, and writes
connection info (host, port, a generated API key) to two places: a per-job
file at `~/.pathology/serve/<job_id>/connection.json`, and a stable one at
`~/.pathology/serve/current-<job-name>/connection.json` (job name defaults
to `pathology-serve`) that doesn't move for the life of the whole chain (see
below) — **hand out the stable one**, or `tail logs/<job_id>.out` for the
same info.

**Multi-day events (e.g. Friday afternoon → Monday EOD): the job
auto-resubmits itself.** Delta's `gpuA100x4` partition caps a single job at
48h, shorter than a weekend-long event. `submit_serve.sbatch` resubmits
itself shortly before hitting its own walltime, so the service keeps running
across the whole event without anyone resubmitting it by hand. This means a
real `scancel` at the end of the event isn't enough by itself to stop it —
see "stop the chain for good" in the script's header comment (`touch
~/.pathology/serve/STOP_CHAIN-<job-name>` before the next walltime boundary,
or right after your final `scancel`).

**Verified on a real Delta allocation on 2026-09-17:** the resubmit fires,
the API key carries over (same key, no redistribution needed), and
`STOP_CHAIN-<job-name>` does stop it. **One real gap found:** the chain pins
the next link to the *same* node (`--nodelist`) so the endpoint doesn't
move, but on the shared `gpuA100x4` queue (no reservation active during this
test) the pinned successor sat `PENDING (Priority)` for 7+ minutes waiting
for that specific node, rather than starting immediately on any free node.
Without a reservation guaranteeing dedicated nodes, this node-pinning
approach risks turning each ~48h resubmission into several minutes of real
downtime — worth re-testing once the reservation is in place (dedicated
nodes should make this instant), and worth revisiting whether to drop
`--nodelist` entirely and accept a moving host (already handled by the
stable `current-<job-name>/connection.json` file) in exchange for the
successor being schedulable on *any* free node.

Teams call it with `client.py`:

```bash
python client.py --endpoint http://<node>:<port> --api-key <key> tile.png
python client.py --endpoint http://<node>:<port> --api-key <key> --prompts prompts.json tile.png
```

or directly with any HTTP client — `POST /embed {"images": [<base64>, ...]}`
or `POST /classify {"images": [...], "prompts": {label: prompt, ...}}`, both
under `Authorization: Bearer <key>`. See `serve.py`'s docstring for the exact
schema.

**Give students `STUDENT_GUIDE.md` and `example_student_usage.py`.** The
guide is the walkthrough (what `/embed` vs `/classify` return, swapping in
their own images/prompts, troubleshooting a bad endpoint or stale key); the
script is its runnable counterpart — self-contained, runs immediately against
synthetic placeholder tiles so it works before they have real data wired up,
then edit in place: swap the placeholders for their own image files and the
sample labels for whatever their team is classifying. The only dependency is
`requests` (plus `pillow`, only for the placeholder tiles it generates) — none
of CONCH/MUSK/torch, since all of that runs server-side. Hand out
`ENDPOINT`/`API_KEY` from `connection.json` and that's the entire setup on
their end.

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
# fine from the login node — this only sends HTTP requests to the remote server
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

The auto-resubmit chain (see "Serve mode" above and its "Verified on a real
Delta allocation" note) has been exercised end-to-end with a short `--time`
override: the signal fires, the resubmit is submitted, the API key carries
over, and `STOP_CHAIN-<job-name>` stops it. The open question from that test
is queue delay for the `--nodelist`-pinned successor without a reservation
— see the note above.
