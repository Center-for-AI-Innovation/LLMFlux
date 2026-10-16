# Hackathon computing: Delta, Lumen, and your own machine

Student-facing overview of what compute you have this weekend
(Sat 2026-10-17 – Sun 2026-10-18, 8:30am–5:00pm) and how to get on it.
For using the pathology embedding servers once you're on, see
`STUDENT_GUIDE.md`.

**GPU time on Delta is limited and shared by every team.** Plan to do most
of your work on your own laptop (or Colab), use **Lumen** for LLMs, use the
organizers' pathology servers for CONCH/MUSK/CLIP embeddings, and save Delta
GPU jobs for the things that genuinely need one.

## 1. Check that your Delta account exists

Your Delta (NCSA) username comes from ACCESS:

1. Log in at <https://allocations.access-ci.org>.
2. Go to **My Projects** → the Delta project for the event → **My Username**.
   That's your NCSA username. (Some NCSA docs tell you to look for a
   "Resource Provider Site Usernames" table on your profile page — that table
   isn't there in the current portal; use My Projects instead.)

The real test is logging in:

```bash
ssh <username>@login.delta.ncsa.illinois.edu
```

You'll be asked for your NCSA password and a Duo prompt (type `1` for a
push). Once you're in, run:

```bash
accounts
```

You should see **`bfqi-delta-gpu`** in the list. If you can't log in, or
`bfqi-delta-gpu` isn't listed, find a TA — don't wait until you need a GPU.

## 2. GPU jobs on Delta

There is a set of A100 nodes reserved for students during the event, under
the `bfqi-delta-gpu` account. The reservation is *magnetic*: you don't need
to name it, any `bfqi-delta-gpu` job on `gpuA100x4` that fits gets pulled
onto those nodes automatically.

Interactive (for debugging — release it when you're done):

```bash
srun --account=bfqi-delta-gpu --partition=gpuA100x4 --gpus-per-node=1 \
     --cpus-per-task=16 --mem=32G --time=01:00:00 --pty bash
```

Batch:

```bash
sbatch --account=bfqi-delta-gpu --partition=gpuA100x4 --gpus-per-node=1 \
       --time=02:00:00 my_job.sbatch
```

Rules of thumb:

- **One GPU per team at a time** unless a TA says otherwise. There are 16
  GPUs for the whole event.
- **Keep `--time` short** (an hour or two) and make sure your job ends before
  5:00pm Saturday / 8:00am Sunday / 6:00pm Sunday. A job whose time limit runs
  past the end of the reservation won't be placed on the reserved nodes; it
  will sit in the general Delta queue instead.
- **Never run training or inference on the login node** — that's what `srun`
  is for. `pip install`, editing, and sending HTTP requests are fine there.
- `squeue -u $USER` shows your jobs; `scancel <jobid>` ends one. Cancel idle
  interactive sessions — they hold a GPU even if you aren't using it.

## 3. Shared datasets and where to save files

Read-only copies of Quilt-1M, PathMMU, NCT-CRC-HE-100K and PathGen will be
on Delta. **The path will be posted in the event channel and at the 8:30
briefing.** Read them in place; don't copy them into your own space.

Where to write your own files:

- **Your home directory** (`~`, 100 GB) for code, small outputs, and
  environments.
- **`/work/hdd/bfqi/$USER`** for larger outputs (embeddings, checkpoints);
  `mkdir -p /work/hdd/bfqi/$USER` if it doesn't exist yet.
- **Not `/projects/bfqi`** — that project space is full, and writes there
  will fail.

## 4. LLMs: Lumen

Lumen (<https://lumen.ncsa.illinois.edu>) hosts a range of open-weight LLMs
(coding models, general chat models) with an OpenAI-compatible API. Event
participants have been added to it.

**Get an API key:**

1. Log in at <https://lumen.ncsa.illinois.edu>.
2. Open your **Profile**.
3. Click **+ New API Key**, give it a name, and **copy the key immediately**
   — it's only shown once. It starts with `sk_`.

Your Profile page also shows your **coin** balance. Every request costs
coins; the organizers will top balances up during the event, but if you see
"budget exhausted" errors, tell a TA.

**Check it works** (from anywhere with internet — laptop or Delta):

```bash
export LUMEN_API_KEY=sk_...        # your key
curl https://lumen.ncsa.illinois.edu/v1/models \
     -H "Authorization: Bearer $LUMEN_API_KEY"
```

That lists the model IDs you can use.

**From Python** (`pip install openai`):

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["LUMEN_API_KEY"],
                base_url="https://lumen.ncsa.illinois.edu/v1")
reply = client.chat.completions.create(
    model="qwen3-coder-next",   # any ID from /v1/models
    messages=[{"role": "user", "content": "Explain H&E staining in two sentences."}],
)
print(reply.choices[0].message.content)
```

**As a coding agent in opencode**: put this in
`~/.config/opencode/opencode.json` and keep `LUMEN_API_KEY` exported in your
shell. Swap in any model IDs from `/v1/models`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "lumen/qwen3-coder-next",
  "provider": {
    "lumen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lumen (NCSA)",
      "options": {
        "baseURL": "https://lumen.ncsa.illinois.edu/v1",
        "apiKey": "{env:LUMEN_API_KEY}"
      },
      "models": {
        "qwen3-coder-next": { "name": "Qwen3 Coder Next" },
        "glm-5.3": { "name": "GLM 5.3" }
      }
    }
  }
}
```

**Never paste your API key into code you commit or share.** Read it from an
environment variable as above. If a key leaks, delete it on your Profile page
and make a new one.

## 5. Your own resources

Any laptop works for calling Lumen and the pathology servers — they're just
HTTP. For your own GPU work beyond what Delta can give you, Google Colab or
your own lab/cloud resources are fair game.
