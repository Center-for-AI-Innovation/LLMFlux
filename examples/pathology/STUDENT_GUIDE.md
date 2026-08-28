# Student walkthrough: using the pathology embedding server

This is for teams at the hackathon using the pathology/CLIP model the
organizers staged for the event. You do **not** need CONCH, MUSK, torch, a
GPU, or any of the model code in this directory — all of that runs on a
server the organizers are hosting. Your machine just sends it images over
HTTP and gets back embeddings or predicted labels.

If you're looking for how to *set up* that server, see `README.md` instead —
this doc is the other side of it: what to do with the endpoint once you have
it.

## 0. What you'll get from the organizers

Three things, usually from a shared doc or `connection.json`:

- **ENDPOINT** — a URL like `http://gpub042:8000`
- **API_KEY** — a token you pass as a bearer token
- **Which model** is running (e.g. CONCH, MUSK, QuiltNet, plain CLIP) — this
  matters for what labels/prompts make sense, not for how you call the API;
  the HTTP interface is identical no matter which model is staged.

## 1. Install

```bash
pip install requests pillow
```

That's the entire dependency list. `pillow` is only used by the demo script
below to generate a placeholder image before you have real data; once you're
sending your own image files, you don't even need it.

## 2. Run the quickstart as-is

```bash
python example_student_usage.py
```

Fill in `ENDPOINT` and `API_KEY` at the top of the file first. Running it
unmodified should print something like:

```
Connected. Server is running model=conch on device=cuda.

/embed: got 3 embeddings, each 512-dimensional.
Use these for clustering/kNN/training your own classifier - see STUDENT_GUIDE.md.

/classify tile 0: predicted 'stroma' (score=0.412)
/classify tile 1: predicted 'stroma' (score=0.398)
/classify tile 2: predicted 'necrosis' (score=0.405)
(these predictions are meaningless here - solid-color placeholders, not real tissue)
```

The exact numbers won't match — they depend on which model the organizers
staged — but you should get through all three sections without an error. If
you don't, see **Troubleshooting** below before going further.

## 3. Swap in your own images

Replace the placeholder tiles with your own files:

```python
# was:
tiles = make_placeholder_tiles_b64()

# becomes:
tiles = [load_your_own_image(p) for p in ["tile_001.png", "tile_002.png", "tile_003.png"]]
```

`load_your_own_image` just reads the file and base64-encodes it — no
resizing, cropping, or normalization needed on your end. The server's model
adapter handles all of that server-side, exactly the way it does for the
placeholder tiles.

## 4. Swap in your own labels

Edit the `prompts` dict to whatever your team is actually trying to
distinguish:

```python
prompts = {
    "your_label_1": "a histopathology image of <description>",
    "your_label_2": "a histopathology image of <description>",
}
```

Write each value as a real sentence describing the tissue/feature, not just
the bare label — zero-shot classification compares your image against the
*text* of each prompt, so a more descriptive prompt usually scores better
than a single word.

## 5. `/embed` vs `/classify` — which one do you want?

- **`/classify`** gives you a label out of a fixed set you provide, plus a
  confidence score. Fastest way to get something working, but you're limited
  to zero-shot comparisons against text prompts you write by hand.
- **`/embed`** gives you the raw embedding vector for each image (its
  position in the model's learned feature space) and nothing else. This is
  what you want once zero-shot isn't enough — e.g.:
  - **Clustering** — group your images by similarity without any labels at all.
  - **k-NN / similarity search** — find the images in your dataset most similar
    to a query image (cosine similarity between embedding vectors).
  - **Train a small classifier on top** — if you have even a handful of labeled
    examples, fit a logistic regression / small MLP on the embeddings instead
    of writing prompts. This is usually far more accurate than zero-shot
    `/classify` once you have any labeled data at all, and it's cheap — you're
    training on ~512-1024-dim vectors, not raw images, so plain scikit-learn on
    a laptop is enough:

    ```python
    from sklearn.linear_model import LogisticRegression

    embeddings = embed(your_image_tiles)      # from the server
    clf = LogisticRegression(max_iter=1000).fit(embeddings, your_labels)
    ```

Both endpoints accept a list of images in one request — send your whole batch
at once rather than looping one image per call; it's faster and is exactly
what the server's micro-batching is built to take advantage of.

## 6. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Could not reach <endpoint>` | `ENDPOINT` is wrong, or the server job ended/was `scancel`ed. Confirm with the organizers. |
| `HTTP 401` | `API_KEY` is wrong or stale — re-check `connection.json` / what the organizers gave you. |
| `{"detail": "... does not support text encoding"}` from `/classify` | The staged model is image-only (rare) — use `/embed` instead, or ask the organizers what the model supports. |
| Request hangs or times out | Large batch or the server is under heavy load from many teams at once — try a smaller batch, or check back with the organizers. |
| Slightly different scores each run | Expected for some models — normal floating-point nondeterminism, not a bug. |

## 7. Reference: raw HTTP, no `client.py`/demo script at all

If you'd rather not use Python, or want to call this from another language:

```
POST <ENDPOINT>/embed
Authorization: Bearer <API_KEY>
Content-Type: application/json

{"images": ["<base64-encoded image bytes>", ...]}
```

```
POST <ENDPOINT>/classify
Authorization: Bearer <API_KEY>
Content-Type: application/json

{"images": ["<base64>", ...], "prompts": {"label": "description", ...}}
```

`GET <ENDPOINT>/health` (no auth needed) returns `{"status": "ok", "model": "...", "device": "..."}` — useful for a quick reachability check from `curl`.
