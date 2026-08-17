# Remote Chair CRISP Run

This is the reproducible runbook for the custom chair concept experiment. The
training corpus and held-out evaluation prompts are included under
`data/chair-corpus/`.

## Machine

Use a CUDA machine with a bfloat16-capable GPU. A 24 GB GPU is the safer
starting point for Gemma 2 2B plus the Gemma SAEs. The chair config uses
conservative sequence and batch sizes:

- model: `google/gemma-2-2b-it`;
- SAE layers: `4, 6, 8, 10, 12, 14`;
- feature batch size: `1`;
- training batch size: `1`;
- maximum sequence length: `512`.

If memory remains insufficient, reduce `max_length` to `384` in
`config-files/crisp-chair-gemma-2b-it.json`. Do not increase batch sizes until
the baseline run completes.

## Setup

Clone the repository on the rented machine, then install the locked
environment:

```bash
uv sync --locked
export HF_TOKEN="your-huggingface-token"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

The token is required for Gemma and SAE downloads. Do not put it in a config
file or commit it.

## Train

Run from the repository root:

```bash
uv run python main.py \
  --config-path config-files/crisp-chair-gemma-2b-it.json \
  --log-level INFO
```

The first run downloads the Gemma SAEs into `gemma_sae_cache/` and creates
feature caches under `crisp_cache/`. The adapter is written to:

```text
outputs/crisp/gemma-2b-it-chair/
```

The generated adapter and model artifacts are intentionally ignored by Git.

## Evaluate

After training:

```bash
uv run python scripts/evaluate_chair_adapter.py \
  outputs/crisp/gemma-2b-it-chair
```

This writes `chair-evaluation.json` inside the adapter directory. It contains
base-model and adapter outputs for the held-out chair and retain prompts.

Review the results for:

- inability to produce a usable physical-chair or wheelchair description;
- preserved descriptions of stools, benches, sofas, tables, beds, and other
  retained objects;
- general output coherence and formatting.

## Layer-Coverage Experiment

`config-files/crisp-chair-gemma-2b-it-layer14.json` is the next experiment
after the overly weak baseline and overly disruptive strong run. It keeps the
same corpus and SAE layers, but changes the trainable LoRA layer coverage from
the historical default of `3-9` to `3-14`.

This matters because the selected SAE layers include `10`, `12`, and `14`.
With LoRA limited to `3-9`, the optimization can only influence those later
representations indirectly. The layer-coverage experiment attaches LoRA to the
attention and MLP projections at every selected SAE layer, while using an
intermediate unlearning strength (`beta=0.95`, two epochs) to reduce the broad
text corruption observed with the strong run.

Run and evaluate it with:

```bash
uv run python main.py \
  --config-path config-files/crisp-chair-gemma-2b-it-layer14.json \
  --log-level INFO

uv run python scripts/evaluate_chair_adapter.py \
  outputs/crisp/gemma-2b-it-chair-layer14
```

LoRA layers `3-14` add modest adapter memory at rank 4. Keep the batch size at
1 and the maximum sequence length at 512 for the first run.

## Recovery

Feature extraction is cached by model family, exact corpus hash, and layer.
If training fails after feature extraction, rerun the same command. It will
reuse compatible feature caches. If the config or corpus changes, a new cache
identity is generated.

For an OOM during feature extraction, lower `feature_batch_size` first. For an
OOM during LoRA training, lower `batch_size` or `max_length`.

## Files To Transfer

The Git commit contains the code, config, corpus, and held-out evaluation
prompts. Do not transfer local `outputs/`, `gemma_sae_cache/`, or generated
model weights through Git; they are downloaded or produced on the rented
machine.
