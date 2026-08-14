# Agent Notes

## Repo Shape
- Current branch: `crisp-implementation`. Primary method is **CRISP** (Sparse Autoencoder-based concept unlearning), not the hybrid pruning pipeline.
- `main.py` is the CRISP entrypoint: downloads SAEs, extracts features from forget/retain datasets, trains a LoRA adapter, and evaluates on MCQ benchmarks.
- Legacy hybrid pruning code (neuron saliency + vocabulary pruning) exists in `src/pruning.py`, `src/neuron_saliency_analysis.py`, `src/pruning_operations.py` but is **not wired to `main.py`**.
- `config-files/chair-3B-hybrid-config.json` is a schema reference for the paper's final study config but will fail with current `main.py` (different schema).

## Commands
- Install pinned environment: `uv sync --locked` or `python -m pip install -r requirements.txt`.
- Run CRISP unlearning: `python main.py --config-path config-files/crisp-gemma-2b.json --log-level INFO`.
- Evaluate a saved adapter without retraining: `python scripts/evaluate_crisp_adapter.py outputs/crisp/gemma-2b-hp --evaluations hp_mcq mmlu_10`.
- Interactive chat with a CRISP adapter: `python scripts/chat_crisp_adapter.py outputs/crisp/gemma-2b-hp`.
- Generate Ollama-backed behavioural validation: `python scripts/behavioural_validation.py --config-path config-files/behavioural-validation-prompts.json --output-dir outputs/behavioural-validation`.
- Run the chair CRISP experiment: `uv run python main.py --config-path config-files/crisp-chair-gemma-2b-it.json --log-level INFO`.
- Evaluate the chair adapter: `uv run python scripts/evaluate_chair_adapter.py outputs/crisp/gemma-2b-it-chair`.
- No test, lint, typecheck, formatter, CI, or pre-commit config exists; do not invent verification commands.

## Prerequisites & Runtime Gotchas
- Set `HF_TOKEN` environment variable before running CRISP; model/SAE loading will fail without it.
- Requires CUDA GPU with bfloat16 support; models load with `torch_dtype=torch.bfloat16` and `device_map="auto"`.
- SAEs auto-download to `gemma_sae_cache/` or `llama_sae_cache/` on first run (substantial disk space).
- Forget/retain datasets auto-download from HuggingFace (e.g., `WutYee/HarryPotter_books_1to7`, `Blackroot/Tiny-Open-Domain-Books`).
- Feature caches are stored in `crisp_cache/` and **are tracked in Git** for reproducibility; delete to regenerate.
- CRISP outputs LoRA adapters under `outputs/crisp/`; adapters are Git-ignored unless explicitly promoted.
- Behavioural validation requires a local Ollama server running at `http://localhost:11434` (configurable via `--ollama-url`).

## Configs & Models
- CRISP configs: `config-files/crisp-gemma-2b.json` (base), `crisp-sweep-{A,B,C,D}.json` (hyperparameter sweeps).
- Key CRISP parameters: `k_features` (features to ablate per layer), `alpha/beta/gamma` (loss weights), `lora_rank`, `num_epochs`.
- Evaluation benchmarks: `hp_mcq` (Harry Potter multiple-choice), `mmlu_10` (10-shot MMLU retain eval).
- Ollama models for validation: baseline `llama3.2:3B`, UTR `martindisley/unlearning-to-rest:latest`.

## Artifact Rules
- Ignored by `.gitignore`: `outputs/`, `crisp-results/`, `models/`, `*.npz`, `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `gemma_sae_cache/`, `llama_sae_cache/`.
- Tracked: final configs, `data/chair-concept-def.json`, `data/chair-crisp-spec.json`, the curated `data/chair-corpus/*.jsonl`, and feature caches in `crisp_cache/`.
- Do not commit generated model checkpoints, activation caches, or response outputs.
- `data/chair-corpus/generation_log.jsonl` remains ignored; the curated JSONL corpus is intentionally committed for the remote CRISP run.

## Implementation Notes
- CRISP selects SAE layers (e.g., `[4, 6, 8, 10, 12, 14]` for Gemma-2-2B), extracts features distinguishing forget vs. retain sets, then applies interventions during LoRA fine-tuning.
- SAE classes: `JumpReLUSAE` (Gemma), `TopkSae` (Llama-3.1-8B); auto-download via `huggingface_hub.snapshot_download`.
- Evaluation data (`hp_mcq.json`, `mmlu_10.json`) auto-downloads from upstream CRISP repo if missing.
- Legacy pruning pipeline: concept pruning fits L1 logistic regression on activations, vocabulary pruning masks token rows in `lm_head` and input embeddings. Not currently executable via `main.py`.
