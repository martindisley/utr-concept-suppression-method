# Agent Notes

## Repo Shape
- This is a paper-specific reproducibility package, not a general-purpose unlearning library.
- `main.py` is the executable entrypoint; pruning orchestration lives in `src/pruning.py`.
- Saliency extraction/analysis is in `src/neuron_saliency_analysis.py`; low-level mask helpers are in `src/pruning_operations.py`.
- The final study config is `config-files/chair-3B-hybrid-config.json`; the target concept/test prompts are in `data/chair-concept-def.json`.

## Commands
- Install with the pinned historical environment: `python -m pip install -r requirements.txt`.
- Run the pipeline with: `python main.py --config-path config-files/chair-3B-hybrid-config.json --log-level INFO`.
- Generate Ollama-backed behavioural validation responses with: `python scripts/behavioural_validation.py --config-path config-files/behavioural-validation-prompts.json --output-dir outputs/behavioural-validation`; use `config-files/behavioural-validation-full-analysis.json` for the 10-sample-per-model run at temperature `0.7`.
- There is no committed test, lint, typecheck, formatter, CI, or pre-commit config; do not invent repo-standard verification commands.

## Runtime Gotchas
- The committed config uses `meta-llama/Llama-3.2-3B-Instruct` on CUDA and `load_model()` sets `torch_dtype=torch.bfloat16` plus `device_map="auto"`.
- `data/chair-simplified-10k.json` and `data/chair-3B-simplified-activations-10k.npz` are intentionally absent; the sample JSON is schema-only.
- If the activation cache file at the configured path is absent, `Pruner.prune_concept()` regenerates activations from the full examples and base model, which can require substantial GPU memory and disk space.
- Running the pipeline writes `logs/app.log` and, with the final config, saves the model/tokenizer under `models/chair-3B-hybrid`.
- Behavioural validation outputs are written under `outputs/` and are ignored by Git unless explicitly promoted to a tracked artifact.

## Artifact Rules
- Do not commit generated data, activation caches, model checkpoints, GGUF files, safetensors, `.bin`, `.pt`, or `.pth` artifacts; `.gitignore` is set up to keep these out.
- `config-files/` and `data/` are mostly ignored; only the final config, READMEs, concept definition, and sample data are intended to be tracked.

## Implementation Notes
- Vocabulary pruning masks target token rows in both `model.lm_head.weight` and `model.get_input_embeddings().weight`, then `apply_weight_masks()` makes masks permanent before saving.
- Concept pruning selects the last `num_layers` Llama modules whose names contain `layers` and have direct `nn.Linear` children; avoid assuming this supports arbitrary model architectures.
- The config `testing.temperature` value is currently not passed into `generate_text_test()`; generation uses sampling with `top_k=50`.
