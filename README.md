# Unlearning to Rest concept suppression method

This repository is a paper-specific reproducibility package for the Unlearning
to Rest model construction method.

Unlearning to Rest is a hybrid concept suppression intervention for a Llama
instruction model. The study model presented in the paper combined concept
saliency pruning, vocabulary pruning, and a later repair fine-tuning stage. The
released model is available through Ollama as `martindisley/unlearning-to-rest`.

## Repository status

This repository does not present a general-purpose unlearning tool. It is a
reviewer-facing reproducibility package for the method and model used in the
paper. It includes the code, final configuration, target concept definition, and
sample data schema needed to inspect and rerun the pipeline. It does not include
large derived artefacts such as activation caches or model checkpoints.

## Final study configuration

The final 3B hybrid configuration is included at:

```text
config-files/chair-3B-hybrid-config.json
```

Key values:

- base model: `meta-llama/Llama-3.2-3B-Instruct`
- target terms: `chair`, `chairs`, `sit`, `sitting`, `seat`, `seats`
- examples file: `data/chair-simplified-10k.json`
- activation cache: `data/chair-3B-simplified-activations-10k.npz`
- concept pruning: enabled
- vocabulary pruning: enabled
- inspected layers: `8`
- pruning fraction: `0.0085`
- output path: `models/chair-3B-hybrid`

The concept definition used by this configuration is included at:

```text
data/chair-concept-def.json
```

## Method summary

The pipeline has four stages.

1. **Example construction**
   - Build or supply a JSON file with `concept_examples` and
     `background_examples`.
   - The final study configuration used `chair-simplified-10k.json`, with
     10,017 concept examples and 10,000 background examples.

2. **Concept saliency pruning**
   - Load the base model and target/background examples.
   - Extract mean activations from selected Llama layers.
   - Fit an L1-regularised logistic regression model to distinguish target
     concept activations from background activations.
   - Use the learned coefficient magnitudes as saliency scores.
   - Map salient activation dimensions to rows in linear submodule weight
     matrices and apply PyTorch pruning masks.

3. **Vocabulary pruning**
   - Tokenise the target terms and search the tokenizer vocabulary for entries
     containing the target strings.
   - Mask corresponding rows in the language modelling head.
   - Mask corresponding rows in the input embedding layer.
   - Apply masks permanently before export.

4. **Repair fine-tuning**
   - Fine-tune the hybrid-pruned model on an instruction dataset filtered to
     remove the target terms.
   - The preserved repair script used the same target term list to filter
     `mlabonne/FineTome-100k`, then applied response-only supervised
     fine-tuning with LoRA via Unsloth.

## Running the pruning pipeline

Install dependencies in an isolated environment. The historical environment was
CUDA-based and used Transformers, PyTorch, scikit-learn, SciPy, and safetensors.

```bash
python main.py --config-path config-files/chair-3B-hybrid-config.json --log-level INFO
```

The command expects the full examples file at the path specified in the config.
The small sample file in this repository is only a schema example and is not
the full training/evaluation data.

If the activation cache specified in the config is missing, the code can
regenerate it from the examples and base model. This may require substantial
GPU memory and disk space.

## Large artefacts

The following artefacts are intentionally not committed:

- model checkpoints;
- GGUF model files;
- full example datasets where licensing or size makes inclusion inappropriate;
- activation caches, including `chair-3B-simplified-activations-10k.npz`.

The activation cache is a large derived intermediate artefact. It is generated
from the base model, final config, example data, and activation extraction code.
For reproducibility, regenerate it rather than storing it in Git.

## Inference behaviour

The tokenizer is unchanged. User input is still processed by the standard
Llama tokenizer, so some target terms can be direct tokens and others can be
split into subword pieces. The vocabulary pruning stage masks target token
pathways in the language modelling head and input embedding layer, but it does
not make all semantically related concepts impossible. In practice, the model
can show reformulation and substitution through adjacent terms such as
`furniture`, `rest`, `support`, and `object`. Malformed text is a possible side
effect of aggressive pruning and should be treated as a limitation rather than
the intended interactional effect.

## Model access

The public study model is available through Ollama:

```bash
ollama run martindisley/unlearning-to-rest
```

Model page:

```text
https://ollama.com/martindisley/unlearning-to-rest
```

## Behavioural validation responses

Matched baseline and Unlearning to Rest responses for appendix/rebuttal drafting
can be generated through the local Ollama API without loading model weights in
Python:

```bash
python scripts/behavioural_validation.py --config-path config-files/behavioural-validation-prompts.json --output-dir outputs/behavioural-validation
```

The fuller validation run uses ten baseline and ten Unlearning to Rest samples
per prompt at temperature `0.7`:

```bash
python scripts/behavioural_validation.py --config-path config-files/behavioural-validation-full-analysis.json --output-dir outputs/behavioural-validation-full-analysis
```

Prompt/model/generation defaults are stored in `config-files/`. Generated
response files are written under `outputs/` and are not intended to be committed
by default.

## Limitations

- This is approximate concept suppression, not certified data deletion.
- Token string matching can catch substrings beyond the intended concept.
- Concept pruning and vocabulary pruning can affect non-target capabilities.
- Repair fine-tuning may indirectly restore some target associations.
- Large activation caches and model weights are not included in this repository.

## Citation

This repository supports the paper:

> Representational Emendation: Designing Cognitive Friction with Subtractive
> Fine-Tuning in Large Language Models.

Full citation details will be added after publication.

---

## CRISP Implementation (Branch: crisp-implementation)

This branch implements the CRISP method for comparison:

- **Original Repository:** https://github.com/technion-cs-nlp/CRISP
- **Paper:** Ashuach et al. (2025). "CRISP: Persistent Concept Unlearning via Sparse Autoencoders." arXiv:2508.13650
- **Authors:** Tomer Ashuach, Dana Arad, Aaron Mueller, Martin Tutek, Yonatan Belinkov (Technion)

### Running CRISP

```bash
python main.py --config-path config-files/crisp-gemma-2b.json --log-level INFO
```

### CRISP Citation

```bibtex
@article{ashuach2025crisp,
  title={CRISP: Persistent Concept Unlearning via Sparse Autoencoders},
  author={Ashuach, Tomer and Arad, Dana and Mueller, Aaron and Tutek, Martin and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2508.13650},
  year={2025},
  url={https://arxiv.org/abs/2508.13650},
}
```
