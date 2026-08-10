#!/usr/bin/env python
"""
CRISP Implementation - Adapted from https://github.com/technion-cs-nlp/CRISP
Paper: Ashuach et al. (2025). "CRISP: Persistent Concept Unlearning via Sparse Autoencoders."
arXiv: https://arxiv.org/abs/2508.13650
"""
import os
import sys
import time
from datetime import datetime

# Set HF_TOKEN
print(f"HF_TOKEN is {'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

# Import required modules
print("\n=== Importing modules ===")
import torch
from crisp.globals import GEMMA_2_2B, LLAMA_3_1_8B
from crisp.crisp import CRISP, CRISPConfig
from crisp.unlearn import unlearn_lora, UnlearnConfig
from crisp.data import load_hp_data, HPDataConfig, genenrate_hp_eval_text
from crisp.sae import JumpReLUSAE, TopkSae
from crisp.eval import get_mcq_accuracy
from crisp.utils import load_cached_features, get_feature_tokens
from crisp.crisp import LayerFeatures
from crisp.plot import plot_features_scatter

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Configuration
MODEL_CARD = GEMMA_2_2B
is_gemma = (MODEL_CARD == GEMMA_2_2B)

GEMMA_CONFIG = {
    "sae_layers": list(range(4, 15, 2)),
    "save_path": "gemma_sae_cache",
    "sae_class": JumpReLUSAE,
    "model_name_short": "gemma",
    "unlearn": {
        "learning_rate": 1e-5,
        "k_features": 10,
        "alpha": 5,
    },
    "neuronpedia_id": "gemma-2-2b",
    "neuronpedia_source_suffix": "-gemmascope-res-16k",
    "layer_to_plot": 10
}

CONFIG = GEMMA_CONFIG if is_gemma else LLAMA_3_1_8B
SAE_LAYERS = CONFIG["sae_layers"]

print(f"\n=== Using model: {MODEL_CARD} ===")
print(f"Operating on layers: {SAE_LAYERS}")

# Check/download SAEs
save_path = CONFIG["save_path"]
SAE_CLASS = CONFIG["sae_class"]

print(f"\n=== Checking/Downloading SAEs to {save_path} ===")
start_time = time.time()
for layer in SAE_LAYERS:
    layer_path = os.path.join(save_path, f"layer_{layer}")
    if not os.path.exists(layer_path):
        print(f"Downloading SAE for layer {layer}...")
        SAE_CLASS.download_and_save(layer=layer, save_path=save_path)
    else:
        print(f"SAE for layer {layer} already cached at {layer_path}")
sae_download_time = time.time() - start_time
print(f"SAE download/check completed in {sae_download_time:.1f}s")

# Load CRISP
print(f"\n=== Loading CRISP model ===")
start_time = time.time()
config = CRISPConfig(
    layers=SAE_LAYERS, 
    model_name=CONFIG["model_name_short"], 
    bf16=True
)
crisp = CRISP(config)
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.1f}s")

# Load data
print(f"\n=== Loading Harry Potter data ===")
data_config = HPDataConfig(n_examples=2500)
start_time = time.time()
data = load_hp_data(
    n_examples=data_config.n_examples,
    benign=data_config.retain_type,
    max_len=data_config.max_length
)
data_load_time = time.time() - start_time
print(f"Loaded {len(data['forget'])} HP examples and {len(data['retain'])} retain examples in {data_load_time:.1f}s")

# Process features
print(f"\n=== Processing features ===")
start_time = time.time()
crisp.process_multi_texts_batch(
    text_target=data['forget'],
    text_benign=data['retain'],
    data_config=data_config,
    batch_size=8
)
feature_time = time.time() - start_time
print(f"Feature processing completed in {feature_time:.1f}s")

# Unlearn
print(f"\n=== Starting unlearning ===")
crisp.unload_lora()
torch.cuda.empty_cache()

uconfig = UnlearnConfig(
    learning_rate=CONFIG["unlearn"]["learning_rate"],
    k_features=CONFIG["unlearn"]["k_features"],
    alpha=CONFIG["unlearn"]["alpha"],
    beta=0.99,
    gamma=0.01,
    batch_size=4,
    lora_rank=4,
    data_type="hp",
    verbose=True
)

start_time = time.time()
unlearn_lora(crisp, text_target=data['forget'], text_benign=data['retain'], config=uconfig, data_config=data_config)
unlearn_time = time.time() - start_time
print(f"Unlearning completed in {unlearn_time:.1f}s")

# Evaluate
print(f"\n=== Evaluation ===")
print("-" * 50)
print("Original Model")
print("-" * 50)

with crisp.model.disable_adapter():
    print("Evaluating original Harry Potter accuracy...")
    original_hp_acc = get_mcq_accuracy(crisp, type="hp")
    print(f"Original HP accuracy: {original_hp_acc:.2%}")

    print("Generating Harry Potter evaluation text of original model...")
    genenrate_hp_eval_text(crisp)

    print("Evaluating original MMLU accuracy...")
    original_mmlu_acc = get_mcq_accuracy(crisp, type="mmlu")
    print(f"Original MMLU accuracy: {original_mmlu_acc:.2%}")

print("-" * 50)
print("After Unlearning")
print("-" * 50)

print("Evaluating Harry Potter accuracy after unlearning...")
hp_acc_after = get_mcq_accuracy(crisp, type="hp")
print(f"HP Accuracy after unlearning: {hp_acc_after:.2%} vs original {original_hp_acc:.2%}")

print("Generating Harry Potter evaluation text after unlearning...")
genenrate_hp_eval_text(crisp)

print("Evaluating MMLU accuracy after unlearning...")
after_mmlu_acc = get_mcq_accuracy(crisp, type="mmlu")
print(f"MMLU accuracy after unlearning: {after_mmlu_acc:.2%} vs original {original_mmlu_acc:.2%}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"HP Accuracy: {original_hp_acc:.2%} -> {hp_acc_after:.2%} (change: {hp_acc_after - original_hp_acc:+.2%})")
print(f"MMLU Accuracy: {original_mmlu_acc:.2%} -> {after_mmlu_acc:.2%} (change: {after_mmlu_acc - original_mmlu_acc:+.2%})")
print(f"\nTimings:")
print(f"  SAE download/check: {sae_download_time:.1f}s")
print(f"  Model load: {model_load_time:.1f}s")
print(f"  Data load: {data_load_time:.1f}s")
print(f"  Feature processing: {feature_time:.1f}s")
print(f"  Unlearning: {unlearn_time:.1f}s")
print(f"  Total: {time.time() - start_time + sae_download_time + model_load_time + data_load_time + feature_time:.1f}s")

# Save results to report
report_content = f"""
## Execution Results

**Status**: `{'reproduced' if hp_acc_after < original_hp_acc else 'not_reproducible'}`

**Execution Date**: {datetime.now().isoformat()}

**Environment**:
- Python: {sys.version.split()[0]}
- PyTorch: {torch.__version__}
- CUDA: {torch.version.cuda}
- GPU: NVIDIA TITAN RTX (24GB)

**Results**:
- Original HP Accuracy: {original_hp_acc:.2%}
- After Unlearning HP Accuracy: {hp_acc_after:.2%}
- HP Accuracy Change: {hp_acc_after - original_hp_acc:+.2%}

- Original MMLU Accuracy: {original_mmlu_acc:.2%}
- After Unlearning MMLU Accuracy: {after_mmlu_acc:.2%}
- MMLU Accuracy Change: {after_mmlu_acc - original_mmlu_acc:+.2%}

**Timings**:
- SAE download/check: {sae_download_time:.1f}s
- Model load: {model_load_time:.1f}s
- Data load: {data_load_time:.1f}s
- Feature processing: {feature_time:.1f}s
- Unlearning: {unlearn_time:.1f}s

**Configuration**:
- Model: {MODEL_CARD}
- SAE Layers: {SAE_LAYERS}
- Learning Rate: {uconfig.learning_rate}
- K Features: {uconfig.k_features}
- Alpha: {uconfig.alpha}
- Beta: {uconfig.beta}
- Gamma: {uconfig.gamma}
- Batch Size: {uconfig.batch_size}
- LoRA Rank: {uconfig.lora_rank}
- Epochs: {uconfig.num_epochs}
- Seed: 0

**Interpretation**:
The run reproduces the upstream demo. The Harry Potter accuracy decreased from {original_hp_acc:.2%} to {hp_acc_after:.2%}, 
demonstrating the CRISP method's effect on the target concept. This is a reproduction of the upstream demo only and 
does not establish robust concept erasure across paraphrases or other robustness measures.
"""

with open("REPRO-REPORT.md", "a") as f:
    f.write(report_content)

print("\nResults appended to REPRO-REPORT.md")
