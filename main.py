#!/usr/bin/env python
"""
CRISP Implementation - Adapted from https://github.com/technion-cs-nlp/CRISP
Paper: Ashuach et al. (2025). "CRISP: Persistent Concept Unlearning via Sparse Autoencoders."
arXiv: https://arxiv.org/abs/2508.13650
"""
import argparse
import os
import sys
import time
import json
import logging
from datetime import datetime

# Setup logging
def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger()

logger = setup_logging("INFO")

# Set HF_TOKEN
if not os.environ.get('HF_TOKEN'):
    logger.warning("HF_TOKEN environment variable is NOT SET - model loading will fail")
else:
    logger.info("HF_TOKEN is set")

# Import required modules
logger.info("\n=== Importing modules ===")
import torch
from src.crisp_globals import GEMMA_2_2B, LLAMA_3_1_8B
from src.crisp import CRISP, CRISPConfig
from src.crisp_unlearn import unlearn_lora, UnlearnConfig
from src.crisp_data import load_hp_data, HPDataConfig, genenrate_hp_eval_text
from src.crisp_sae import JumpReLUSAE, TopkSae
from src.crisp_eval import get_mcq_accuracy
from src.crisp_utils import load_cached_features, get_feature_tokens
from src.crisp import LayerFeatures

logger.info(f"Torch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main(config_path, log_level):
    """Main CRISP execution function"""
    # Setup logging
    logger = setup_logging(log_level)
    
    # Load configuration
    logger.info(f"\n=== Loading config from {config_path} ===")
    config = load_config(config_path)
    
    # Extract configuration sections
    base_model_path = config["base_model"]["base_model_path"]
    sae_layers = config["sae_config"]["sae_layers"]
    sae_cache_path = config["sae_config"]["sae_cache_path"]
    crisp_cfg = config["crisp_config"]
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    output_cfg = config["output"]
    
    logger.info(f"\n=== Using model: {base_model_path} ===")
    logger.info(f"Operating on layers: {sae_layers}")
    
    # Check/download SAEs
    logger.info(f"\n=== Checking/Downloading SAEs to {sae_cache_path} ===")
    start_time = time.time()
    SAE_CLASS = JumpReLUSAE if "gemma" in base_model_path else TopkSae
    for layer in sae_layers:
        layer_path = os.path.join(sae_cache_path, f"layer_{layer}")
        if not os.path.exists(layer_path):
            logger.info(f"Downloading SAE for layer {layer}...")
            SAE_CLASS.download_and_save(layer=layer, save_path=sae_cache_path)
        else:
            logger.info(f"SAE for layer {layer} already cached at {layer_path}")
    sae_download_time = time.time() - start_time
    logger.info(f"SAE download/check completed in {sae_download_time:.1f}s")
    
    # Load CRISP model
    logger.info(f"\n=== Loading CRISP model ===")
    start_time = time.time()
    model_name_short = "gemma" if "gemma" in base_model_path else "llama"
    crisp_config = CRISPConfig(
        layers=sae_layers, 
        model_name=model_name_short, 
        bf16=True
    )
    crisp = CRISP(crisp_config)
    model_load_time = time.time() - start_time
    logger.info(f"Model loaded in {model_load_time:.1f}s")
    
    # Load data
    logger.info(f"\n=== Loading data ===")
    start_time = time.time()
    data_config = HPDataConfig(
        n_examples=data_cfg["n_examples"],
        retain_type="book"
    )
    data = load_hp_data(
        n_examples=data_config.n_examples,
        benign=data_config.retain_type,
        max_len=data_cfg["max_length"]
    )
    data_load_time = time.time() - start_time
    logger.info(f"Loaded {len(data['forget'])} HP examples and {len(data['retain'])} retain examples in {data_load_time:.1f}s")
    
    # Process features
    logger.info(f"\n=== Processing features ===")
    start_time = time.time()
    crisp.process_multi_texts_batch(
        text_target=data['forget'],
        text_benign=data['retain'],
        data_config=data_config,
        batch_size=8
    )
    feature_time = time.time() - start_time
    logger.info(f"Feature processing completed in {feature_time:.1f}s")
    
    # Unlearn
    logger.info(f"\n=== Starting unlearning ===")
    crisp.unload_lora()
    torch.cuda.empty_cache()
    
    uconfig = UnlearnConfig(
        learning_rate=crisp_cfg["learning_rate"],
        k_features=crisp_cfg["k_features"],
        alpha=crisp_cfg["alpha"],
        beta=crisp_cfg["beta"],
        gamma=crisp_cfg["gamma"],
        batch_size=crisp_cfg["batch_size"],
        lora_rank=crisp_cfg["lora_rank"],
        num_epochs=crisp_cfg["num_epochs"],
        data_type="hp",
        verbose=True
    )
    
    start_time = time.time()
    unlearn_lora(crisp, text_target=data['forget'], text_benign=data['retain'], config=uconfig, data_config=data_config)
    unlearn_time = time.time() - start_time
    logger.info(f"Unlearning completed in {unlearn_time:.1f}s")
    
    # Evaluate
    logger.info(f"\n=== Evaluation ===")
    logger.info("-" * 50)
    logger.info("Original Model")
    logger.info("-" * 50)
    
    with crisp.model.disable_adapter():
        logger.info("Evaluating original Harry Potter accuracy...")
        original_hp_acc = get_mcq_accuracy(crisp, type=eval_cfg["forget_eval"])
        logger.info(f"Original HP accuracy: {original_hp_acc:.2%}")
        
        logger.info("Generating Harry Potter evaluation text of original model...")
        genenrate_hp_eval_text(crisp)
        
        logger.info("Evaluating original MMLU accuracy...")
        original_mmlu_acc = get_mcq_accuracy(crisp, type=eval_cfg["retain_eval"])
        logger.info(f"Original MMLU accuracy: {original_mmlu_acc:.2%}")
    
    logger.info("-" * 50)
    logger.info("After Unlearning")
    logger.info("-" * 50)
    
    logger.info("Evaluating Harry Potter accuracy after unlearning...")
    hp_acc_after = get_mcq_accuracy(crisp, type=eval_cfg["forget_eval"])
    logger.info(f"HP Accuracy after unlearning: {hp_acc_after:.2%} vs original {original_hp_acc:.2%}")
    
    logger.info("Generating Harry Potter evaluation text after unlearning...")
    genenrate_hp_eval_text(crisp)
    
    logger.info("Evaluating MMLU accuracy after unlearning...")
    after_mmlu_acc = get_mcq_accuracy(crisp, type=eval_cfg["retain_eval"])
    logger.info(f"MMLU accuracy after unlearning: {after_mmlu_acc:.2%} vs original {original_mmlu_acc:.2%}")
    
    # Save adapter if configured
    if output_cfg.get("save_adapter", True):
        from src.crisp_utils import save_model
        adapter_path = output_cfg.get("adapter_path", "outputs/crisp/gemma-2b-hp")
        logger.info(f"\n=== Saving adapter to {adapter_path} ===")
        configs = {
            "crisp_config": crisp_config.to_dict(),
            "unlearn_config": uconfig.to_dict(),
            "data_config": data_config.to_dict()
        }
        save_model(crisp.model, configs=configs, path=adapter_path, tokenizer=crisp.tokenizer)
        logger.info(f"Adapter saved to {adapter_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"HP Accuracy: {original_hp_acc:.2%} -> {hp_acc_after:.2%} (change: {hp_acc_after - original_hp_acc:+.2%})")
    logger.info(f"MMLU Accuracy: {original_mmlu_acc:.2%} -> {after_mmlu_acc:.2%} (change: {after_mmlu_acc - original_mmlu_acc:+.2%})")
    logger.info(f"\nTimings:")
    logger.info(f"  SAE download/check: {sae_download_time:.1f}s")
    logger.info(f"  Model load: {model_load_time:.1f}s")
    logger.info(f"  Data load: {data_load_time:.1f}s")
    logger.info(f"  Feature processing: {feature_time:.1f}s")
    logger.info(f"  Unlearning: {unlearn_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CRISP concept unlearning")
    parser.add_argument('--config-path', type=str, required=True, 
                        help='Path to the configuration JSON file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()
    
    main(args.config_path, args.log_level)
