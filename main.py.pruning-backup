import argparse
from src.pruning import Pruner
from src.logger import setup_logging, log_config
from src.utils import load_config, load_model, load_data
from src.utils import generate_text_test
from src.pruning_operations import apply_weight_masks
import logging

def main(config_path, log_level):
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger()
    # Load the configuration file
    config = load_config(config_path)
    log_config(config)

    try:
        # Load the model and tokenizer
        model, tokenizer = load_model(config["base_model"]["base_model_path"])    
        # Load the target words and test prompts
        concept_definition, test_prompts = load_data(config)
    except Exception as e:
        logger.error(f"Error loading model and data: {e}")
        raise

    try:
        # Load the model pruner
        pruner = Pruner(model, tokenizer, concept_definition, config)
        if config["neural_pruning"]["prune_concept"]:
            # Prune the model network for concpt nuerons
            pruner.prune_concept()
        if config["neural_pruning"]["prune_vocabulary"]:
            # Prune the model vocabulary
            pruner.prune_vocabulary()
    except Exception as e:
        logger.error(f"Error pruning the model: {e}")
        raise

    try:
        generate_text_test(
            pruner.model, 
            tokenizer, 
            test_prompts, 
            max_tokens=config["testing"]["max_length"], 
            num_sequences=config["testing"]["num_return_sequences"]
        )
    except Exception as e:
        logger.error(f"Error generating test text: {e}")
        raise

    if config.get("output", {}).get("make_pruning_permanent", True):
        logger.info("Applying weight masks to make pruning permanent...")
        apply_weight_masks(pruner.model)

    if config["output"]["output_model_path"]:
        try:
            # Save the ablated model
            output_model_path = config["output"]["output_model_path"]
            pruner.model.save_pretrained(output_model_path)
            tokenizer.save_pretrained(output_model_path)
            logger.info(f"Ablated model saved to {output_model_path}")
        except Exception as e:
            logger.error(f"Error saving the ablated model: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model pruning and testing script.")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--log-level', type=str, help='Logging level.', default='INFO')
    args = parser.parse_args()
    main(args.config_path, args.log_level)
