from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch
import json
import os

# Setup logging
logger = logging.getLogger()

# Load the model and tokenizer
def load_model(model_id):
    """
    Loads a pre-trained causal language model and its corresponding tokenizer.

    This function retrieves the model and tokenizer using their respective 'from_pretrained'
    methods from the Hugging Face transformers library. The model is loaded with bfloat16 precision 
    and automatically mapped to available devices.

    Parameters:
        model_id (str): The identifier or path of the pre-trained model to load.

    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The loaded causal language model.
            - tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Model and tokenizer loaded for model_id: {model_id}")
    return model, tokenizer

def load_data(config):
    """
    Load concept definitions and test prompts from a JSON file specified in the configuration.

    Args:
        config (dict): Configuration dictionary containing the path to the concept definition file.

    Returns:
        tuple: A tuple containing:
            - target_words (list): List of target words from the concept definition.
            - test_prompts (list): List of test prompts from the concept definition.
    """
    with open(config['neural_pruning']['concept_definition'], "r") as file:
        concept_def = json.load(file)
        target_words = concept_def["concept-definition"]
        test_prompts = concept_def["test-prompts"]
        logger.info("Concept definition and test prompts loaded.")
    return target_words,test_prompts

def load_examples(config):
        """
        Load concept and background examples from a JSON file specified in the configuration.

        Args:
            config (dict): Configuration dictionary containing the path to the examples JSON file.
                           The path should be under the key "neural_pruning" -> "examples".

        Returns:
            tuple: A tuple containing two lists:
                   - concept_examples (list)
                   - background_examples (list)
        """
        with open(config["neural_pruning"]["examples"], 'r') as f:
            examples = json.load(f)
        concept_examples = examples.get('concept_examples', [])
        background_examples = examples.get('background_examples', [])
        return concept_examples, background_examples


# Prepare the prompt and generate multiple sequences
def generate_text_test(model, tokenizer, prompts, max_tokens=50, num_sequences=4):
    """
    Generates multiple sequences of text based on a list of prompts using a specified model and tokenizer.

    Args:
        model (PreTrainedModel): The language model used for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the prompts and decode the generated text.
        prompts (list of str): A list of initial text prompts to generate text from.
        undesired_indices (list): A list of token indices that should be avoided in the generated text.
        max_tokens (int, optional): The maximum number of new tokens to generate for each sequence. Defaults to 50.
        num_sequences (int, optional): The number of sequences to generate for each prompt. Defaults to 10.

    Returns:
        None: The function logs the generated text sequences for each prompt.
    """
    for prompt in prompts:
        tokenizer_output = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenizer_output.input_ids.to(model.device)
        attention_mask = tokenizer_output.attention_mask.to(model.device)

        outputs = model.generate(
            input_ids, 
            max_new_tokens=max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            num_return_sequences=num_sequences,
            do_sample=True,
            top_k=50)

        log_message = (
                "========================================\n"
                f"Test Prompt {prompt}\n"
                "========================================\n"
            )
        logger.info(log_message)
        generated_texts = [tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True) for output in outputs]
        for i, text in enumerate(generated_texts, start=1):
            log_message = (
            f"Generated Text {i}:\n"
            "----------------------------------------\n"
            f"{text}\n"
            "----------------------------------------"
            )
            logger.info(log_message)


def generate_activations_file_path(config):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    dataset_name = config["neural_pruning"]["data"].split("/")[-1].split(".")[0]
    return f"data/{dataset_name}_{model_name}_activations.npz"

def generate_model_path(config, model_type):
    model_name = config["base_model"]["base_model_path"].replace("/", "-")
    if model_type == "pruned":
        return f"models/{model_name}{model_type}"
    elif model_type == "retrained":
        return f"models/{model_name}{model_type}_{config['retraining']['num_train_epochs']}_epochs"

def load_config(config_path):
    # os.makedirs("data/activations/", exist_ok=True)
    os.makedirs("logs/", exist_ok=True)
    # os.makedirs("models/", exist_ok=True)

    with open(config_path) as config_file:
        config = json.load(config_file)

    if config['neural_pruning']['activations_file_path'] == None:
        config['neural_pruning']['compute_activations'] = True
    else:
        config['neural_pruning']['compute_activations'] = False
    # config["neural_pruning"]["pruned_model_path"] = generate_model_path(config, "pruned")
    # config["neural_pruning"]["activations_file_path"] = generate_activations_file_path(config)
    return config