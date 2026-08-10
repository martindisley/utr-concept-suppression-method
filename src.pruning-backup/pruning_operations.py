import logging
import torch
import torch.nn as nn

# Create a logger for this module
logger = logging.getLogger(__name__)

def get_vocabulary_indexes(tokenizer, targets):
    """
    Get the indexes of vocabulary tokens that match the target words.
    This function tokenizes the target words using the provided tokenizer and 
    finds all vocabulary entries that contain the target words. It returns a 
    list of unique token indexes that either match the target words directly 
    or contain them as substrings.
    Args:
        tokenizer: A tokenizer object that provides methods `encode`, 
                   `convert_ids_to_tokens`, and supports iteration over its 
                   vocabulary.
        targets (list of str): A list of target words to search for in the 
                               tokenizer's vocabulary.
    Returns:
        list of int: A list of unique token indexes that match or contain the 
                     target words.
    """
    # Check how each string is tokenized
    tokenized_indexes = []
    for test_str in targets:
        tokens = tokenizer.encode(test_str)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        logger.debug(f"'{test_str}' → {list(zip(tokens, token_strings))}")
        tokenized_indexes.extend(tokens)

    # Find all vocabulary entries that contain target words
    related_tokens = []
    for i in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(i)
        for target in targets:
            if target.lower() in token.lower():
                related_tokens.append((i, token))
                logger.debug(f"Related token found - ID: {i}, Token: '{token}'")
                break  # No need to check other targets if one matches

    token_indexes = list(set(tokenized_indexes + [token_id for token_id, _ in related_tokens]))
    
    logger.debug("\nAll vocabulary tokens containing target words:")
    for token_id in token_indexes:
        token = tokenizer.convert_ids_to_tokens(token_id)
        logger.debug(f"ID: {token_id}, Token: '{token}'")

    return token_indexes

def get_layers(model, num_layers):
    """
    Retrieve the names of the top `num_layers` layers from a given model that 
    contain at least one `nn.Linear` submodule.
    Args:
        model (torch.nn.Module): The neural network model from which to retrieve layer names.
        num_layers (int): The number of top layers to retrieve.
    Returns:
        list: A list of layer names that contain at least one `nn.Linear` submodule.
    """
    layer_names = [
        name for name, module in model.named_modules()
        if 'layers' in name and isinstance(module, nn.Module) and 
        any(isinstance(submodule, nn.Linear) 
            for submodule in module.children())
    ]
    
    # Select the top `num_layers` layers
    layer_names = layer_names[-num_layers:]
    return layer_names

def apply_weight_masks(model):
    """
    Apply weight masks to the model's weights to make pruning permanent.
    """
    for name, module in model.named_modules():
        if hasattr(module, "weight_mask"):
            with torch.no_grad():
                module.weight.data *= module.weight_mask  # Apply the mask
            del module.weight_mask  # Remove the mask after applying it
            logging.info(f"Applied and removed weight_mask from {name}")
        if hasattr(module, "weight_orig"):
            # Replace weight_orig with the pruned weight and delete weight_orig
            del module.weight_orig
            logging.info(f"Removed weight_orig from {name}")