"""Token position functions for the MCQA task.

This module provides functions to locate specific tokens in MCQA prompts,
such as answer symbols, periods, and the last token.
"""

import re
from neural.LM_units import TokenPosition, get_last_token_index, get_substring_token_ids


def get_symbol_index(input_sample, pipeline, index):
    """
    Find the index of the correct answer symbol in the prompt.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline
        index: The index of the symbol to find

    Returns:
        list[int]: List containing the index of the correct answer symbol token
    """
    prompt = input_sample["raw_input"]

    # Find all single uppercase letters in the prompt
    matches = list(re.finditer(r"\b[A-Z]\b", prompt))

    # Use position-based indexing instead of dictionary lookup
    # During interventions, input_sample dict may not match the prompt
    if index >= len(matches):
        raise ValueError(f"Symbol index {index} out of range in prompt: {prompt}")

    symbol_match = matches[index]

    # Use helper function to get token indices for the substring up to and including the symbol
    substring = prompt[:symbol_match.end()]
    token_indices = get_substring_token_ids(prompt, substring, pipeline)

    # Return the last token index (which corresponds to the symbol itself)
    return [token_indices[-1]]


def create_symbol_token_position(pipeline, index):
    """Create a TokenPosition for the correct answer symbol."""
    return TokenPosition(
        lambda x, **kwargs: get_symbol_index(x, pipeline, index),
        pipeline,
        id=f"symbol{index}"
    )


def create_symbol_period_token_position(pipeline, index):
    """Create a TokenPosition for the period after the correct answer symbol."""
    return TokenPosition(
        lambda x, **kwargs: [get_symbol_index(x, pipeline, index)[0] + 1],
        pipeline,
        id=f"symbol{index}_period"
    )


def get_correct_symbol_index(input_sample, pipeline):
    """
    Find the index of the correct answer symbol in the prompt by parsing it.

    Args:
        input_sample (Dict): The input dictionary to a causal model
        pipeline: The tokenizer pipeline

    Returns:
        list[int]: List containing the index of the correct answer symbol token
    """
    prompt = input_sample["raw_input"]

    # Parse correct answer from prompt: "The X is Y. What color..."
    first_sentence = prompt.split('.')[0]
    parts = first_sentence.split(' is ')
    if len(parts) != 2:
        raise ValueError(f"Could not parse color from prompt: {prompt}")

    correct_color = parts[1].strip()

    # Find all symbols and their choices
    symbol_matches = list(re.finditer(r"\b[A-Z]\b", prompt))
    choice_pattern = r"\b([A-Z])\.\s+(\w+)"
    choice_matches = list(re.finditer(choice_pattern, prompt))

    # Find which symbol has the correct color
    correct_symbol = None
    for match in choice_matches:
        symbol = match.group(1)
        choice = match.group(2)
        if choice == correct_color:
            correct_symbol = symbol
            break

    if not correct_symbol:
        # Correct answer not in choices (counterfactual) - use first symbol
        correct_symbol = prompt[symbol_matches[0].start():symbol_matches[0].end()] if symbol_matches else None
        if not correct_symbol:
            raise ValueError(f"No symbols found in prompt: {prompt}")

    # Find position of correct symbol
    symbol_match = None
    for match in symbol_matches:
        if prompt[match.start():match.end()] == correct_symbol:
            symbol_match = match
            break

    if not symbol_match:
        raise ValueError(f"Could not find symbol {correct_symbol} in prompt: {prompt}")

    # Use helper function to get token indices for the substring up to and including the symbol
    substring = prompt[:symbol_match.end()]
    token_indices = get_substring_token_ids(prompt, substring, pipeline)

    # Return the last token index (which corresponds to the symbol itself)
    return [token_indices[-1]]


def create_correct_symbol_token_position(pipeline):
    """Create a TokenPosition for the correct answer symbol."""
    return TokenPosition(
        lambda x, **kwargs: get_correct_symbol_index(x, pipeline),
        pipeline,
        id="correct_symbol"
    )


def create_correct_symbol_period_token_position(pipeline):
    """Create a TokenPosition for the period after the correct answer symbol."""
    return TokenPosition(
        lambda x, **kwargs: [get_correct_symbol_index(x, pipeline)[0] + 1],
        pipeline,
        id="correct_symbol_period"
    )


def create_last_token_position(pipeline):
    """Create a TokenPosition for the last token in the input."""
    return TokenPosition(
        lambda x, **kwargs: get_last_token_index(x, pipeline),
        pipeline,
        id="last_token"
    )


def create_token_positions(pipeline):
    """
    Create all token positions for the MCQA task.

    Args:
        pipeline: The tokenizer pipeline

    Returns:
        dict: Dictionary mapping token position names to TokenPosition objects
    """
    from .causal_models import NUM_CHOICES

    token_positions = {
        "correct_symbol": create_correct_symbol_token_position(pipeline),
        "correct_symbol_period": create_correct_symbol_period_token_position(pipeline),
        "last_token": create_last_token_position(pipeline),
    }

    # Add symbol positions for each choice
    for i in range(NUM_CHOICES):
        token_positions[f"symbol{i}"] = create_symbol_token_position(pipeline, i)
        token_positions[f"symbol{i}_period"] = create_symbol_period_token_position(pipeline, i)

    return token_positions
