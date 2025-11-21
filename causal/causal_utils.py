"""Utility functions for working with causal models."""

import copy
from typing import Callable, Dict, List, Union

import numpy as np
import torch

from causal.counterfactual_dataset import CounterfactualDataset


class Checker:
    """Base class for checking if intervention/attribution output is correct."""

    def __call__(self, output_or_score, expected_label=None, is_intervention: bool = True, **kwargs) -> float:
        """
        Check correctness based on method type.

        Args:
            output_or_score: For interventions: output_dict with model outputs
                           For attribution: metric score (e.g., from cheap_argmax)
            expected_label: Expected output from causal model (only used for interventions)
            is_intervention: True for regular interventions, False for attribution patching
            **kwargs: Additional arguments passed to check_intervention or check_attribution

        Returns:
            Score (typically 0 or 1) indicating correctness
        """
        if is_intervention:
            return self.check_intervention(output_or_score, expected_label)
        else:
            return self.check_attribution(output_or_score, **kwargs)

    def check_intervention(self, output_dict, expected_label) -> float:
        """Check if regular intervention output matches expected label."""
        raise NotImplementedError("Subclasses must implement check_intervention")

    def check_attribution(self, metric_score) -> float:
        """Check if attribution patching score indicates correctness."""
        raise NotImplementedError("Subclasses must implement check_attribution")


class StringMatchChecker(Checker):
    """Checker that uses string matching for intervention correctness."""

    def check_intervention(self, output_dict, expected_label) -> float:
        """
        Check if expected label appears in output string or vice versa.

        This is the default logic used in compute_interchange_scores.
        """
        output_str = output_dict["string"]
        result = expected_label in output_str or output_str in expected_label
        return 1.0 if result else 0.0

    def check_attribution(self, *args, **kwargs) -> float:
        """Not implemented for string matching."""
        raise NotImplementedError("StringMatchChecker only supports interventions")


class ExactMatchChecker(Checker):
    """Checker that uses exact string equality."""

    def check_intervention(self, output_dict, expected_label) -> float:
        """Check if output exactly matches expected label."""
        return 1.0 if output_dict["string"] == expected_label else 0.0

    def check_attribution(self, *args, **kwargs) -> float:
        """Not implemented for exact matching."""
        raise NotImplementedError("ExactMatchChecker only supports interventions")


class CheapArgmaxChecker(Checker):
    """Checker that uses cheap_argmax for attribution patching correctness."""

    def check_intervention(self, output_dict, expected_label) -> float:
        """Not supported - CheapArgmaxChecker is only for attribution patching."""
        raise NotImplementedError("CheapArgmaxChecker only supports attribution patching")

    @staticmethod
    def compute_score(
        logits: torch.Tensor, correct: torch.Tensor, other_choices: torch.Tensor
    ):
        """
        Compute difference between correct choice logit and other choice logits.

        Args:
            logits: Logits tensor, shape [batch_size, vocab_size]
            correct: Correct token index, shape [batch_size]
            other_choices: Other choice token indices, shape [batch_size, n_choices]

        Returns:
            tuple: (metric, sum_other_logits) where
                - metric = logits[correct] - sum(logits[other_choices])
                - sum_other_logits = sum(logits[other_choices])
        """
        # logits: [batch_size, vocab_size]
        # correct: [batch_size]
        # other_choices: [batch_size, n_choices]

        assert (logits.argmax(-1) == correct).all(), \
            f"Logits don't argmax to correct tokens. This means the model isn't predicting correctly on the base run."
        correct_logits = logits.gather(-1, correct.unsqueeze(-1)).squeeze(
            -1
        )  # [batch_size]

        # Gather all other choice logits: [batch_size, n_choices]
        other_logits = logits.gather(-1, other_choices)
        sum_other_logits = other_logits.sum(dim=-1)  # [batch_size]

        # Return metric and sum of other logits
        return other_logits.shape[
            -1
        ] * correct_logits - sum_other_logits, sum_other_logits

    @torch.no_grad()
    def check_attribution(self, attribution_score, **kwargs) -> float:
        """
        Check if attribution score indicates correct prediction.

        The attribution_score (approx) approximates: logit_cf[correct]
        We patch this into base logits and check if argmax is correct.

        Args:
            attribution_score: Approximated correct logit tensor [batch_size]
            **kwargs: Must contain:
                - logits: base logits tensor [batch_size, vocab_size]
                - correct: correct token index [batch_size]

        Returns:
            Tensor of shape [batch_size] with 1.0 where argmax(patched_logits) == correct, 0.0 otherwise
        """
        logits = kwargs.get("logits")
        correct_index = kwargs.get("correct")

        if logits is None or correct_index is None:
            raise ValueError("check_attribution requires 'logits' and 'correct' in kwargs")

        # Batched processing
        # attribution_score: [batch_size]
        # correct_index: [batch_size]
        # logits: [batch_size, vocab_size]
        logits = logits.clone()  # Don't modify the original
        # Replace the correct token's logit with the attribution score
        patched_logits = logits.scatter(
            dim=-1,
            index=correct_index.unsqueeze(-1),
            src=attribution_score.unsqueeze(-1)
        )

        # Check if the correct token has highest logit
        predictions = torch.argmax(patched_logits, dim=-1)
        return (predictions == correct_index).float()


def can_distinguish_with_dataset(
    dataset,
    causal_model1,
    target_variables1,
    causal_model2=None,
    target_variables2=None
):
    """
    Check if two causal models can be distinguished using interchange interventions
    on a counterfactual dataset.

    Compares the outputs from running interchange interventions with target_variables1
    on causal_model1 against either:
    - Interchange interventions with target_variables2 on causal_model2 (if provided)
    - The forward pass output of causal_model1 (if causal_model2 is None)

    Parameters:
    -----------
    dataset : Dataset
        Dataset containing "input" and "counterfactual_inputs" fields.
    causal_model1 : CausalModel
        The first causal model to run interchange interventions on.
    target_variables1 : list
        List of variable names to use for interchange in the first model.
    causal_model2 : CausalModel, optional
        The second causal model to compare against (default is None).
    target_variables2 : list, optional
        List of variable names to use for interchange in the second model.
        Only used if causal_model2 is provided (default is None).

    Returns:
    --------
    dict
        A dictionary containing:
            - "proportion": The proportion of examples where outputs differ
            - "count": The number of examples where outputs differ
    """
    count = 0
    for example in dataset:
        input_data = example["input"]
        counterfactual_inputs = example["counterfactual_inputs"]
        assert len(counterfactual_inputs) == 1

        # Run interchange intervention on first model
        setting1 = causal_model1.run_interchange(
            input_data,
            {var: counterfactual_inputs[0] for var in target_variables1}
        )

        if causal_model2 is not None and target_variables2 is not None:
            # Run interchange intervention on second model
            setting2 = causal_model2.run_interchange(
                input_data,
                {var: counterfactual_inputs[0] for var in target_variables2}
            )
            if setting1["raw_output"] != setting2["raw_output"]:
                count += 1
        else:
            # Compare against forward pass of first model
            if setting1["raw_output"] != causal_model1.run_forward(input_data)["raw_output"]:
                count += 1

    proportion = count / len(dataset)
    print(f"Can distinguish between {target_variables1} and {target_variables2}: {count} out of {len(dataset)} examples")
    print(f"Proportion of distinguishable examples: {proportion:.2f}")
    return {"proportion": proportion, "count": count}

def statement_conjunction_function(filled_statements: List, delimiters: list) -> str:
    """
    Combine multiple filled statements into a single conjunction.

    Args:
        filled_statements: List of filled statement strings
        delimiters: List of delimiters to use between statements
    
    Returns:
        A single string combining all statements with proper punctuation, seen below:
        "Statement one delimiter one statement two delimiter two ... statement N delimiter N+1."

    """
    #Capitalize first letter and ensure it ends with a period.
    fill_index = delimiters.index("FILL")
    filler = delimiters[fill_index-1]
    new_delimiters = delimiters[:fill_index-1] + delimiters[fill_index+1:]
    for _ in range(len(filled_statements) - len(new_delimiters)):
        new_delimiters.insert(fill_index-1, filler)
    
    if len(new_delimiters) > len(filled_statements):
        new_delimiters = new_delimiters[-len(filled_statements):]

    statements = []
    for i in range(len(filled_statements)):
        statement = filled_statements[i]
        # Decompose into words
        words = statement.split()
        # Capitalize first letter and ensure it ends with a period.
        words[0] = words[0].capitalize()
        statements.append(' '.join(words).rstrip(new_delimiters[-1]))
    conjunction = statements[0] 
    for i in range(1, len(statements)):
        conjunction += new_delimiters[i-1] + statements[i]
    conjunction += new_delimiters[-1]
    return conjunction


def _detach_tensors(obj):
    """Recursively detach PyTorch tensors in nested structures."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {key: _detach_tensors(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_detach_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_detach_tensors(item) for item in obj)
    else:
        return obj


def compute_attribution_scores(
    raw_results: Dict,
    causal_model,
    datasets: Union[Dict, "CounterfactualDataset"],
    target_variables_list: List[List[str]],
    checker: Callable,
    pipeline = None,
) -> Dict:
    """
    Compute attribution scores for target variables using checker.

    Similar to compute_interchange_scores, but for attribution patching.
    Uses the stored metric scores and inputs to evaluate correctness per target variable.

    IMPORTANT: This function checks whether the approximated intervened output matches
    the COUNTERFACTUAL answer (not the base answer). This is consistent with how
    interchange interventions work: we patch counterfactual activations into the base
    run and check if the model now predicts the counterfactual answer.

    Args:
        raw_results: Dictionary from perform_attribution_patching containing
                    metric_scores and inputs for each example
        causal_model: CausalModel used to generate expected outputs
        datasets: Dictionary mapping dataset names to CounterfactualDataset objects,
                 or single CounterfactualDataset (will be converted to dict)
        target_variables_list: List of target variable groups to evaluate
        checker: Checker instance to evaluate correctness (should support is_intervention=False)
        pipeline: Pipeline with tokenizer to convert counterfactual labels to token IDs

    Returns:
        Dictionary with same structure as compute_interchange_scores:
            results["dataset"][dataset_name]["model_unit"][unit_str][target_var_str] = {
                "scores": [...],
                "average_score": float
            }
    """
    # Convert single dataset to dictionary
    if isinstance(datasets, CounterfactualDataset):
        datasets = {datasets.id: datasets}

    # Detach tensors before deep copying (tensors with gradients can't be deep copied)
    raw_results_detached = _detach_tensors(raw_results)
    
    # Create a deep copy to avoid modifying the input
    results = copy.deepcopy(raw_results_detached)

    # Process each dataset and model unit combination
    for dataset_name in datasets.keys():
        if dataset_name not in results["dataset"]:
            continue

        for model_units_str, model_unit_data in results["dataset"][dataset_name][
            "model_unit"
        ].items():
            if model_unit_data is None:
                continue

            # Get stored metric scores and inputs from attribution patching
            attribution_data = model_unit_data.get("attribution_data")
            if attribution_data is None or len(attribution_data) == 0:
                continue

            # Get the first unit's data (they all have the same inputs/approx scores)
            unit_data = attribution_data[0][0]
            inputs = unit_data.get("inputs", [])
            approx_scores = unit_data.get("approx", None)
            logits = unit_data.get("logits", [])
            correct_token_ids = unit_data.get("correct_token_ids", [])

            # Fall back to metric_scores if approx not available (backward compatibility)
            if approx_scores is None:
                approx_scores = unit_data.get("metric_scores", [])

            if not inputs:
                continue

            # Convert approx to list if it's a tensor
            if isinstance(approx_scores, torch.Tensor):
                approx_scores = approx_scores.cpu().tolist()

            if not approx_scores:
                continue

            # Evaluate for each target variable
            for target_variables in target_variables_list:
                target_variable_str = "-".join(target_variables)

                # Generate expected outputs from causal model
                labeled_data = causal_model.label_counterfactual_data(
                    datasets[dataset_name], target_variables
                )

                # Validate alignment
                assert len(labeled_data) == len(inputs), (
                    f"Length mismatch: {len(labeled_data)} vs {len(inputs)}"
                )
                assert len(labeled_data) == len(approx_scores), (
                    f"Length mismatch: {len(labeled_data)} vs {len(approx_scores)}"
                )

                # Compute correctness scores using checker (batched)
                # Convert to batched tensors
                approx_tensor = torch.tensor(approx_scores) if not isinstance(approx_scores, torch.Tensor) else approx_scores
                logits_tensor = torch.stack(logits)

                # Get COUNTERFACTUAL correct tokens (what the model should output after patching)
                if pipeline is not None:
                    # Extract counterfactual labels and convert to token IDs
                    counterfactual_labels = [example["label"] for example in labeled_data]
                    counterfactual_token_ids = [
                        pipeline.tokenizer.encode(label, add_special_tokens=False)[0]
                        for label in counterfactual_labels
                    ]
                    counterfactual_tensor = torch.tensor(counterfactual_token_ids)
                else:
                    # Fallback to base correct tokens (old behavior)
                    counterfactual_tensor = torch.stack(correct_token_ids)

                # Call checker once with batched inputs
                scores_tensor = checker(
                    approx_tensor,
                    is_intervention=False,
                    logits=logits_tensor,
                    correct=counterfactual_tensor  # Check against counterfactual answer!
                )

                # Convert to list
                scores = scores_tensor.cpu().tolist() if isinstance(scores_tensor, torch.Tensor) else list(scores_tensor)

                # Store processed results
                results["dataset"][dataset_name]["model_unit"][model_units_str][
                    target_variable_str
                ] = {"scores": scores, "average_score": np.mean(scores)}

    return results


def compute_interchange_scores(
    raw_results: Dict,
    causal_model,
    datasets: Union[Dict, 'CounterfactualDataset'],
    target_variables_list: List[List[str]],
    checker: Callable
) -> Dict:
    """
    Process raw intervention results by computing scores for target variables.

    This function takes the raw outputs from perform_interventions and adds
    target-variable-specific score fields to the results dictionary. It matches
    the exact data structure that perform_interventions would create if
    target_variables_list was passed directly, allowing all existing visualization
    code to work without changes.

    This separation allows you to:
    1. Run expensive interventions once
    2. Analyze results with different target_variables combinations
    3. Experiment with different causal model interpretations post-hoc

    Args:
        raw_results: Dictionary from perform_interventions containing:
            - raw_outputs: Model generation outputs (sequences, scores, strings)
            - causal_model_inputs: Base inputs and counterfactual inputs for each example
            - metadata: Model unit metadata (layer, position, etc.)
            - feature_indices: Selected features for each model unit
        causal_model: CausalModel used to generate expected outputs via label_counterfactual_data
        datasets: Dictionary mapping dataset names to CounterfactualDataset objects,
                 or single CounterfactualDataset (will be converted to dict)
        target_variables_list: List of target variable groups to evaluate.
                              Each group is a list of variable names to intervene on.
        checker: Function with signature (output_dict, expected_label) -> score
                Used to compare model outputs against causal model expectations.

    Returns:
        Dictionary with same structure as raw_results, but with added fields for each
        target variable group:
            results["dataset"][dataset_name]["model_unit"][unit_str][target_var_str] = {
                "scores": [...],           # List of scores for each example
                "average_score": 0.85      # Mean score across all examples
            }

    Example:
        >>> # Step 1: Run interventions once (expensive)
        >>> raw_results = experiment.perform_interventions(datasets, save_dir="./results")
        >>>
        >>> # Step 2: Try different target variable combinations (cheap)
        >>> results_A = compute_interchange_scores(
        ...     raw_results, causal_model, datasets,
        ...     target_variables_list=[["A"]], checker=exact_match
        ... )
        >>> results_AB = compute_interchange_scores(
        ...     raw_results, causal_model, datasets,
        ...     target_variables_list=[["A", "B"]], checker=exact_match
        ... )
        >>>
        >>> # Step 3: Visualize both (same visualization code)
        >>> experiment.plot_heatmaps(results_A, target_variables=["A"])
        >>> experiment.plot_heatmaps(results_AB, target_variables=["A", "B"])
    """
    # Convert single dataset to dictionary
    if isinstance(datasets, CounterfactualDataset):
        datasets = {datasets.id: datasets}

    # Create a deep copy to avoid modifying the input
    results = copy.deepcopy(raw_results)

    # Process each dataset and model unit combination
    for dataset_name in datasets.keys():
        if dataset_name not in results["dataset"]:
            continue

        for model_units_str, model_unit_data in results["dataset"][dataset_name]["model_unit"].items():
            if model_unit_data is None:
                continue

            # Get raw outputs and causal inputs
            raw_outputs = model_unit_data.get("raw_outputs")
            causal_model_inputs = model_unit_data.get("causal_model_inputs")

            if raw_outputs is None or causal_model_inputs is None:
                continue

            # Process and decode model outputs from batch dictionaries
            dumped_outputs = []
            flattened_outputs = []
            for batch_dict in raw_outputs:
                # Use the string field that's already in batch_dict
                batch_strings = batch_dict["string"]
                # Always treat as a list for consistent processing
                if not isinstance(batch_strings, list):
                    batch_strings = [batch_strings]

                dumped_outputs.extend(batch_strings)
                # Create individual output dicts for each example in the batch
                for idx, decoded_str in enumerate(batch_strings):
                    example_dict = {"sequences": batch_dict["sequences"][idx:idx+1]}

                    # Handle top-K formatted scores (list of dicts)
                    if "scores" in batch_dict and batch_dict["scores"]:
                        example_dict["scores"] = []
                        for score_dict in batch_dict["scores"]:
                            sliced_score = {
                                "top_k_logits": score_dict["top_k_logits"][idx:idx+1],
                                "top_k_indices": score_dict["top_k_indices"][idx:idx+1],
                                "top_k_tokens": [score_dict["top_k_tokens"][idx]]
                            }
                            example_dict["scores"].append(sliced_score)

                    example_dict["string"] = decoded_str
                    flattened_outputs.append(example_dict)

            # Evaluate results for each target variable group
            # This replicates the logic from intervention_experiment.py lines 219-239
            for target_variables in target_variables_list:
                target_variable_str = "-".join(target_variables)

                # Generate expected outputs from causal model
                labeled_data = causal_model.label_counterfactual_data(
                    datasets[dataset_name],
                    target_variables
                )

                # Validate alignment
                assert len(labeled_data) == len(dumped_outputs), \
                    f"Length mismatch: {len(labeled_data)} vs {len(dumped_outputs)}"
                assert len(labeled_data) == len(flattened_outputs), \
                    f"Length mismatch: {len(labeled_data)} vs {len(flattened_outputs)}"

                # Compute intervention scores - pass neural dict and expected label
                scores = []
                for example, output_dict in zip(labeled_data, flattened_outputs):
                    score = checker(output_dict, example["label"])
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    scores.append(float(score))

                # Store processed results in the same structure as perform_interventions
                results["dataset"][dataset_name]["model_unit"][model_units_str][target_variable_str] = {
                    "scores": scores,
                    "average_score": np.mean(scores)
                }

    return results