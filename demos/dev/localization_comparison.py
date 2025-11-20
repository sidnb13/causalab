"""
Comparison script: Activation Patching vs Attribution Patching.

This script runs both methods on the same datasets and saves side-by-side visualizations.
"""

import matplotlib

matplotlib.use(
    "Agg"
)  # Use non-interactive backend to prevent plot windows from showing

import copy
import os
import time

import torch

from causal.causal_utils import CheapArgmaxChecker, StringMatchChecker
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream
from neural.pipeline import LMPipeline
from tasks.MCQA.causal_models import get_answer, get_answer_position
from tasks.MCQA.mcqa import MCQA_task

# Setup - Use GPU acceleration if available (CUDA or MPS)
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_name = "Qwen/Qwen2.5-0.5B"
pipeline = LMPipeline(
    model_name, max_new_tokens=1, device=device, dtype=torch.float32, max_length=32
)
pipeline.tokenizer.padding_side = "left"

causal_model = MCQA_task.causal_models["positional"]

# Use different checkers for each experiment type

causal_model = MCQA_task.causal_models["positional"]


intervention_checker = StringMatchChecker()  # For activation patching
attribution_checker = CheapArgmaxChecker()  # For attribution patching

# Create output directories relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir_patching = os.path.join(
    script_dir, "comparison_results/activation_patching"
)
results_dir_attribution = os.path.join(
    script_dir, "comparison_results/attribution_patching"
)
os.makedirs(results_dir_patching, exist_ok=True)
os.makedirs(results_dir_attribution, exist_ok=True)

# Create datasets
size = 256  # Smaller for faster iteration while testing
counterfactual_datasets = MCQA_task.create_datasets(size)

# Filter datasets using intervention checker
exp = FilterExperiment(pipeline, causal_model, intervention_checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=64)

# Check which datasets have valid examples after filtering
if not filtered_datasets:
    raise ValueError(
        "No valid examples remain after filtering. All datasets were filtered out."
    )

# Print which datasets are available
print(f"\nAvailable datasets after filtering: {list(filtered_datasets.keys())}")
for dataset_name, examples in filtered_datasets.items():
    print(f"  - {dataset_name}: {len(examples)} examples")

# Only proceed if we have at least one dataset with examples
if len(filtered_datasets) == 0:
    raise ValueError("No datasets available for experimentation.")

# Create token positions - just use a few for faster testing
# Avoid "correct_symbol" since some counterfactuals have unanswerable questions
all_token_positions = MCQA_task.create_token_positions(pipeline)
token_positions = [v for k, v in all_token_positions.items() if k != "correct_symbol"]

# Setup experiment
start = 0
end = pipeline.get_num_layers()
# Sample every 4th layer to reduce intervention space
layers_to_test = list(range(start, end, 4))  # Every 4th layer
config = {"batch_size": 32}  # Reduced batch size
target_variables_list = [["answer"], ["answer_position"]]

experiment = PatchResidualStream(
    pipeline,
    layers_to_test,
    token_positions,
    causal_model=causal_model,
    checker=intervention_checker,
    config=config,
)

# ==============================================================================
# METHOD 1: REGULAR ACTIVATION PATCHING
# ==============================================================================
start_time = time.time()

patching_results = experiment.perform_interventions(
    filtered_datasets, verbose=True, target_variables_list=target_variables_list
)

patching_time = time.time() - start_time

# Generate activation patching visualizations

# Different Symbol
if "different_symbol" in filtered_datasets:
    diff_results_patching = copy.deepcopy(patching_results)
    if "same_symbol_different_position" in diff_results_patching["dataset"]:
        del diff_results_patching["dataset"]["same_symbol_different_position"]
    if "random_counterfactual" in diff_results_patching["dataset"]:
        del diff_results_patching["dataset"]["random_counterfactual"]
    experiment.plot_heatmaps(
        diff_results_patching, ["answer"], save_path=results_dir_patching
    )
    experiment.plot_heatmaps(
        diff_results_patching, ["answer_position"], save_path=results_dir_patching
    )

# Same Symbol Different Position
if "same_symbol_different_position" in filtered_datasets:
    same_diff_results_patching = copy.deepcopy(patching_results)
    if "different_symbol" in same_diff_results_patching["dataset"]:
        del same_diff_results_patching["dataset"]["different_symbol"]
    if "random_counterfactual" in same_diff_results_patching["dataset"]:
        del same_diff_results_patching["dataset"]["random_counterfactual"]
    experiment.plot_heatmaps(
        same_diff_results_patching, ["answer_position"], save_path=results_dir_patching
    )
    experiment.plot_heatmaps(
        same_diff_results_patching, ["answer"], save_path=results_dir_patching
    )

# Random
if "random_counterfactual" in filtered_datasets:
    random_results_patching = copy.deepcopy(patching_results)
    if "different_symbol" in random_results_patching["dataset"]:
        del random_results_patching["dataset"]["different_symbol"]
    if "same_symbol_different_position" in random_results_patching["dataset"]:
        del random_results_patching["dataset"]["same_symbol_different_position"]
    experiment.plot_heatmaps(
        random_results_patching, ["answer_position"], save_path=results_dir_patching
    )
    experiment.plot_heatmaps(
        random_results_patching, ["answer"], save_path=results_dir_patching
    )

# ==============================================================================
# METHOD 2: ATTRIBUTION PATCHING
# ==============================================================================


# Define token extraction function
def get_correct_token(item):
    pos = get_answer_position(item["object_color"], item["choice0"], item["choice1"])
    return get_answer(pos, item["symbol0"], item["symbol1"])


# Define function to get other choice tokens (for fixed attribution patching)
def get_other_choice_tokens(item):
    """Return the token that is NOT the correct answer."""
    pos = get_answer_position(item["object_color"], item["choice0"], item["choice1"])
    correct = get_answer(pos, item["symbol0"], item["symbol1"])
    # Return the other choice
    if correct == item["symbol0"]:
        return [item["symbol1"]]
    else:
        return [item["symbol0"]]


# Define metric function for batched inputs (FIXED: now requires other_choice_token_ids)
def metric_fn(logits, correct_token_ids, other_choice_token_ids):
    # logits: (batch_size, seq_len, vocab_size)
    # correct_token_ids: (batch_size,)
    # other_choice_token_ids: (batch_size, n_choices)
    last_logits = logits[:, -1, :]  # Get logits for last token

    # compute_score now returns (metric, sum_other_logits)
    results = [
        CheapArgmaxChecker.compute_score(
            last_logits[i], correct_token_ids[i], other_choice_token_ids[i]
        )
        for i in range(len(correct_token_ids))
    ]

    # Unpack metrics and sum_other_logits
    metrics = torch.stack([r[0] for r in results])
    sum_other_logits = torch.stack([r[1] for r in results])

    return metrics, sum_other_logits


start_time = time.time()

# Switch to attribution checker for this experiment
experiment.checker = attribution_checker

attribution_results = experiment.perform_attribution_patching(
    filtered_datasets,
    metric_fn=metric_fn,
    get_correct_token_fn=get_correct_token,
    get_other_choice_tokens_fn=get_other_choice_tokens,  # FIXED: Pass other choices function
    verbose=True,
    target_variables_list=target_variables_list,  # Use same target variables as activation patching
)

# Switch back to intervention checker
experiment.checker = intervention_checker

attribution_time = time.time() - start_time

# Generate attribution patching visualizations

# Different Symbol
if "different_symbol" in filtered_datasets:
    diff_results_attribution = copy.deepcopy(attribution_results)
    if "same_symbol_different_position" in diff_results_attribution["dataset"]:
        del diff_results_attribution["dataset"]["same_symbol_different_position"]
    if "random_counterfactual" in diff_results_attribution["dataset"]:
        del diff_results_attribution["dataset"]["random_counterfactual"]
    experiment.plot_heatmaps(
        diff_results_attribution, ["answer"], save_path=results_dir_attribution
    )
    experiment.plot_heatmaps(
        diff_results_attribution, ["answer_position"], save_path=results_dir_attribution
    )

# Same Symbol Different Position
if "same_symbol_different_position" in filtered_datasets:
    same_diff_results_attribution = copy.deepcopy(attribution_results)
    if "different_symbol" in same_diff_results_attribution["dataset"]:
        del same_diff_results_attribution["dataset"]["different_symbol"]
    if "random_counterfactual" in same_diff_results_attribution["dataset"]:
        del same_diff_results_attribution["dataset"]["random_counterfactual"]
    experiment.plot_heatmaps(
        same_diff_results_attribution,
        ["answer_position"],
        save_path=results_dir_attribution,
    )
    experiment.plot_heatmaps(
        same_diff_results_attribution, ["answer"], save_path=results_dir_attribution
    )

# Random
if "random_counterfactual" in filtered_datasets:
    random_results_attribution = copy.deepcopy(attribution_results)
    if "different_symbol" in random_results_attribution["dataset"]:
        del random_results_attribution["dataset"]["different_symbol"]
    if "same_symbol_different_position" in random_results_attribution["dataset"]:
        del random_results_attribution["dataset"]["same_symbol_different_position"]
    experiment.plot_heatmaps(
        random_results_attribution,
        ["answer_position"],
        save_path=results_dir_attribution,
    )
    experiment.plot_heatmaps(
        random_results_attribution, ["answer"], save_path=results_dir_attribution
    )

print(f"\nExecution times:")
print(f"  Activation Patching: {patching_time:.2f}s")
print(f"  Attribution Patching: {attribution_time:.2f}s")
