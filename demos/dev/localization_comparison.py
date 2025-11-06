"""
Comparison script: Activation Patching vs Attribution Patching.

This script runs both methods on the same datasets and saves side-by-side visualizations.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plot windows from showing

import os
import torch
import copy
import time
from tasks.MCQA.mcqa import MCQA_task
from neural.pipeline import LMPipeline
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream

# Setup - Use GPU acceleration if available (CUDA or MPS)
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
model_name = "Qwen/Qwen2.5-0.5B"
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float32, max_length=32)
pipeline.tokenizer.padding_side = "left"

causal_model = MCQA_task.causal_models["positional"]

def checker(neural_output, causal_output):
    return causal_output in neural_output["string"] or neural_output["string"] in causal_output

# Create output directories relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir_patching = os.path.join(script_dir, "comparison_results/activation_patching")
results_dir_attribution = os.path.join(script_dir, "comparison_results/attribution_patching")
os.makedirs(results_dir_patching, exist_ok=True)
os.makedirs(results_dir_attribution, exist_ok=True)

print("="*80)
print("ACTIVATION PATCHING vs ATTRIBUTION PATCHING COMPARISON")
print("="*80)

# Create datasets
print("\n[1/5] Creating counterfactual datasets...")
size = 16  # Reduced from 64 to speed up experimentation
counterfactual_datasets = MCQA_task.create_datasets(size)
print(f"✓ Created {len(counterfactual_datasets)} datasets with {size} examples each")

# Filter datasets
print("\n[2/5] Filtering datasets based on model performance...")
exp = FilterExperiment(pipeline, causal_model, checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=64)

different_symbol_pairs = filtered_datasets["different_symbol"]
same_symbol_diff_position_pairs = filtered_datasets["same_symbol_different_position"]
random_pairs = filtered_datasets["random_counterfactual"]

print(f"\n✓ Filtered datasets:")
print(f"  - different_symbol: {len(different_symbol_pairs)} examples")
print(f"  - same_symbol_different_position: {len(same_symbol_diff_position_pairs)} examples")
print(f"  - random_counterfactual: {len(random_pairs)} examples")

# Create token positions
print("\n[3/5] Setting up token positions...")
token_positions = list(MCQA_task.create_token_positions(pipeline).values())
print(f"✓ Created {len(token_positions)} token positions")

# Setup experiment
start = 0
end = pipeline.get_num_layers()
# Sample every 2nd layer to reduce intervention space
layers_to_test = list(range(start, end, 2))  # Every 2nd layer
config = {"batch_size": 32}  # Reduced batch size
target_variables_list = [["answer"], ["answer_position"]]

experiment = PatchResidualStream(
    pipeline, causal_model, layers_to_test,
    token_positions, checker, config=config
)

print(f"\n✓ Experiment setup:")
print(f"  - Layers: {layers_to_test} (sampling every 2nd layer)")
print(f"  - Token positions: {len(token_positions)}")
print(f"  - Target variables: {target_variables_list}")
print(f"  - Total interventions per dataset: ~{len(layers_to_test) * len(token_positions)}")

# ==============================================================================
# METHOD 1: REGULAR ACTIVATION PATCHING
# ==============================================================================
print("\n[4a/5] Running REGULAR ACTIVATION PATCHING...")
print("-" * 80)
start_time = time.time()

patching_results = experiment.perform_interventions(
    filtered_datasets,
    verbose=True,
    target_variables_list=target_variables_list
)

patching_time = time.time() - start_time
print(f"\n✓ Regular activation patching completed in {patching_time:.2f} seconds")

# Generate activation patching visualizations
print("\n  Generating activation patching heatmaps...")

# Different Symbol
diff_results_patching = copy.deepcopy(patching_results)
del diff_results_patching["dataset"]["same_symbol_different_position"]
del diff_results_patching["dataset"]["random_counterfactual"]
experiment.plot_heatmaps(diff_results_patching, ["answer"], save_path=results_dir_patching)
experiment.plot_heatmaps(diff_results_patching, ["answer_position"], save_path=results_dir_patching)

# Same Symbol Different Position
same_diff_results_patching = copy.deepcopy(patching_results)
del same_diff_results_patching["dataset"]["different_symbol"]
del same_diff_results_patching["dataset"]["random_counterfactual"]
experiment.plot_heatmaps(same_diff_results_patching, ["answer_position"], save_path=results_dir_patching)
experiment.plot_heatmaps(same_diff_results_patching, ["answer"], save_path=results_dir_patching)

# Random
random_results_patching = copy.deepcopy(patching_results)
del random_results_patching["dataset"]["different_symbol"]
del random_results_patching["dataset"]["same_symbol_different_position"]
experiment.plot_heatmaps(random_results_patching, ["answer_position"], save_path=results_dir_patching)
experiment.plot_heatmaps(random_results_patching, ["answer"], save_path=results_dir_patching)

# ==============================================================================
# METHOD 2: ATTRIBUTION PATCHING
# ==============================================================================
print("\n[4b/5] Running ATTRIBUTION PATCHING...")
print("-" * 80)
start_time = time.time()

attribution_results = experiment.perform_attribution_patching(
    filtered_datasets,
    verbose=True,
    target_variables_list=target_variables_list  # Use same target variables as activation patching
)

attribution_time = time.time() - start_time
print(f"\n✓ Attribution patching completed in {attribution_time:.2f} seconds")
if attribution_time < patching_time:
    print(f"  Speedup: {patching_time / attribution_time:.2f}x faster than activation patching")
else:
    print(f"  Slowdown: {attribution_time / patching_time:.2f}x slower than activation patching")

# Generate attribution patching visualizations
print("\n  Generating attribution patching heatmaps...")

# Different Symbol
diff_results_attribution = copy.deepcopy(attribution_results)
del diff_results_attribution["dataset"]["same_symbol_different_position"]
del diff_results_attribution["dataset"]["random_counterfactual"]
experiment.plot_heatmaps(diff_results_attribution, ["answer"], save_path=results_dir_attribution)
experiment.plot_heatmaps(diff_results_attribution, ["answer_position"], save_path=results_dir_attribution)

# Same Symbol Different Position
same_diff_results_attribution = copy.deepcopy(attribution_results)
del same_diff_results_attribution["dataset"]["different_symbol"]
del same_diff_results_attribution["dataset"]["random_counterfactual"]
experiment.plot_heatmaps(same_diff_results_attribution, ["answer_position"], save_path=results_dir_attribution)
experiment.plot_heatmaps(same_diff_results_attribution, ["answer"], save_path=results_dir_attribution)

# Random
random_results_attribution = copy.deepcopy(attribution_results)
del random_results_attribution["dataset"]["different_symbol"]
del random_results_attribution["dataset"]["same_symbol_different_position"]
experiment.plot_heatmaps(random_results_attribution, ["answer_position"], save_path=results_dir_attribution)
experiment.plot_heatmaps(random_results_attribution, ["answer"], save_path=results_dir_attribution)

# ==============================================================================
# COMPARISON SUMMARY
# ==============================================================================
print("\n[5/5] Generating comparison summary...")
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"\nExecution Time:")
print(f"  Activation Patching:  {patching_time:.2f} seconds")
print(f"  Attribution Patching: {attribution_time:.2f} seconds")
if attribution_time < patching_time:
    print(f"  Speedup:              {patching_time / attribution_time:.2f}x (attribution is faster)")
else:
    print(f"  Slowdown:             {attribution_time / patching_time:.2f}x (attribution is slower)")

print(f"\nResults saved to:")
print(f"  Regular Patching:    {os.path.abspath(results_dir_patching)}")
print(f"  Attribution Patching: {os.path.abspath(results_dir_attribution)}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print("\nTo compare results, open the heatmap images side by side:")
print(f"  - {results_dir_patching}/*.png")
print(f"  - {results_dir_attribution}/*.png")
print("\nThe plots show intervention effects at each (layer, position) combination.")
print("Compare the patterns to see if attribution patching approximates activation patching!")
