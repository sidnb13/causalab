"""
Clean localization script using activation patching.

This script runs end-to-end and saves all plots to the same directory.
"""

import os
import torch
import copy
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

# Create output directory
results_dir = "localization_results"
os.makedirs(results_dir, exist_ok=True)

print("="*80)
print("ACTIVATION PATCHING LOCALIZATION EXPERIMENT")
print("="*80)

# Create datasets
print("\n[1/5] Creating counterfactual datasets...")
size = 64
counterfactual_datasets = MCQA_task.create_datasets(size)
print(f"✓ Created {len(counterfactual_datasets)} datasets with {size} examples each")

# Filter datasets
print("\n[2/5] Filtering datasets based on model performance...")
exp = FilterExperiment(pipeline, causal_model, checker)
filtered_datasets = exp.filter(counterfactual_datasets, verbose=True, batch_size=128)

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
print(f"✓ Created {len(token_positions)} token positions:")
for token_position in token_positions:
    print(f"  - {token_position.id}")

# Run activation patching experiment
print("\n[4/5] Running activation patching experiment...")
start = 0
end = pipeline.get_num_layers()
config = {"batch_size": 64}
target_variables_list = [["answer"], ["answer_position"]]

experiment = PatchResidualStream(
    pipeline, causal_model, list(range(start, end)),
    token_positions, checker, config=config
)

print(f"✓ Experiment setup:")
print(f"  - Layers: {start} to {end}")
print(f"  - Token positions: {len(token_positions)}")
print(f"  - Target variables: {target_variables_list}")
print(f"  - Batch size: {config['batch_size']}")

raw_results = experiment.perform_interventions(
    filtered_datasets,
    verbose=True,
    target_variables_list=target_variables_list
)

# Generate visualizations
print("\n[5/5] Generating heatmap visualizations...")

# Different Symbol - Answer variable
print("\n  Plotting 'different_symbol' dataset - 'answer' variable...")
diff_results = copy.deepcopy(raw_results)
del diff_results["dataset"]["same_symbol_different_position"]
del diff_results["dataset"]["random_counterfactual"]
experiment.plot_heatmaps(diff_results, ["answer"], save_path=results_dir)

# Different Symbol - Answer Position variable
print("\n  Plotting 'different_symbol' dataset - 'answer_position' variable...")
experiment.plot_heatmaps(diff_results, ["answer_position"], save_path=results_dir)

# Same Symbol Different Position - Answer Position variable
print("\n  Plotting 'same_symbol_different_position' dataset - 'answer_position' variable...")
same_diff_results = copy.deepcopy(raw_results)
del same_diff_results["dataset"]["different_symbol"]
del same_diff_results["dataset"]["random_counterfactual"]
experiment.plot_heatmaps(same_diff_results, ["answer_position"], save_path=results_dir)

# Same Symbol Different Position - Answer variable
print("\n  Plotting 'same_symbol_different_position' dataset - 'answer' variable...")
experiment.plot_heatmaps(same_diff_results, ["answer"], save_path=results_dir)

# Random Counterfactual - Answer Position variable
print("\n  Plotting 'random_counterfactual' dataset - 'answer_position' variable...")
random_results = copy.deepcopy(raw_results)
del random_results["dataset"]["different_symbol"]
del random_results["dataset"]["same_symbol_different_position"]
experiment.plot_heatmaps(random_results, ["answer_position"], save_path=results_dir)

# Random Counterfactual - Answer variable
print("\n  Plotting 'random_counterfactual' dataset - 'answer' variable...")
experiment.plot_heatmaps(random_results, ["answer"], save_path=results_dir)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"Results saved to: {os.path.abspath(results_dir)}")
print("\nGenerated plots:")
for f in sorted(os.listdir(results_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
