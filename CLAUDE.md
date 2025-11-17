# CLAUDE.md - AI Assistant Guide for CausalLab

## Overview

**CausalLab** is a research toolkit for mechanistic interpretability that uses causal abstraction to reverse-engineer algorithms implemented by neural networks. This codebase serves as the baseline for the Mechanistic Interpretability Benchmark (MIB) causal variable localization track.

**Repository**: Causal abstraction for mechanistic interpretability
**Primary Language**: Python 3.10+
**Key Dependencies**: PyTorch, pyvene, transformers, scikit-learn
**Lines of Code**: ~5,400 (core modules)
**Testing**: pytest with comprehensive unit and integration tests

## Core Philosophy

This codebase follows a **causal abstraction approach**:
- **High-level models**: Abstract algorithms represented as causal models
- **Low-level models**: Neural networks represented as causal models
- **Causal abstraction**: The relationship proving a neural network implements a specific algorithm
- **Features**: Agnostic building blocks for understanding AI systems (neurons, PCA components, SAE features, etc.)

## Repository Structure

```
causalab/
├── causal/                    # Causal model abstractions
│   ├── causal_model.py       # CausalModel class with variables, values, parents, mechanisms
│   ├── counterfactual_dataset.py  # Dataset handling for counterfactual generation
│   └── causal_utils.py       # Utility functions for causal reasoning
│
├── neural/                    # Neural network components
│   ├── pipeline.py           # Pipeline abstraction (Pipeline, LMPipeline)
│   ├── model_units.py        # AtomicModelUnit base classes for accessing model components
│   ├── LM_units.py           # Language model units (ResidualStreamUnit, AttentionHeadUnit)
│   ├── featurizers.py        # Invertible feature transformations & intervention builders
│   └── __init__.py
│
├── experiments/               # Experiment framework
│   ├── pyvene_core.py        # Core pyvene utilities (prepare model, run interventions, train)
│   ├── intervention_experiment.py  # Base InterventionExperiment class
│   ├── filter_experiment.py  # Feature selection experiments
│   ├── benchmark_experiment.py     # Benchmark evaluation framework
│   ├── config.py             # Default configurations & presets
│   ├── experiment_utils.py   # Shared experiment utilities
│   └── LM_experiments/       # Language model specific experiments
│
├── tasks/                     # Task definitions
│   ├── task.py               # Base Task class (causal_models, dataset_generators, token_positions)
│   └── MCQA/                 # Multiple Choice Question Answering task
│       └── mcqa.py           # MCQA implementation with positional causal model
│
├── tests/                     # Comprehensive test suite
│   ├── test_*.py             # Unit tests for each module
│   ├── integration/          # Integration tests matching tutorial notebooks
│   │   ├── test_01_mcqa_task_definition.py
│   │   ├── test_02_residual_stream_tracing.py
│   │   ├── test_03_localization_with_patching.py
│   │   └── test_DAS_and_DBM_integration.py
│   └── test_pyvene_core/     # Specialized pyvene integration tests
│
├── demos/                     # Tutorials and demonstrations
│   ├── onboarding_tutorial/  # Step-by-step tutorial notebooks
│   │   ├── 01_define_MCQA_task.ipynb
│   │   ├── 02_trace_residual_stream.ipynb
│   │   ├── 03_localize_with_patching.ipynb
│   │   └── 04_train_DAS_and_DBM.ipynb
│   └── causal_model_demo.ipynb
│
├── pyproject.toml            # Poetry configuration
├── requirements.txt          # Pip requirements
└── README.md                 # User-facing documentation
```

## Core Concepts & Components

### 1. Causal Models (`causal/causal_model.py`)

**CausalModel** represents high-level algorithms:

```python
class CausalModel:
    variables: list          # Abstract concepts (e.g., "answer_position")
    values: dict            # Possible assignments per variable
    parents: dict           # Directed causal dependencies
    mechanisms: dict        # Functions computing variable values from parents
```

**Key constraints**:
- Must include `raw_input` and `raw_output` variables
- Automatically computes `children`, `inputs`, `outputs`, `timesteps`
- Supports counterfactual generation with intervention specifications

**Important methods**:
- `generate_dataset()`: Create factual examples
- `generate_counterfactual_dataset()`: Create intervention pairs
- `display()`: Interactive Dash visualization

### 2. Neural Network Abstractions (`neural/`)

#### Pipelines (`pipeline.py`)
- **Pipeline** (ABC): Base class with `load()`, `dump()`, `generate()`, `intervenable_generate()`
- **LMPipeline**: Concrete implementation for HuggingFace causal language models
  - Manages tokenizer and model
  - Handles device placement (CUDA/CPU) and dtype inference
  - Provides chat template support

#### Model Units (`model_units.py`, `LM_units.py`)
**AtomicModelUnit**: Base class for addressable neural network components
- Subclasses: `ResidualStreamUnit`, `AttentionHeadUnit`, `MLPUnit`
- Each unit knows its layer, position, and intervention configuration
- `is_static()`: Whether position indices are fixed
- `create_intervention_config()`: Generate pyvene config

**Example usage**:
```python
# Residual stream at layer 3, position 5
unit = ResidualStreamUnit(layer=3, position=5, featurizer=my_featurizer)

# Attention head 2 at layer 1, dynamic position
unit = AttentionHeadUnit(layer=1, head=2, position=lambda x: get_answer_index(x))
```

#### Featurizers (`featurizers.py`)
**Featurizer**: Invertible transformations on hidden states

```python
class Featurizer:
    featurizer: nn.Module             # x → (features, error)
    inverse_featurizer: nn.Module     # (features, error) → x̂
    n_features: int                   # Feature dimensionality
```

**Built-in types**:
- **Identity**: No transformation
- **Interchange**: Swap feature values between inputs
- **Collect**: Extract features without intervention
- **Mask**: Binary masking with Gumbel-Softmax (for DBM)

### 3. Intervention Framework (`experiments/pyvene_core.py`)

**Core functions** (all prefixed with `_` indicating internal use):

1. **`_prepare_intervenable_model(pipeline, model_units_list, intervention_type)`**
   - Creates pyvene IntervenableModel
   - Links inner lists → shared counterfactual inputs
   - Optimizes for static vs dynamic indices

2. **`_prepare_intervenable_inputs(pipeline, batch, model_units_list)`**
   - Loads base and counterfactual inputs
   - Prepares intervention indices for each unit

3. **`_run_interchange_intervention(intervenable_model, base, sources, ...)`**
   - Executes interventions with counterfactual data
   - Returns model outputs and optionally collected features

4. **`_train_interventions(intervenable_model, train_dataloader, ...)`**
   - Trains DAS/DBM feature selection methods
   - Supports learning rate scheduling, early stopping
   - TensorBoard logging integration

### 4. Experiment Classes (`experiments/`)

**InterventionExperiment** (`intervention_experiment.py`): Base class for all experiments
- `run_experiment(dataset, model_units)`: Execute interventions
- `train_interventions(dataset, model_units, method="DAS")`: Train feature selection
- `load_interventions(path)`: Load trained parameters
- `save_interventions(path)`: Save trained parameters
- Configuration via `config` dict (see `config.py`)

**FilterExperiment** (`filter_experiment.py`): Feature selection and filtering
- Extends InterventionExperiment for ablation studies

**BenchmarkExperiment** (`benchmark_experiment.py`): MIB evaluation
- Implements MIB protocol for standardized evaluation

### 5. Task System (`tasks/`)

**Task**: Organizes causal models, datasets, and token positions

```python
@dataclass
class Task:
    name: str
    causal_models: Dict[str, Callable]        # Factory functions for CausalModel
    dataset_generators: Dict[str, Callable]   # Dataset generation functions
    token_positions: Dict[str, Callable]      # Token position computation

    def create_datasets(num_samples, suffix=""): ...
    def create_token_positions(pipeline): ...
```

**Example tasks**:
- **MCQA** (`tasks/MCQA/mcqa.py`): Multiple choice color questions
  - Variables: template, object_color, symbols, choices, answer_position, answer
  - Demonstrates positional reasoning

## Development Workflows

### Creating a New Task

1. Define causal model components:
   ```python
   variables = ["input_var1", "input_var2", "raw_input", "intermediate", "raw_output"]
   values = {"input_var1": [...], ...}
   parents = {"intermediate": ["input_var1", "input_var2"], ...}
   mechanisms = {"intermediate": lambda x, y: ..., ...}
   ```

2. Create CausalModel instance:
   ```python
   model = CausalModel(variables, values, parents, mechanisms, id="my_task")
   ```

3. Define dataset generator and token positions:
   ```python
   def dataset_generator():
       return model.generate_counterfactual_dataset(
           num_samples=1000,
           intervene_on=["intermediate"]
       )

   def get_token_positions(pipeline):
       return [TokenPosition(layer, lambda x: get_position(x), "description")]
   ```

4. Package as Task:
   ```python
   task = Task(
       name="my_task",
       causal_models={"main": lambda: model},
       dataset_generators={"main": dataset_generator},
       token_positions={"main": get_token_positions}
   )
   ```

### Running Intervention Experiments

**Standard workflow**:

1. **Initialize pipeline**:
   ```python
   pipeline = LMPipeline("gpt2", max_new_tokens=1, device="cuda")
   ```

2. **Create experiment**:
   ```python
   from experiments import InterventionExperiment
   from experiments.config import DEFAULT_CONFIG

   config = DEFAULT_CONFIG.copy()
   config["train_batch_size"] = 64

   experiment = InterventionExperiment(pipeline, config=config)
   ```

3. **Define model units**:
   ```python
   from neural.LM_units import ResidualStreamUnit

   model_units = [
       [ResidualStreamUnit(layer=3, position=5)],  # Single intervention point
       [ResidualStreamUnit(layer=5, position=5)],
   ]
   ```

4. **Run experiment**:
   ```python
   results = experiment.run_experiment(dataset, model_units)
   accuracy = results["accuracy"]
   ```

5. **Train feature selection (DAS/DBM)**:
   ```python
   train_results = experiment.train_interventions(
       train_dataset,
       model_units,
       method="DAS",  # or "DBM"
       eval_dataset=eval_dataset
   )
   ```

6. **Save/load trained interventions**:
   ```python
   experiment.save_interventions("results/my_experiment/")
   experiment.load_interventions("results/my_experiment/")
   ```

### Testing Strategy

**Run tests**:
```bash
pytest tests/                           # All tests
pytest tests/integration/               # Integration tests
pytest tests/test_pyvene_core/          # pyvene-specific tests
pytest tests/test_causal_model.py -v    # Single module with verbose
```

**Test organization**:
- **Unit tests**: `test_*.py` for each module (e.g., `test_causal_model.py`)
- **Integration tests**: `tests/integration/test_0X_*.py` matching tutorial notebooks
- **Fixtures**: `conftest.py` provides shared fixtures (pipelines, datasets, etc.)

**Key test fixtures** (from `tests/integration/conftest.py`):
- `tiny_pipeline`: Small GPT-2 for fast testing
- `mcqa_task`: MCQA task instance
- `mcqa_dataset`: Pre-generated MCQA dataset

## Key Conventions & Patterns

### Code Style

1. **Imports**: Group by standard lib, third-party, local
   ```python
   import random
   from typing import Dict, List

   import torch
   import pyvene as pv

   from causal.causal_model import CausalModel
   from neural.pipeline import Pipeline
   ```

2. **Docstrings**: Google-style for classes/functions
   ```python
   def my_function(param1, param2):
       """Brief description.

       Longer explanation if needed.

       Args:
           param1: Description
           param2: Description

       Returns:
           Description of return value
       """
   ```

3. **Type hints**: Use where helpful, especially for public APIs
   ```python
   def process_data(data: Dict[str, Any]) -> torch.Tensor:
       ...
   ```

4. **Private functions**: Prefix with `_` (e.g., `_prepare_intervenable_model`)

### Configuration Management

**Always use config dicts** from `experiments/config.py`:

```python
from experiments.config import DEFAULT_CONFIG

# Create experiment-specific config
config = DEFAULT_CONFIG.copy()
config["train_batch_size"] = 128
config["DAS"]["n_features"] = 64
config["masking"]["regularization_coefficient"] = 1e-3
```

**Important config keys**:
- `train_batch_size`, `evaluation_batch_size`: Batch sizes
- `training_epoch`, `init_lr`: Training hyperparameters
- `DAS.n_features`: Number of learned orthogonal directions
- `masking.regularization_coefficient`: DBM sparsity penalty
- `masking.temperature_schedule`: Gumbel-Softmax annealing (start, end)
- `output_scores`: Whether to keep scores in memory
- `save_top_k_logits`: Number of top logits to save to disk

### Memory Management

**Critical for CUDA**:
- Always use `_delete_intervenable_model()` when done
- Set `memory_cleanup_freq` in config for long training runs
- Use `gc.collect()` and `torch.cuda.empty_cache()` explicitly

```python
from experiments.pyvene_core import _delete_intervenable_model

# After experiment
_delete_intervenable_model(intervenable_model)
```

### Model Unit Patterns

**Static vs Dynamic positions**:
```python
# Static: position known at model creation
unit = ResidualStreamUnit(layer=3, position=5)

# Dynamic: position computed per example
unit = ResidualStreamUnit(
    layer=3,
    position=lambda batch: get_last_token_index(batch["input_ids"])
)
```

**Linked interventions** (shared counterfactual):
```python
# These share the same counterfactual input
model_units = [
    [
        ResidualStreamUnit(layer=2, position=5),
        ResidualStreamUnit(layer=3, position=5),  # Same counterfactual as layer 2
    ],
    [
        AttentionHeadUnit(layer=4, head=0, position=5),  # Different counterfactual
    ]
]
```

### Dataset Conventions

**CounterfactualDataset structure**:
```python
dataset = CounterfactualDataset({
    "base": [...],          # Factual examples
    "source": [...],        # Counterfactual examples
    "causal_model": model,
    "variable_values": {
        "var1": [...],      # Variable values for each example
        "var2": [...],
    }
})
```

**Access patterns**:
```python
dataset[0]  # Single example
dataset.subset(indices)  # Create subset
dataset.shuffle()  # Shuffle in place
dataset.split(train_frac=0.8)  # Split into train/val
```

## Common Operations & Examples

### 1. Activation Patching

```python
from experiments import InterventionExperiment
from neural.LM_units import ResidualStreamUnit

# Setup
pipeline = LMPipeline("gpt2")
experiment = InterventionExperiment(pipeline)

# Patch residual stream at layer 5, position 10
units = [[ResidualStreamUnit(layer=5, position=10)]]
results = experiment.run_experiment(dataset, units)

print(f"Intervention accuracy: {results['accuracy']:.2%}")
```

### 2. Training DAS Features

```python
config = DEFAULT_CONFIG.copy()
config["DAS"]["n_features"] = 32
config["training_epoch"] = 10

experiment = InterventionExperiment(pipeline, config=config)

# Define units (DAS will learn orthogonal directions)
units = [[ResidualStreamUnit(layer=6, position=10)]]

# Train
train_results = experiment.train_interventions(
    train_dataset,
    units,
    method="DAS",
    eval_dataset=val_dataset
)

# Save learned features
experiment.save_interventions("results/das_features/")
```

### 3. Training DBM on SAE Features

```python
from neural.featurizers import Featurizer

# Assume you have SAE encoder/decoder
sae_featurizer = Featurizer(
    featurizer=sae_encoder,
    inverse_featurizer=sae_decoder,
    n_features=16384,  # SAE latent dimension
    id="sae_features"
)

# Create unit with SAE features
unit = ResidualStreamUnit(layer=8, position=10, featurizer=sae_featurizer)

# Train DBM to select sparse subset
config["masking"]["regularization_coefficient"] = 1e-4
experiment = InterventionExperiment(pipeline, config=config)

train_results = experiment.train_interventions(
    train_dataset,
    [[unit]],
    method="DBM"
)

# Analyze learned mask
mask = train_results["masks"][0]  # Binary mask over 16384 features
active_features = (mask > 0.5).sum()
print(f"Selected {active_features} out of {len(mask)} features")
```

### 4. Analyzing Multiple Layers

```python
# Sweep across layers
results_by_layer = {}
for layer in range(12):
    units = [[ResidualStreamUnit(layer=layer, position=10)]]
    results = experiment.run_experiment(dataset, units)
    results_by_layer[layer] = results["accuracy"]

# Plot
import matplotlib.pyplot as plt
plt.plot(results_by_layer.keys(), results_by_layer.values())
plt.xlabel("Layer")
plt.ylabel("Intervention Accuracy")
plt.show()
```

### 5. Creating Custom Causal Models

```python
# Define a simple sentiment task
variables = ["raw_input", "sentiment", "intensity", "raw_output"]

values = {
    "sentiment": ["positive", "negative"],
    "intensity": ["weak", "strong"],
    "raw_input": None,
    "raw_output": None,
}

parents = {
    "raw_input": [],
    "sentiment": [],
    "intensity": [],
    "raw_output": ["sentiment", "intensity"],
}

def generate_output(sentiment, intensity):
    templates = {
        ("positive", "weak"): "This is okay.",
        ("positive", "strong"): "This is amazing!",
        ("negative", "weak"): "This is not great.",
        ("negative", "strong"): "This is terrible!",
    }
    return templates[(sentiment, intensity)]

mechanisms = {
    "raw_input": lambda: "Review: ",
    "sentiment": lambda: random.choice(["positive", "negative"]),
    "intensity": lambda: random.choice(["weak", "strong"]),
    "raw_output": generate_output,
}

model = CausalModel(variables, values, parents, mechanisms, id="sentiment")
```

## Important Notes for AI Assistants

### When Adding New Features

1. **Maintain abstraction boundaries**:
   - `causal/`: Pure causal reasoning, no neural network code
   - `neural/`: Neural network abstractions, no task-specific code
   - `experiments/`: Intervention logic, relies on neural abstractions
   - `tasks/`: Task definitions, combines causal + neural

2. **Always add tests**:
   - Unit tests in `tests/test_<module>.py`
   - Integration tests if modifying workflows
   - Use existing fixtures from `conftest.py`

3. **Update configurations**:
   - Add new hyperparameters to `experiments/config.py`
   - Document their purpose and reasonable ranges

4. **Consider memory**:
   - CUDA OOM is common with large models
   - Add cleanup logic for new stateful components
   - Test with different batch sizes

### When Debugging

1. **Check intervention configs**: Most issues arise from incorrect `model_units_list` structure
   - Inner lists = shared counterfactual
   - Each list element = separate counterfactual

2. **Validate causal models**: Use `model.display()` to visualize
   - Ensure `raw_input` and `raw_output` are present
   - Check that mechanisms match parent structure

3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Use small models first**: Test with `gpt2` before scaling to larger models

5. **Check for static/dynamic mismatch**:
   - If positions are static, ensure `is_static()` returns `True`
   - Dynamic positions require callable position functions

### Common Pitfalls

1. **Empty feature indices**: When feature_indices are empty, interventions are skipped (this is intentional but can be confusing)

2. **Dataset format**: CounterfactualDataset expects specific structure
   - `base`: factual inputs
   - `source`: counterfactual inputs
   - `variable_values`: ground truth causal variable values

3. **Featurizer requirements**:
   - Interchange: needs `featurizer` and `inverse_featurizer`
   - Collect: only needs `featurizer`
   - Mask: requires `n_features` to be set

4. **Config immutability**: Always copy `DEFAULT_CONFIG` before modifying:
   ```python
   config = DEFAULT_CONFIG.copy()  # Good
   config = DEFAULT_CONFIG         # Bad - modifies global
   ```

5. **pyvene version**: This codebase uses the main branch of pyvene from GitHub, not PyPI version

### File Locations for Common Tasks

| Task | Primary Files |
|------|---------------|
| Modify causal model logic | `causal/causal_model.py` |
| Add new model unit type | `neural/model_units.py`, `neural/LM_units.py` |
| Add new featurizer | `neural/featurizers.py` |
| Modify intervention execution | `experiments/pyvene_core.py` |
| Add experiment method | `experiments/intervention_experiment.py` |
| Add configuration option | `experiments/config.py` |
| Create new task | `tasks/<task_name>/` |
| Add tests | `tests/test_*.py` or `tests/integration/` |

### Dependencies & Installation

**Poetry (recommended)**:
```bash
poetry install
```

**Pip**:
```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `pyvene`: Installed from GitHub main branch (not PyPI)
- `torch`: Version depends on CUDA availability
- `transformers`: For HuggingFace models
- `pytest`: For testing

**Optional visualization**:
- `dash`, `dash-cytoscape`: For interactive causal model graphs
- `tensorboard`: For training monitoring
- `jupyter`: For tutorial notebooks

### Git Workflow

- **Main branch**: Production-ready code
- **Feature branches**: Named `claude/claude-md-*` for AI assistant work
- **Commit style**: Descriptive messages focused on "why" not "what"
- **Before committing**: Run `pytest tests/` to ensure tests pass

### Performance Considerations

1. **Batch size**: Start with 32, increase if memory allows
2. **Model size**: GPT-2 for prototyping, scale up for production
3. **Dataset size**: 1000 examples usually sufficient for development
4. **Training epochs**: 3-10 for DAS/DBM is typical
5. **GPU memory**: Monitor with `nvidia-smi`, use `memory_cleanup_freq` if needed

---

## Quick Reference

**Most common imports**:
```python
from causal.causal_model import CausalModel, CounterfactualDataset
from neural.pipeline import LMPipeline
from neural.LM_units import ResidualStreamUnit, AttentionHeadUnit, TokenPosition
from neural.featurizers import Featurizer
from experiments import InterventionExperiment
from experiments.config import DEFAULT_CONFIG
from tasks.task import Task
```

**Typical experiment setup**:
```python
# 1. Create pipeline
pipeline = LMPipeline("gpt2", max_new_tokens=1)

# 2. Load task and generate dataset
task = load_my_task()
datasets = task.create_datasets(num_samples=1000)
dataset = datasets["main"]

# 3. Create experiment
config = DEFAULT_CONFIG.copy()
experiment = InterventionExperiment(pipeline, config=config)

# 4. Define interventions
model_units = [[ResidualStreamUnit(layer=5, position=10)]]

# 5. Run
results = experiment.run_experiment(dataset, model_units)
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

**Last Updated**: 2025-11-17
**Codebase Version**: Based on commit dad2da2 (Merge PR #8 - attention)

For questions or issues, refer to the [README.md](README.md) or the [onboarding tutorial notebooks](demos/onboarding_tutorial/).
