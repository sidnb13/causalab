"""
Basic integration test for attribution patching.

Tests that attribution patching runs end-to-end and produces valid scores.
"""

import numpy as np
import pytest
import torch

from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream
from tasks.MCQA.mcqa import MCQA_task

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


class TestAttributionPatchingBasic:
    """Basic tests to ensure attribution patching works.

    These tests are optimized for speed:
    - 1 layer (instead of all layers)
    - 1 token position (instead of all positions)
    - 4 examples (instead of larger datasets)
    - Batch size 2 (for faster gradient computation)
    """

    def test_attribution_patching_runs(
        self, pipeline, causal_model, checker, small_different_symbol_dataset
    ):
        """Test that attribution patching runs without errors.

        This is a minimal smoke test to verify the attribution patching
        implementation works end-to-end with gradients and produces scores.
        """
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())

        # Optimize for speed: test only 1 layer and 1 position
        layers = list(range(0, min(1, pipeline.get_num_layers())))

        experiment = PatchResidualStream(
            pipeline=pipeline,
            causal_model=causal_model,
            layers=layers,
            token_positions=token_positions[:1],  # Just 1 position
            checker=checker,
            config={"batch_size": 2},  # Small batch for speed
        )

        datasets = {"test": small_different_symbol_dataset}

        # Run attribution patching
        results = experiment.perform_attribution_patching(datasets, verbose=True)

        # Verify basic structure
        assert results is not None
        assert results["method_name"] == "attribution_patching"
        assert "test" in results["dataset"]

        # Verify scores exist
        first_unit_key = next(iter(results["dataset"]["test"]["model_unit"].keys()))
        unit_result = results["dataset"]["test"]["model_unit"][first_unit_key]
        assert "attribution" in unit_result
        assert "average_score" in unit_result["attribution"]
        assert isinstance(unit_result["attribution"]["average_score"], float)

        print(f"\n✓ Attribution patching completed successfully!")
        print(
            f"✓ Tested {len(results['dataset']['test']['model_unit'])} intervention locations"
        )
