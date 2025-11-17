"""
Basic integration test for attribution patching.

Tests that attribution patching runs end-to-end and produces valid scores.
"""

import numpy as np
import pytest
import torch

from causal.causal_utils import CheapArgmaxChecker
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream
from tasks.MCQA.mcqa import MCQA_task
from tasks.MCQA.causal_models import get_answer_position, get_answer

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
        self, pipeline, causal_model, small_different_symbol_dataset
    ):
        """Test that attribution patching runs without errors.

        This is a minimal smoke test to verify the attribution patching
        implementation works end-to-end with gradients and produces scores.
        """
        token_positions = list(MCQA_task.create_token_positions(pipeline).values())

        # Optimize for speed: test only 1 layer and 1 position
        layers = list(range(0, min(1, pipeline.get_num_layers())))

        # Use CheapArgmaxChecker for attribution patching
        checker = CheapArgmaxChecker()

        experiment = PatchResidualStream(
            pipeline=pipeline,
            causal_model=causal_model,
            layers=layers,
            token_positions=token_positions[:1],  # Just 1 position
            checker=checker,
            config={"batch_size": 2},  # Small batch for speed
        )

        datasets = {"test": small_different_symbol_dataset}

        # Define token extraction function
        def get_correct_token(item):
            pos = get_answer_position(item['object_color'], item['choice0'], item['choice1'])
            return get_answer(pos, item['symbol0'], item['symbol1'])

        # Define metric function for batched inputs
        def metric_fn(logits, correct_token_ids):
            # logits: (batch_size, seq_len, vocab_size)
            # correct_token_ids: (batch_size,)
            last_logits = logits[:, -1, :]  # Get logits for last token
            return torch.stack([
                CheapArgmaxChecker.compute_attribution(last_logits[i], correct_token_ids[i])
                for i in range(len(correct_token_ids))
            ])

        # Run attribution patching
        results = experiment.perform_attribution_patching(
            datasets,
            metric_fn=metric_fn,
            get_correct_token_fn=get_correct_token,
            target_variables_list=[["answer"]],
            verbose=True
        )

        # Verify basic structure
        assert results is not None
        assert results["method_name"] == "attribution_patching"
        assert "test" in results["dataset"]

        # Verify scores exist
        first_unit_key = next(iter(results["dataset"]["test"]["model_unit"].keys()))
        unit_result = results["dataset"]["test"]["model_unit"][first_unit_key]
        assert "answer" in unit_result
        assert "average_score" in unit_result["answer"]
        assert isinstance(unit_result["answer"]["average_score"], float)

        print(f"\n✓ Attribution patching completed successfully!")
        print(
            f"✓ Tested {len(results['dataset']['test']['model_unit'])} intervention locations"
        )
