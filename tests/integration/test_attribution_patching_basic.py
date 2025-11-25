"""
Basic integration test for attribution patching.

Tests that attribution patching runs end-to-end and produces valid scores
using the 26-letter gradient method.
"""

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

    The attribution patching uses 26 backward passes (one per letter A-Z)
    to get consistent gradients across all examples, then approximates
    post-intervention logits and takes argmax to determine the prediction.
    """

    def test_attribution_patching_runs(
        self, pipeline, causal_model, small_different_symbol_dataset
    ):
        """Test that attribution patching runs without errors.

        This is a minimal smoke test to verify the 26-letter attribution patching
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

        # Define token extraction function (used for ground truth correct answer)
        def get_correct_token(item):
            pos = get_answer_position(item['object_color'], item['choice0'], item['choice1'])
            return get_answer(pos, item['symbol0'], item['symbol1'])

        # Run attribution patching with 26-letter gradient method
        # No need for metric_fn or get_other_choice_tokens_fn anymore
        results = experiment.perform_attribution_patching(
            datasets,
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

        # Verify the score is between 0 and 1 (it's a correctness score)
        score = unit_result["answer"]["average_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"

        print("\n✓ Attribution patching (26-letter method) completed successfully!")
        print(
            f"✓ Tested {len(results['dataset']['test']['model_unit'])} intervention locations"
        )
        print(f"✓ Average score: {score:.4f}")
