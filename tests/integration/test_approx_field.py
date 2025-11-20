"""Test that approx field is computed correctly in attribution patching"""
import torch
import pytest

from causal.causal_utils import CheapArgmaxChecker, StringMatchChecker
from experiments.filter_experiment import FilterExperiment
from experiments.LM_experiments.residual_stream_experiment import PatchResidualStream
from neural.pipeline import LMPipeline
from tasks.MCQA.causal_models import get_answer, get_answer_position
from tasks.MCQA.mcqa import MCQA_task


class TestApproxField:
    """Test approx field computation in attribution patching"""

    @pytest.fixture(scope="class")
    def setup(self):
        """Setup pipeline and datasets"""
        # Use CPU for testing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Qwen/Qwen2.5-0.5B"
        pipeline = LMPipeline(
            model_name, max_new_tokens=1, device=device, dtype=torch.float32, max_length=32
        )
        pipeline.tokenizer.padding_side = "left"

        causal_model = MCQA_task.causal_models["positional"]

        # Create checkers
        intervention_checker = StringMatchChecker()
        attribution_checker = CheapArgmaxChecker()

        # Create small dataset
        size = 8
        counterfactual_datasets = MCQA_task.create_datasets(size)

        # Filter datasets
        exp = FilterExperiment(pipeline, causal_model, intervention_checker)
        filtered_datasets = exp.filter(counterfactual_datasets, verbose=False, batch_size=8)

        return {
            "pipeline": pipeline,
            "causal_model": causal_model,
            "filtered_datasets": filtered_datasets,
            "attribution_checker": attribution_checker,
        }

    def test_approx_field_exists_and_correct(self, setup):
        """Test that approx field is computed and has correct structure"""
        pipeline = setup["pipeline"]
        causal_model = setup["causal_model"]
        filtered_datasets = setup["filtered_datasets"]
        attribution_checker = setup["attribution_checker"]

        # Setup experiment
        all_token_positions = MCQA_task.create_token_positions(pipeline)
        token_positions = [v for k, v in all_token_positions.items() if k != "correct_symbol"]

        layers_to_test = [0]  # Just test first layer
        config = {"batch_size": 4}
        target_variables_list = [["answer"]]

        experiment = PatchResidualStream(
            pipeline,
            layers_to_test,
            token_positions,
            causal_model=causal_model,
            checker=attribution_checker,
            config=config,
        )

        # Define token extraction and metric functions
        def get_correct_token(item):
            pos = get_answer_position(item["object_color"], item["choice0"], item["choice1"])
            return get_answer(pos, item["symbol0"], item["symbol1"])

        def metric_fn(logits, correct_token_ids):
            last_logits = logits[:, -1, :]
            return torch.stack(
                [
                    CheapArgmaxChecker.compute_score(last_logits[i], correct_token_ids[i])
                    for i in range(len(correct_token_ids))
                ]
            )

        # Run attribution patching
        attribution_results = experiment.perform_attribution_patching(
            filtered_datasets,
            metric_fn=metric_fn,
            get_correct_token_fn=get_correct_token,
            verbose=False,
            target_variables_list=target_variables_list,
        )

        # Check that approx field exists and is computed correctly
        found_data = False
        for dataset_name, dataset_results in attribution_results["dataset"].items():
            for model_unit_str, unit_data in dataset_results["model_unit"].items():
                if unit_data is None:
                    continue
                attribution_data = unit_data.get("attribution_data")
                if attribution_data is None:
                    continue

                found_data = True

                # Check first unit's data
                unit_data_dict = attribution_data[0][0]

                # Verify fields exist
                assert "approx" in unit_data_dict, "approx field missing!"
                assert "metric_scores" in unit_data_dict, "metric_scores field missing!"
                assert "attribution_score" in unit_data_dict, "attribution_score field missing!"

                approx = unit_data_dict["approx"]
                metric_scores = unit_data_dict["metric_scores"]

                # Verify they have the same length
                if isinstance(approx, torch.Tensor):
                    approx_len = approx.shape[0]
                else:
                    approx_len = len(approx)

                assert approx_len == len(metric_scores), \
                    f"Length mismatch: approx {approx_len} vs metric_scores {len(metric_scores)}"

                # Verify approx is different from metric_scores (it should include gradient term)
                if isinstance(approx, torch.Tensor):
                    approx_vals = approx.tolist()
                else:
                    approx_vals = approx

                # They should be different (approx = metric + gradient_term)
                # Note: We use != here which works for lists
                assert approx_vals != metric_scores, "approx should differ from metric_scores!"

                print(f"✓ approx field correctly computed for {model_unit_str}")

        assert found_data, "No attribution data found in results!"
