"""End-to-end test for the Prefect training flow in src.modelling.train_flow."""

import pytest

prefect = pytest.importorskip(
    "prefect",
    reason="Prefect not installed; skipping flow tests."
)


def test_training_flow_end_to_end(tmp_path, sample_csv):
    """Run the full training flow and verify model output and evaluation metrics."""
    from src.modelling.train_flow import training_flow

    metrics = training_flow(
        trainset_path=sample_csv,
        output_dir=tmp_path,
        test_size=0.3,
        random_state=123,
    )

    assert set(metrics) == {"mae", "mse", "r2"}
    assert (tmp_path / "model.pkl").exists()
