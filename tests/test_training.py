"""Tests for model training in src.modelling.training."""

from pathlib import Path
from src.modelling.training import train_model
from src.modelling.preprocessing import preprocess_data


def test_train_model_saves_and_returns(tmp_path, sample_csv):
    """Verify that training saves a model and returns valid evaluation metrics."""
    pre_df = preprocess_data(sample_csv)

    assert "rings" in pre_df.columns, (
        f"Preprocessing did not create 'rings'. Columns: {list(pre_df.columns)}"
    )

    model, metrics = train_model(pre_df, tmp_path)

    model_path = tmp_path / "model.pkl"
    assert model_path.exists()

    assert set(metrics) == {"mae", "mse", "r2"}
    assert isinstance(metrics["mae"], float)
