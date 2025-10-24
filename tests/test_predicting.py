"""Tests for prediction functions in src.modelling.predicting."""

import importlib
import numpy as np
import pandas as pd


def test_predict_on_unseen_data(tmp_path, sample_csv, monkeypatch):
    """Verify prediction pipeline logic using patched preprocessing and model loading."""
    predicting = importlib.import_module("src.modelling.predicting")

    calls = {"preprocess": False, "load": False, "predict": False}

    def fake_preprocess_data(path):
        """Simulate minimal preprocessing for unseen data."""
        calls["preprocess"] = True
        df = pd.read_csv(path)
        if "age" in df.columns:
            df = df.drop(columns=["age"])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "sex" in df.columns:
            df = pd.get_dummies(df, columns=["sex"], drop_first=True)
        if "rings" in df.columns:
            df = df.drop(columns=["rings"])
        return df

    class FakeModel:
        """Mock model returning sequential numeric predictions."""
        def predict(self, X):
            calls["predict"] = True
            return np.arange(len(X)) + 0.5

    def fake_load_model(_):
        """Return the fake model."""
        calls["load"] = True
        return FakeModel()

    monkeypatch.setattr(predicting, "preprocess_data", fake_preprocess_data)
    monkeypatch.setattr(predicting, "load_model", fake_load_model)

    preds = predicting.predict_on_unseen_data(sample_csv, tmp_path / "model.pkl")

    assert isinstance(preds, np.ndarray)
    assert len(preds) > 0
    assert all(calls.values())
