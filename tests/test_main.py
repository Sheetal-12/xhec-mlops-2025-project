"""Tests for the CLI entrypoint in src.modelling.main."""

from pathlib import Path
import importlib


def test_main_calls_pipeline(monkeypatch, sample_csv):
    """Ensure main() calls preprocessing and training with the expected output path."""
    mod = importlib.import_module("src.modelling.main")

    called = {"preprocess": False, "train": False, "output_dir": None}

    def fake_preprocess_data(_):
        called["preprocess"] = True
        import pandas as pd

        return pd.DataFrame({"rings": [1, 2], "sex_M": [1, 0], "Length": [0.3, 0.4]})

    def fake_train_model(_, outdir: Path):
        called["train"] = True
        called["output_dir"] = outdir

    monkeypatch.setattr(mod, "preprocess_data", fake_preprocess_data)
    monkeypatch.setattr(mod, "train_model", fake_train_model)

    mod.main(sample_csv)

    assert called["preprocess"] is True
    assert called["train"] is True

    expected_tail = Path("web_service") / "local_objects"
    assert str(called["output_dir"]).endswith(str(expected_tail))
