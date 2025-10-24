"""Tests for object serialization utilities in src.modelling.utils."""

from pathlib import Path
from src.modelling import utils


def test_pickle_roundtrip(tmp_path: Path):
    """Ensure that pickling and unpickling an object preserves its contents."""
    obj = {"a": 1, "b": [1, 2, 3]}
    file_path = tmp_path / "artifact.pkl"

    utils.pickle_object(obj, file_path)
    assert file_path.exists()

    loaded = utils.load_pickle(file_path)
    assert loaded == obj
