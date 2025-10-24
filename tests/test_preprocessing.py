"""Tests for data preprocessing utilities in src.modelling.preprocessing."""

import pandas as pd
import numpy as np
from src.modelling.preprocessing import to_snake, remove_iqr_outliers, preprocess_data


def test_to_snake_basic():
    """Ensure to_snake correctly converts strings to snake_case."""
    assert to_snake("Whole weight") == "whole_weight"


def test_remove_iqr_outliers_removes_extremes():
    """Verify that remove_iqr_outliers removes extreme values using the IQR method."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000], "y": [10, 11, 10, 12, 11, 9]})
    cleaned = remove_iqr_outliers(df, ["x"])
    assert 1000 not in cleaned["x"].values
    assert cleaned["x"].min() >= 1
    assert cleaned["x"].max() <= 5


def test_preprocess_data_drops_age_and_encodes_sex(sample_csv):
    """Check preprocessing: drop 'age', keep 'rings', and one-hot encode 'sex'."""
    df = preprocess_data(sample_csv)

    assert "age" not in df.columns
    assert "rings" in df.columns

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    assert "rings" in numeric_cols

    sex_dummy_cols = [c for c in df.columns if c.startswith("sex_")]
    assert len(sex_dummy_cols) == 2
