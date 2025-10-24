"""Pytest configuration and shared fixtures for the modelling test suite."""

from pathlib import Path
import sys
import pandas as pd
import pytest
import logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small sample Abalone-like DataFrame used across tests."""
    return pd.DataFrame(
        {
            "Sex": ["M", "F", "I", "M", "F", "I"],
            "Length": [0.5, 0.45, 0.3, 0.6, 0.4, 0.35],
            "Diameter": [0.4, 0.35, 0.25, 0.5, 0.32, 0.28],
            "Height": [0.1, 0.12, 0.08, 0.11, 0.1, 0.09],
            "Whole weight": [0.8, 0.6, 0.2, 1.0, 0.55, 0.3],
            "Shucked weight": [0.35, 0.28, 0.1, 0.45, 0.25, 0.15],
            "Viscera weight": [0.17, 0.14, 0.05, 0.2, 0.13, 0.07],
            "Shell weight": [0.25, 0.2, 0.08, 0.3, 0.18, 0.1],
            "Rings": [10, 8, 5, 12, 7, 6],
            "age": [50, 40, 25, 60, 35, 30],
        }
    )


@pytest.fixture
def sample_csv(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Write the sample DataFrame to a temporary CSV file and return its path."""
    csv_path = tmp_path / "abalone.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="session", autouse=True)
def _silence_prefect_console_logging():
    """Suppress Prefect console logging noise during test teardown."""
    for name in ("prefect", "prefect.server", "prefect.server.api.server"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
