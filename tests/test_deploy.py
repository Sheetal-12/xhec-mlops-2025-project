"""Tests for the Prefect deployment creation in src.modelling.deploy."""

import pytest

prefect = pytest.importorskip(
    "prefect",
    reason="Prefect not installed; skipping deployment tests."
)


def test_create_deployment_has_expected_metadata():
    """Validate that the Prefect deployment is correctly created and configured."""
    from src.modelling.deploy import create_deployment

    deployment = create_deployment()
    assert deployment is not None

    # Validate basic attributes if available
    if hasattr(deployment, "name"):
        assert deployment.name == "abalone-weekly-retraining"
    if hasattr(deployment, "description"):
        assert "Automated weekly retraining" in deployment.description
    if hasattr(deployment, "tags"):
        expected_tags = {"ml", "training", "abalone", "weekly"}
        assert expected_tags.issubset(set(deployment.tags))

    # Validate cron schedule representation across Prefect versions
    cron_candidates = []
    if hasattr(deployment, "cron"):
        cron_candidates.append(getattr(deployment, "cron"))
    if hasattr(deployment, "schedule") and deployment.schedule is not None:
        sched = deployment.schedule
        for attr in ("cron", "cron_string", "cron_schedule", "cron_pattern"):
            if hasattr(sched, attr):
                cron_candidates.append(getattr(sched, attr))

    if cron_candidates:
        assert any(str(c) == "0 2 * * 0" for c in cron_candidates)
