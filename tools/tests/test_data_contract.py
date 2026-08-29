"""Tests for the versioned peer-method data contract."""

from __future__ import annotations

import numpy as np

from tools.data_contract import load_manifest, load_oracle, load_peer_input


def test_manifest_marks_only_synthetic_inputs_release_eligible() -> None:
    """Synthetic fixtures are eligible while airway remains local-only."""

    manifest = load_manifest()
    eligible = {
        spec.dataset_id for spec in manifest.datasets if spec.release_eligible
    }
    assert eligible == {
        "sim_500_seed42",
        "dense_500_seed42",
        "sim_5000_seed42",
        "sim_50000_seed42",
    }
    assert not next(
        spec for spec in manifest.datasets if spec.dataset_id == "airway_seed42"
    ).release_eligible


def test_loader_normalizes_synthetic_fixture() -> None:
    """A fixture returns one-dimensional float64 arrays with matching length."""

    peer_input = load_peer_input("sim_500_seed42")
    assert peer_input.pvalues.dtype == np.dtype(np.float64)
    assert peer_input.covariates.dtype == np.dtype(np.float64)
    assert peer_input.pvalues.shape == (500,)
    assert peer_input.covariates.shape == (500,)
    assert peer_input.groups is None
    assert peer_input.oracle_id is None


def test_loader_attaches_frozen_infinity_partition() -> None:
    """The infinity oracle exposes groups, folds, and normalized lambdas."""

    peer_input = load_peer_input(
        "sim_5000_seed42", oracle_id="sim_5000_inf_n5"
    )
    assert peer_input.groups is not None
    assert peer_input.folds is not None
    assert peer_input.fold_lambdas is not None
    assert peer_input.groups.shape == (5000,)
    assert peer_input.folds.shape == (5000,)
    np.testing.assert_array_equal(peer_input.fold_lambdas, np.full(5, np.inf))


def test_auto_oracle_preserves_partial_lambda_as_unavailable() -> None:
    """Auto oracle lambdas with empty-fold values do not become fake inputs."""

    peer_input = load_peer_input("sim_5000_seed42", oracle_id="sim_5000_auto")
    assert peer_input.fold_lambdas is None


def test_oracle_loader_returns_frozen_outputs() -> None:
    """The oracle loader returns finite reference outputs and metadata."""

    record = load_oracle("sim_5000_inf_n1")
    assert record.r_rejections == 163
    assert record.metadata["case_id"] == "sim_5000_inf_n1"
    assert np.all(np.isfinite(record.adjusted_pvalues))
    assert np.all(np.isfinite(record.weights))
