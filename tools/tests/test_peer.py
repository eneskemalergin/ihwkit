"""Tests for generated inputs, frozen R records, and the unified peer runner."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ihw import IHWResult, _p_adjust, _safe_divide, adjust_ihw
from tools.peer import (
    PARITY_GATE_IDS,
    REFERENCE_IDS,
    ReferenceRecord,
    _r_output_fold_count,
    load_peer_input,
    load_reference,
)

ROOT = Path(__file__).resolve().parents[2]


def test_named_synthetic_inputs_are_generated_deterministically() -> None:
    """Named tool inputs have readable recipes and stable arrays."""

    expected_sizes = {
        "sim_500_seed42": 500,
        "sim_1500_seed42": 1_500,
        "dense_500_seed42": 500,
        "sim_5000_seed42": 5_000,
        "sim_15000_seed42": 15_000,
        "sim_50000_seed42": 50_000,
    }
    for dataset_id, size in expected_sizes.items():
        first = load_peer_input(dataset_id)
        second = load_peer_input(dataset_id)
        assert first.size == size
        assert first.source_path.startswith("generated:")
        assert first.pvalues.shape == (size,)
        assert first.covariates.shape == (size,)
        assert first.truth_labels is not None
        np.testing.assert_array_equal(first.pvalues, second.pvalues)
        np.testing.assert_array_equal(first.covariates, second.covariates)


def test_frozen_r_records_are_self_contained() -> None:
    """Each retained R record contains its exact inputs, partitions, and outputs."""

    for reference_id in REFERENCE_IDS:
        record = load_reference(reference_id)
        peer_input = record.peer_input
        assert peer_input.reference_id == reference_id
        expected_size = 33_469 if record.spec.dataset_id == "airway" else 5_000
        assert peer_input.size == expected_size
        assert peer_input.groups is not None
        assert peer_input.folds is not None
        assert peer_input.fold_lambdas is not None
        assert record.metadata["reference_id"] == reference_id
        assert record.metadata["r_ihw_version"] == "1.40.0"
        assert np.all(np.isfinite(record.adjusted_pvalues))
        assert np.all(np.isfinite(record.weights))


def test_airway_input_comes_from_the_benchmark_record() -> None:
    """The real-data shape is available without root data or a download."""

    peer_input = load_peer_input("airway")

    assert peer_input.size == 33_469
    assert peer_input.source_path == "bench/data/airway_r_ihw_1_40_0.npz"
    assert "existing local" in peer_input.provenance.lower()


def test_production_fit_on_generated_input() -> None:
    """The production path returns finite arrays on a generated tool input."""

    peer_input = load_peer_input("sim_500_seed42")
    result = adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        0.1,
        nbins=4,
        nfolds=1,
        seed=42,
    )
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))


def test_auto_lambda_on_generated_input_is_finite() -> None:
    """The auto-lambda path remains finite on the larger generated input."""

    peer_input = load_peer_input("sim_5000_seed42")
    result = adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        0.1,
        nbins="auto",
        nfolds=5,
        lambdas="auto",
        seed=42,
    )
    assert result.nbins == 3
    assert result.nfolds == 5
    assert np.all(np.isfinite(result.weights))
    assert np.all(np.isfinite(result.adj_pvalues))


def test_numpy_peer_emits_normalized_arrays(tmp_path: Path) -> None:
    """The unified command emits arrays for the pinned NumPy baseline."""

    result_path = tmp_path / "numpy.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "peer.py"),
            "--method",
            "ihwkit_numpy",
            "--dataset",
            "sim_500_seed42",
            "--nbins",
            "4",
            "--nfolds",
            "1",
            "--include-arrays",
            "--result",
            str(result_path),
            "--quiet",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert document["status"] == "ok"
    assert document["method_id"] == "ihwkit_numpy"
    assert 0 <= document["rejection_count"] <= 500
    assert len(document["adjusted_pvalues"]) == 500
    assert len(document["weights"]) == 500


def test_r_one_bin_shortcut_reports_one_effective_fold() -> None:
    """R IHW reduces a requested cross-fit to one fold when only one bin exists."""

    assert _r_output_fold_count(1, 5) == 1
    assert _r_output_fold_count(3, 5) == 5


def test_unsupported_pyihw_is_not_reported_as_success(tmp_path: Path) -> None:
    """A missing or unreviewed pyihw version has explicit unavailable status."""

    supported = False
    if importlib.util.find_spec("pyihw") is not None:
        supported = importlib.metadata.version("pyihw") == "0.2.0"
    result_path = tmp_path / "pyihw.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "peer.py"),
            "--method",
            "pyihw",
            "--dataset",
            "sim_500_seed42",
            "--result",
            str(result_path),
            "--quiet",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert completed.returncode in {0, 3}, completed.stderr
    assert document["exit_code"] == completed.returncode
    if supported:
        assert document["status"] == "ok"
    else:
        assert document["status"] == "unavailable"
        assert document["error"]["type"] == "PeerUnavailable"


def test_production_path_matches_frozen_r_records() -> None:
    """Frozen groups and folds reproduce the retained R output vectors."""

    for reference_id in PARITY_GATE_IDS:
        record = load_reference(reference_id)
        result = _fit_reference(record)
        np.testing.assert_allclose(
            result.adj_pvalues,
            record.adjusted_pvalues,
            atol=1e-8,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            result.weights,
            record.weights,
            atol=1e-8,
            rtol=1e-6,
        )
        assert int(np.sum(result.adj_pvalues <= 0.1)) == record.r_rejections
        np.testing.assert_array_equal(result.folds, record.peer_input.folds)


def test_bh_on_r_weights_matches_frozen_r_records() -> None:
    """The local weighted-BH step agrees with each retained R record."""

    for reference_id in REFERENCE_IDS:
        record = load_reference(reference_id)
        adjusted = _p_adjust(
            _safe_divide(record.peer_input.pvalues, record.weights), "fdr_bh"
        )
        np.testing.assert_allclose(
            adjusted,
            record.adjusted_pvalues,
            atol=1e-8,
            rtol=1e-6,
        )
        assert int(np.sum(adjusted <= 0.1)) == record.r_rejections


def _fit_reference(record: ReferenceRecord) -> IHWResult:
    peer_input = record.peer_input
    assert peer_input.groups is not None
    assert peer_input.folds is not None
    assert peer_input.fold_lambdas is not None
    return adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        0.1,
        nbins=int(np.max(peer_input.groups)) + 1,
        nfolds=int(np.max(peer_input.folds)) + 1,
        groups=peer_input.groups,
        folds=peer_input.folds,
        fold_lambdas=peer_input.fold_lambdas,
        seed=peer_input.seed,
    )
