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
    METHODS,
    PARITY_GATE_IDS,
    REFERENCE_IDS,
    ReferenceRecord,
    load_peer_input,
    load_reference,
)
from tools.simulators import SCENARIO_BUILDERS

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


def test_scipy_peer_emits_normalized_arrays_when_available(tmp_path: Path) -> None:
    """The unified command exposes only the retained solver baseline."""

    result_path = tmp_path / "scipy.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "peer.py"),
            "--method",
            "ihwkit_scipy",
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
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert document["method_id"] == "ihwkit_scipy"
    if importlib.util.find_spec("scipy") is None:
        assert completed.returncode == 3
        assert document["status"] == "unavailable"
    else:
        assert completed.returncode == 0, completed.stderr
        assert document["status"] == "ok"
        assert 0 <= document["rejection_count"] <= 500
        assert len(document["adjusted_pvalues"]) == 500
        assert len(document["weights"]) == 500


def test_benchmark_methods_are_distinct_implementations() -> None:
    assert METHODS == ("ihwkit", "ihwkit_scipy", "pyihw", "r_ihw")


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


def test_unregularized_production_matches_frozen_airway_vectors() -> None:
    """The direct default path handles the retained real-data shape."""

    for reference_id in ("airway_inf_n1", "airway_inf_n5"):
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


def test_known_default_false_infeasibility_is_resolved() -> None:
    """The seed-2034 feasible default fit remains a regression case."""

    draw = SCENARIO_BUILDERS["mixture_mild"](3_000, 2_034)
    result = adjust_ihw(
        draw.pvalues,
        draw.covariates,
        0.1,
        nfolds=5,
        seed=2_034,
    )

    assert int(np.sum(result.adj_pvalues <= 0.1)) == 51
    np.testing.assert_allclose(np.mean(result.weights), 1.0)


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
    return adjust_ihw(
        peer_input.pvalues,
        peer_input.covariates,
        0.1,
        nbins=int(np.max(peer_input.groups)) + 1,
        nfolds=int(np.max(peer_input.folds)) + 1,
        groups=peer_input.groups,
        folds=peer_input.folds,
        seed=peer_input.seed,
    )
