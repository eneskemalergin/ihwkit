"""Integration checks for tool-owned peer adapter result records."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_numpy_peer_emits_normalized_arrays(tmp_path: Path) -> None:
    """The pinned NumPy peer returns an explicit successful result record."""

    result_path = tmp_path / "numpy.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "peers" / "ihwkit_numpy.py"),
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


def test_missing_optional_peer_is_not_reported_as_success(tmp_path: Path) -> None:
    """An absent optional package produces an unavailable result status."""

    package_available = importlib.util.find_spec("pyihw") is not None
    result_path = tmp_path / "pyihw.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "peers" / "pyihw.py"),
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
    if not package_available:
        assert completed.returncode == 3
        assert document["status"] == "unavailable"
        assert document["error"]["type"] == "AdapterUnavailable"
    elif completed.returncode == 0:
        assert document["status"] == "ok"
    else:
        assert document["status"] == "unavailable"
        assert document["error"]["type"] == "AdapterUnavailable"
