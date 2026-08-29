"""Adapt the optional Julia IHW package to the peer result contract."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_contract import PeerInput
from tools.peers.runner import (
    AdapterUnavailable,
    FitResult,
    RunConfig,
    adapter_main,
)

JULIA_SCRIPT = Path(__file__).with_name("julia_ihw.jl")
IMPLEMENTATION_VERSION = "IndependentHypothesisWeighting.jl-adapter-runtime-version"


def fit(peer_input: PeerInput, config: RunConfig) -> FitResult:
    """Run the preliminary Julia adapter when its runtime and package exist."""

    julia = shutil.which("julia")
    if julia is None:
        raise AdapterUnavailable("julia is not installed")
    if peer_input.groups is not None or peer_input.folds is not None:
        raise AdapterUnavailable(
            "the Julia adapter does not accept frozen groups and folds"
        )
    nfolds = 5 if config.nfolds is None else config.nfolds
    nbins = (
        max(1, min(40, peer_input.size // 1500))
        if config.nbins == "auto"
        else config.nbins
    )
    if not isinstance(nbins, int):
        raise TypeError("Julia adapter could not resolve nbins")
    with tempfile.TemporaryDirectory(prefix="ihwkit_julia_peer_") as temporary:
        directory = Path(temporary)
        pvalues_path = directory / "pvalues.txt"
        covariates_path = directory / "covariates.txt"
        output_prefix = directory / "result"
        np.savetxt(pvalues_path, peer_input.pvalues, fmt="%.17g")
        np.savetxt(covariates_path, peer_input.covariates, fmt="%.17g")
        command = [
            julia,
            "--startup-file=no",
            str(JULIA_SCRIPT),
            "--pvalues",
            str(pvalues_path),
            "--covariates",
            str(covariates_path),
            "--alpha",
            str(config.alpha),
            "--nbins",
            str(nbins),
            "--nfolds",
            str(nfolds),
            "--lambda-policy",
            config.lambda_policy,
            "--seed",
            str(config.seed),
            "--output-prefix",
            str(output_prefix),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = completed.stderr.strip() or completed.stdout.strip()
        if completed.returncode == 3:
            raise AdapterUnavailable(detail or "Julia IHW is unavailable")
        if completed.returncode != 0:
            raise RuntimeError(detail or "Julia IHW adapter failed")
        adjusted = _read_vector(output_prefix.with_suffix(".adj.txt"))
        weights = _read_vector(output_prefix.with_suffix(".weights.txt"))
        rejection_text = output_prefix.with_suffix(".rejections.txt").read_text(
            encoding="utf-8"
        )
    return FitResult(adjusted, weights, int(rejection_text.strip()))


def _read_vector(path: Path) -> np.ndarray:
    """Read one numeric vector written by the Julia adapter."""

    return np.atleast_1d(np.loadtxt(path, dtype=np.float64))


if __name__ == "__main__":
    raise SystemExit(adapter_main("julia_ihw", IMPLEMENTATION_VERSION, fit))
