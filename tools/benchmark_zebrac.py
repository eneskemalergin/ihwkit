"""Run reproducible zebrac measurements through the unified peer command."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.peer import METHODS, PeerDataError, PeerInput, load_peer_input

PEER_SCRIPT = ROOT / "tools" / "peer.py"
DEFAULT_METHODS = METHODS
METRIC_FIELDS = (
    "wall_time",
    "peak_rss",
    "minor_faults",
    "major_faults",
    "cpu_cycles",
    "instructions",
    "cache_references",
    "cache_misses",
    "branch_misses",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run peer preflights and a zebrac comparison."""

    args = _argument_parser().parse_args(argv)
    if args.name is None:
        args.name = f"zebrac_{args.dataset}_{args.lambda_policy}"
    result_dir = (ROOT / args.result_dir).resolve()
    _ensure_result_dir(result_dir)
    try:
        peer_input = load_peer_input(args.dataset, oracle_id=args.oracle)
    except PeerDataError as exc:
        print(f"peer data error: {exc}", file=sys.stderr)
        return 2
    methods = tuple(args.methods)
    preflight_documents: dict[str, dict[str, object]] = {}
    available_methods: list[str] = []
    for method_id in methods:
        result_path = result_dir / f"{args.name}.{method_id}.json"
        document = _run_preflight(method_id, args, result_path)
        preflight_documents[method_id] = document
        if document.get("status") == "ok":
            available_methods.append(method_id)
    if "ihwkit_numpy_numba" not in available_methods:
        _write_metadata(
            result_dir / f"{args.name}.metadata.json",
            args,
            peer_input,
            preflight_documents,
            None,
            "production method is unavailable or failed its preflight",
        )
        return 1
    zebrac_info = _available_zebrac()
    if zebrac_info is None:
        _write_metadata(
            result_dir / f"{args.name}.metadata.json",
            args,
            peer_input,
            preflight_documents,
            None,
            "zebrac is unavailable or did not report a version",
        )
        return 1
    zebrac_path, zebrac_version = zebrac_info
    raw_path = result_dir / f"{args.name}.zebrac.json"
    command_strings = [
        _peer_command(method_id, args) for method_id in available_methods
    ]
    zebrac_command = [
        str(zebrac_path),
        "--color",
        "never",
        "--warmup",
        str(args.warmup),
        "--min-samples",
        str(args.min_samples),
        "--max-samples",
        str(args.max_samples),
        "--duration",
        str(args.duration),
        f"--json={raw_path}",
        "--",
        *command_strings,
    ]
    completed = subprocess.run(
        zebrac_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    raw_document = _read_json(raw_path) if raw_path.is_file() else None
    metadata_path = result_dir / f"{args.name}.metadata.json"
    _write_metadata(
        metadata_path,
        args,
        peer_input,
        preflight_documents,
        raw_document,
        None if completed.returncode == 0 else _process_detail(completed),
        zebrac_version,
        zebrac_path,
    )
    if not args.quiet:
        output = completed.stdout.strip()
        if output:
            print(output)
        if completed.stderr.strip():
            print(completed.stderr.strip(), file=sys.stderr)
        print(f"wrote {metadata_path.relative_to(ROOT)}")
    return completed.returncode


def _run_preflight(
    method_id: str, args: argparse.Namespace, result_path: Path
) -> dict[str, object]:
    """Run one method outside zebrac and return its result document."""

    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(PEER_SCRIPT),
        "--method",
        method_id,
        "--dataset",
        args.dataset,
        "--alpha",
        str(args.alpha),
        "--nbins",
        str(args.nbins),
        "--lambda-policy",
        args.lambda_policy,
        "--seed",
        str(args.seed),
        "--result",
        str(result_path),
        "--quiet",
    ]
    if args.oracle is not None:
        command.extend(["--oracle", args.oracle])
    if args.nfolds is not None:
        command.extend(["--nfolds", str(args.nfolds)])
    if args.adjustment_type != "bh":
        command.extend(["--adjustment-type", args.adjustment_type])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result_path.is_file():
        document = _read_json(result_path)
    else:
        document = {
            "method_id": method_id,
            "dataset_id": args.dataset,
            "status": "error",
            "exit_code": completed.returncode,
            "error": {
                "type": "MissingResultRecord",
                "message": _process_detail(completed),
            },
        }
    return document


def _peer_command(method_id: str, args: argparse.Namespace) -> str:
    """Build one direct command string accepted by zebrac."""

    tokens = [
        sys.executable,
        str(PEER_SCRIPT),
        "--method",
        method_id,
        "--dataset",
        args.dataset,
        "--alpha",
        str(args.alpha),
        "--nbins",
        str(args.nbins),
        "--lambda-policy",
        args.lambda_policy,
        "--seed",
        str(args.seed),
        "--quiet",
    ]
    if args.oracle is not None:
        tokens.extend(["--oracle", args.oracle])
    if args.nfolds is not None:
        tokens.extend(["--nfolds", str(args.nfolds)])
    if args.adjustment_type != "bh":
        tokens.extend(["--adjustment-type", args.adjustment_type])
    return " ".join(shlex.quote(token) for token in tokens)


def _available_zebrac() -> tuple[Path, str] | None:
    executable = shutil.which("zebrac")
    if executable is None:
        return None
    path = Path(executable).resolve()
    completed = subprocess.run(
        [str(path), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    version = completed.stdout.strip()
    if completed.returncode != 0 or not version:
        return None
    return path, version


def _write_metadata(
    path: Path,
    args: argparse.Namespace,
    peer_input: PeerInput,
    preflight_documents: dict[str, dict[str, object]],
    raw_document: dict[str, object] | None,
    error: str | None,
    zebrac_version: str | None = None,
    zebrac_path: Path | None = None,
) -> None:
    """Write metadata tying peer records to raw zebrac metrics."""

    dataset_id = peer_input.dataset_id
    methods: dict[str, object] = {}
    for method_id, document in preflight_documents.items():
        methods[method_id] = {
            "status": document.get("status"),
            "version": document.get("version"),
            "exit_code": document.get("exit_code"),
            "error": document.get("error"),
            "rejection_count": document.get("rejection_count"),
            "weights_available": document.get("weights_available"),
        }
    if raw_document is not None:
        for result in raw_document.get("results", []):
            if isinstance(result, dict):
                command = result.get("command")
                method_id = _method_from_command(command)
                if method_id is not None:
                    methods.setdefault(method_id, {})
                    methods[method_id] = {
                        **_mapping(methods[method_id]),
                        "sample_count": result.get("sample_count"),
                        "failed_sample_count": result.get("failed_sample_count"),
                        "metrics": {
                            field: result.get(field) for field in METRIC_FIELDS
                        },
                    }
    document: dict[str, object] = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "measurement_scope": "process_startup_and_algorithm",
        "dataset_id": dataset_id,
        "data": {
            "path": peer_input.source_path,
            "size": peer_input.size,
            "seed": peer_input.seed,
            "provenance": peer_input.provenance,
        },
        "configuration": {
            "alpha": args.alpha,
            "nbins": args.nbins,
            "nfolds": args.nfolds,
            "lambda_policy": args.lambda_policy,
            "adjustment_type": args.adjustment_type,
            "seed": args.seed,
            "oracle_id": args.oracle,
            "warmup": args.warmup,
            "min_samples": args.min_samples,
            "max_samples": args.max_samples,
            "duration_ms": args.duration,
        },
        "zebrac": {
            "path": None if zebrac_path is None else str(zebrac_path),
            "version": zebrac_version,
        },
        "methods": methods,
        "error": error,
    }
    if raw_document is not None:
        document["zebrac_raw"] = raw_document
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oracle")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--nbins", default="auto")
    parser.add_argument("--nfolds", type=int)
    parser.add_argument("--lambda-policy", choices=("inf", "auto"), default="inf")
    parser.add_argument("--adjustment-type", choices=("bh", "bonferroni"), default="bh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--result-dir", type=Path, default=Path("tmp/results"))
    parser.add_argument("--name")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _ensure_result_dir(path: Path) -> None:
    """Ensure benchmark output stays below the ignored temporary directory."""

    temporary_root = (ROOT / "tmp").resolve()
    if not path.is_relative_to(temporary_root):
        raise SystemExit(f"result directory must be under {temporary_root}")
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return {str(key): item for key, item in value.items()}


def _process_detail(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stderr.strip() or completed.stdout.strip() or "process failed"


def _method_from_command(command: object) -> str | None:
    if not isinstance(command, str):
        return None
    try:
        tokens = shlex.split(command)
        index = tokens.index("--method")
    except (ValueError, IndexError):
        return None
    if index + 1 < len(tokens) and tokens[index + 1] in METHODS:
        return tokens[index + 1]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
