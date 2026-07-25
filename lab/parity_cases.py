from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class ParityCase:
    case_id: str
    sim_path: Path
    oracle_path: Path
    nfolds: int
    n: int
    lambdas: str = "inf"
    alpha: float = 0.1
    nbins: int = 4
    seed: int = 1

    def oracle_ready(self) -> bool:
        return self.oracle_path.is_file()

    def sim_ready(self) -> bool:
        return self.sim_path.is_file()

    def replay_kwargs(self) -> dict:
        return {
            "nbins": self.nbins,
            "nfolds": self.nfolds,
            "seed": self.seed,
            "lp_backend": "highs",
        }


def r_gold_cases() -> tuple[ParityCase, ...]:
    return (
        ParityCase(
            case_id="sim_n2000_inf_n1",
            sim_path=FIXTURES / "sim_n2000_seed1.npz",
            oracle_path=FIXTURES / "r_inf_n1.npz",
            nfolds=1,
            n=2000,
            seed=1,
        ),
        ParityCase(
            case_id="sim_n2000_inf_n5",
            sim_path=FIXTURES / "sim_n2000_seed1.npz",
            oracle_path=FIXTURES / "r_inf_n5.npz",
            nfolds=5,
            n=2000,
            seed=1,
        ),
        ParityCase(
            case_id="sim_n5000_inf_n1",
            sim_path=FIXTURES / "sim_n5000_seed42.npz",
            oracle_path=FIXTURES / "r_inf_n1_n5000.npz",
            nfolds=1,
            n=5000,
            seed=42,
        ),
        ParityCase(
            case_id="sim_n5000_inf_n5",
            sim_path=FIXTURES / "sim_n5000_seed42.npz",
            oracle_path=FIXTURES / "r_inf_n5_n5000.npz",
            nfolds=5,
            n=5000,
            seed=42,
        ),
    )


def available_r_gold_cases() -> tuple[ParityCase, ...]:
    return tuple(c for c in r_gold_cases() if c.oracle_ready())
