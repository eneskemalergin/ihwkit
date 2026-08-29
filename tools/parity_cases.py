from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / "data" / "oracles"


@dataclass(frozen=True)
class ParityCase:
    """Identify one frozen oracle replay case."""

    oracle_id: str

    def oracle_ready(self) -> bool:
        """Return whether both oracle files are present."""

        return (
            (ORACLES / f"{self.oracle_id}.npz").is_file()
            and (ORACLES / f"{self.oracle_id}.json").is_file()
        )

    @property
    def case_id(self) -> str:
        """Return the display name for the oracle case."""

        return self.oracle_id


def r_gold_cases() -> tuple[ParityCase, ...]:
    """Return the frozen synthetic R replay cases."""

    return (
        ParityCase(oracle_id="sim_5000_inf_n1"),
        ParityCase(oracle_id="sim_5000_inf_n5"),
    )


def available_r_gold_cases() -> tuple[ParityCase, ...]:
    """Return frozen replay cases whose files are available."""

    return tuple(c for c in r_gold_cases() if c.oracle_ready())
