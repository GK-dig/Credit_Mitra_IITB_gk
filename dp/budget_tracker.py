"""
dp/budget_tracker.py
--------------------
RDP-based privacy budget tracker for the Credit Mitra pipeline.

Tracks cumulative epsilon consumed across all 5 pipeline stages.
Raises BudgetExhaustedError if a query would push total epsilon
over the configured hard limit.

Theory basis:
  - Basic composition: total ε = Σ εᵢ  (Dwork & Roth 2014, Prop 3.13)
  - Hard limit set at ε_total = 8.0 per data subject per the spec (Section 6, Stage 5)
  - δ tracked separately; set to 1e-5 globally (spec Section 2.1)

Usage:
    tracker = PrivacyBudgetTracker(epsilon_limit=8.0, delta=1e-5)
    tracker.consume("stage_1_amounts", epsilon=2.0)
    tracker.consume("stage_2_payee",   epsilon=1.0)
    print(tracker.report())
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


class BudgetExhaustedError(Exception):
    """Raised when a query would push cumulative ε over the hard limit."""
    pass


@dataclass
class BudgetEntry:
    stage: str
    epsilon: float
    delta: float
    timestamp: str
    note: str


@dataclass
class PrivacyBudgetTracker:
    """
    Thread-safe cumulative DP budget tracker.

    Parameters
    ----------
    epsilon_limit : float
        Hard cap on total ε consumed (default 8.0 per spec Section 6).
    delta : float
        Global δ value — probability ε bound is violated (default 1e-5).
    """
    epsilon_limit: float = 8.0
    delta: float = 1e-5
    _ledger: List[BudgetEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def consumed(self) -> float:
        """Total ε consumed so far (basic composition sum)."""
        return sum(e.epsilon for e in self._ledger)

    def remaining(self) -> float:
        """ε budget remaining before hard limit."""
        return max(0.0, self.epsilon_limit - self.consumed())

    def consume(
        self,
        stage: str,
        epsilon: float,
        note: str = "",
        dry_run: bool = False,
    ) -> float:
        """
        Consume epsilon units from the budget.

        Parameters
        ----------
        stage : str
            Label for this query (e.g. "stage_1_amounts").
        epsilon : float
            Privacy cost of this query.
        note : str
            Optional human-readable description.
        dry_run : bool
            If True, check budget without consuming it.

        Returns
        -------
        float
            Remaining budget after this consumption.

        Raises
        ------
        BudgetExhaustedError
            If this query would exceed the hard limit.
        ValueError
            If epsilon <= 0.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        with self._lock:
            if self.consumed() + epsilon > self.epsilon_limit:
                raise BudgetExhaustedError(
                    f"Query '{stage}' requests ε={epsilon:.4f} but only "
                    f"ε={self.remaining():.4f} remains "
                    f"(limit={self.epsilon_limit}, consumed={self.consumed():.4f})."
                )

            if dry_run:
                return self.remaining() - epsilon

            entry = BudgetEntry(
                stage=stage,
                epsilon=epsilon,
                delta=self.delta,
                timestamp=datetime.now(timezone.utc).isoformat(),
                note=note,
            )
            self._ledger.append(entry)
            return self.remaining()

    def report(self) -> Dict:
        """Return a structured dict of the full budget ledger."""
        with self._lock:
            return {
                "epsilon_limit": self.epsilon_limit,
                "epsilon_consumed": round(self.consumed(), 6),
                "epsilon_remaining": round(self.remaining(), 6),
                "delta": self.delta,
                "entries": [
                    {
                        "stage": e.stage,
                        "epsilon": e.epsilon,
                        "delta": e.delta,
                        "timestamp": e.timestamp,
                        "note": e.note,
                    }
                    for e in self._ledger
                ],
            }

    def report_json(self, indent: int = 2) -> str:
        """Return the report as a formatted JSON string."""
        return json.dumps(self.report(), indent=indent)

    def reset(self) -> None:
        """
        Reset the ledger. Use only in tests.
        DO NOT call in production — budget resets remove the audit trail.
        """
        with self._lock:
            self._ledger.clear()


# ---------------------------------------------------------------------------
# Module-level singleton — import and use this across all pipeline stages
# ---------------------------------------------------------------------------
# Each pipeline stage imports this and calls budget.consume(...)
# ε values come directly from the spec (Section 6):
#   Stage 1 — extraction amounts:  ε = 2.0
#   Stage 2 — payee tokens:        ε = 1.0
#   Stage 3 — merchant binary:     ε = 3.0
#   Stage 4 — category histogram:  ε = 2.0
#   Stage 5 — storage/aggregate:   tracked cumulatively, alert at 8.0
# Total: 2 + 1 + 3 + 2 = 8.0 — exactly at the hard limit.

budget = PrivacyBudgetTracker(epsilon_limit=8.0, delta=1e-5)
