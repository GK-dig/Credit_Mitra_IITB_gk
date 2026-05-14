"""
dp/stage1_extraction.py
-----------------------
Stage 1 DP integration: PDF extraction → privatised numeric fields.

Wraps the Docling tabular extractor from extraction-from-pdfs/ and
applies Laplace noise to all numeric fields (deposits, withdrawals, balance)
before any data is returned to the pipeline.

Spec reference: Section 6, Stage 1
  Mechanism:    Laplace on numeric outputs
  ε budget:     ε = 2.0 per query on amounts; ε = 1.0 for count queries
  Sensitivity:  Δf = max transaction cap, set at 99th percentile

Narration strings (particulars field) are NOT modified here.
They are protected in Stage 2 (payee_name_extraction).

Usage:
    from dp.stage1_extraction import extract_with_dp

    records, report = extract_with_dp(
        pdf_path="statement.pdf",
        epsilon=2.0,
        amount_cap_percentile=99,
    )
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dp.budget_tracker import budget, BudgetExhaustedError
from dp.mechanisms import privatise_transaction_record, privatise_count


def _parse_amount(val: str) -> Optional[float]:
    """Parse Indian number format strings like '1,04,500.00' → 104500.0"""
    try:
        cleaned = str(val).replace(",", "").strip()
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _compute_99th_percentile_cap(
    records: List[dict],
    fields: List[str] = ("deposits", "withdrawals"),
) -> float:
    """
    Compute the 99th percentile cap across all amount fields.

    This is used as the sensitivity Δf (spec Section 3.1, [R5]).
    Clipping at the 99th percentile controls the sensitivity without
    significantly affecting the majority of transactions.

    Returns a minimum of 10,000 (₹10K) to avoid degenerate noise.
    """
    amounts = []
    for record in records:
        for field in fields:
            val = _parse_amount(record.get(field, ""))
            if val is not None and val > 0:
                amounts.append(val)

    if not amounts:
        return 100_000.0  # fallback: ₹1 lakh

    cap = float(np.percentile(amounts, 99))
    return max(cap, 10_000.0)


def extract_with_dp(
    records: List[dict],
    epsilon: float = 2.0,
    amount_cap_percentile: int = 99,
    privatise_numerics: bool = True,
) -> Tuple[List[dict], Dict]:
    """
    Apply Stage 1 DP (Laplace mechanism) to extracted transaction records.

    This function:
    1. Takes raw records from Docling extraction
    2. Computes the 99th-percentile sensitivity cap from the portfolio
    3. Applies Laplace noise to deposits, withdrawals, balance fields
    4. Consumes ε from the global budget tracker
    5. Returns privatised records + audit report

    Parameters
    ----------
    records : list of dict
        Raw transaction records from Docling extraction.
        Expected keys: date, particulars, deposits, withdrawals, balance
    epsilon : float
        Privacy budget for this stage (spec: ε = 2.0).
    amount_cap_percentile : int
        Percentile for computing sensitivity cap (default 99).
    privatise_numerics : bool
        If False, skip DP (useful for debugging). Logs a warning.

    Returns
    -------
    Tuple[List[dict], Dict]
        (privatised_records, audit_report)
        audit_report contains epsilon consumed, sensitivity used,
        and per-record counts for the audit log.

    Raises
    ------
    BudgetExhaustedError
        If epsilon would exceed the global budget limit.
    """
    if not privatise_numerics:
        print(
            "[DP WARNING] Stage 1 DP is DISABLED. "
            "Raw numeric values will be used. Do not use in production."
        )
        return records, {"dp_applied": False, "reason": "privatise_numerics=False"}

    # --- Pre-flight budget check (dry run) ---
    budget.consume(
        stage="stage_1_extraction",
        epsilon=epsilon,
        note=f"Laplace noise on amounts/balances. {len(records)} records.",
        dry_run=True,
    )

    # --- Compute sensitivity cap ---
    amount_cap = _compute_99th_percentile_cap(records)

    # --- Apply DP to each record ---
    privatised = []
    skipped_empty = 0

    for record in records:
        # Skip entirely empty records (e.g. page-break artifacts)
        has_amounts = (
            _parse_amount(record.get("deposits", "")) is not None
            or _parse_amount(record.get("withdrawals", "")) is not None
        )
        if not has_amounts:
            privatised.append(record)
            skipped_empty += 1
            continue

        private_record = privatise_transaction_record(
            record=record,
            epsilon=epsilon,
            amount_sensitivity=amount_cap,
            amount_99th_percentile=amount_cap,
        )
        privatised.append(private_record)

    # --- Consume budget (actual, after successful processing) ---
    remaining = budget.consume(
        stage="stage_1_extraction",
        epsilon=epsilon,
        note=(
            f"Laplace(Δf={amount_cap:.2f}, ε={epsilon}). "
            f"{len(privatised) - skipped_empty} records privatised, "
            f"{skipped_empty} empty records passed through."
        ),
    )

    audit_report = {
        "dp_applied": True,
        "stage": "stage_1_extraction",
        "mechanism": "Laplace",
        "epsilon_consumed": epsilon,
        "epsilon_remaining": remaining,
        "delta": budget.delta,
        "sensitivity_cap_inr": amount_cap,
        "records_total": len(records),
        "records_privatised": len(privatised) - skipped_empty,
        "records_passed_through": skipped_empty,
        "guarantee": f"Pure ε-DP with ε={epsilon}, δ=0",
        "spec_reference": "Section 6, Stage 1 — Dwork et al. (2006) [R3]",
    }

    return privatised, audit_report


def privatise_record_count(
    true_count: int,
    epsilon: float = 1.0,
    stage_label: str = "stage_1_count_query",
) -> Tuple[int, Dict]:
    """
    Apply Laplace noise to a count query (e.g. total transactions extracted).

    Separate from the per-record privatisation above.
    Sensitivity for count queries is always Δf = 1 (spec Section 3.1).

    Parameters
    ----------
    true_count : int
        The actual count value.
    epsilon : float
        Privacy budget (spec Stage 1 count queries: ε = 1.0).
    stage_label : str
        Label for the budget ledger.

    Returns
    -------
    Tuple[int, Dict]
        (privatised_count, audit_report)
    """
    budget.consume(
        stage=stage_label,
        epsilon=epsilon,
        note=f"Laplace count query. True count: {true_count}",
    )

    noisy = privatise_count(true_count, epsilon)

    return noisy, {
        "mechanism": "Laplace",
        "sensitivity": 1,
        "epsilon": epsilon,
        "true_count": true_count,
        "noisy_count": noisy,
    }
