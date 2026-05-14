"""
dp/ — Differential Privacy module for Credit Mitra
====================================================
Implements the mechanisms specified in:
  "Differential Privacy in Fintech: A Scientific & Legal Framework"
  Smart Narration Parser — Privacy Engineering Brief

Public API
----------
Stage 1 (PDF Extraction):
    from dp import extract_with_dp, privatise_record_count

Stage 2 (Payee Extraction):
    from dp import privatise_payee, analyse_payee_population

Budget tracking:
    from dp import budget
    print(budget.report_json())

Quick example:
    from dp import extract_with_dp, privatise_payee, budget

    # Stage 1 — privatise extracted records
    records = [{"deposits": "450.00", "withdrawals": "", "balance": "12,000.00", ...}]
    private_records, report1 = extract_with_dp(records, epsilon=2.0)

    # Stage 2 — privatise payee
    private_payee = privatise_payee("ZOMATO LI", epsilon=1.0)

    # Check budget
    print(budget.report_json())
"""

from dp.budget_tracker import budget, BudgetExhaustedError, PrivacyBudgetTracker

from dp.mechanisms import (
    laplace_noise,
    privatise_amount,
    privatise_count,
    privatise_transaction_record,
    randomised_response_binary,
    privatise_payee_token_presence,
    privatise_payee_token_frequencies,
    exponential_mechanism,
)

from dp.stage1_extraction import (
    extract_with_dp,
    privatise_record_count,
)

from dp.stage2_payee import (
    tokenise_payee,
    privatise_payee,
    analyse_payee_population,
)

__all__ = [
    # Budget
    "budget",
    "BudgetExhaustedError",
    "PrivacyBudgetTracker",
    # Core mechanisms
    "laplace_noise",
    "privatise_amount",
    "privatise_count",
    "privatise_transaction_record",
    "randomised_response_binary",
    "privatise_payee_token_presence",
    "privatise_payee_token_frequencies",
    "exponential_mechanism",
    # Stage 1
    "extract_with_dp",
    "privatise_record_count",
    # Stage 2
    "tokenise_payee",
    "privatise_payee",
    "analyse_payee_population",
]
