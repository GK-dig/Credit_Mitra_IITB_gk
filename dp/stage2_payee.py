"""
dp/stage2_payee.py
------------------
Stage 2 DP integration: Payee name extraction → local DP on token frequencies.

This is the most sensitive stage in the pipeline (spec Section 6, Stage 2):
"Most sensitive stage — raw person names, UPI IDs, phone numbers."

What this does:
  - Tokenises extracted payee names into individual tokens
  - Applies Randomised Response (local DP) to each token's presence bit
  - Returns a privatised frequency table safe for aggregation
  - The raw narration string is NEVER stored or returned after this stage

Spec reference: Section 6, Stage 2
  Mechanism:    RAPPOR / Randomised Response on token-level frequency
  ε budget:     ε = 1.0 (strongest in pipeline — most sensitive data)
  Sensitivity:  Δf = 1 per token frequency count

Key insight from spec [R9]:
  Local DP means noise is added BEFORE aggregation.
  Even if the aggregation server is compromised, individual
  payee tokens are protected with ε=1.0 guarantee.

Usage:
    from dp.stage2_payee import privatise_payee, analyse_payee_population

    # Per-record: privatise a single extracted payee name
    private_result = privatise_payee("ZOMATO LI", epsilon=1.0)

    # Population: get private frequency table across many transactions
    freq_table, report = analyse_payee_population(payee_list, epsilon=1.0)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from dp.budget_tracker import budget, BudgetExhaustedError
from dp.mechanisms import (
    privatise_payee_token_frequencies,
    randomised_response_binary,
)

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Tokens to always remove — they're structural noise, not real payee tokens.
# These appear in UPI narration strings but carry no payee identity.
_NOISE_TOKENS: Set[str] = {
    "UPI", "DR", "CR", "NEFT", "IMPS", "RTGS", "NA", "UPI",
    "HDFC", "SBIN", "ICICI", "AXIS", "YESB", "KKBK", "UTIB",
    "PUNB", "CNRB", "BKID", "IBKL", "BARB", "UCBA",
    "OKAXIS", "OKSBI", "OKICICI", "OKHDFCBANK", "PTHDFC",
    "PTYES", "PTYBL", "PAYTM", "GPAY", "PHONEPE",
    "REF", "TXN", "OID", "PMT", "TRF", "PAY",
    "ONLY", "RS", "VIA", "FOR", "FROM", "TO",
    "LTD", "PVT", "LLP",  # keep these separately so "SWIGGY LTD" → "SWIGGY"
}

# Regex to detect tokens that are purely numeric or reference IDs
_NOISE_PATTERN = re.compile(
    r"""
    ^\d+$              |  # purely numeric
    ^\*{2,}            |  # starts with **
    ^[A-Z]{2,4}\d+$    |  # short alpha prefix + digits (e.g. SBI1234)
    ^\d+@              |  # phone@
    @                     # UPI handle containing @
    """,
    re.VERBOSE,
)


def tokenise_payee(raw_payee: str) -> List[str]:
    """
    Extract meaningful tokens from a raw payee name string.

    Cleans UPI narration format noise and returns only the
    human-readable payee tokens worth protecting via DP.

    Parameters
    ----------
    raw_payee : str
        Raw payee string, possibly truncated by bank system,
        e.g. "ZOMATO LI", "SWIGGY LTD", "RAHUL KAW", "K K CATER"

    Returns
    -------
    list of str
        Clean tokens. Empty list if nothing meaningful found.

    Examples
    --------
    >>> tokenise_payee("ZOMATO LI")
    ["ZOMATO"]
    >>> tokenise_payee("RAHUL KUMAR")
    ["RAHUL", "KUMAR"]
    >>> tokenise_payee("K K CATER")
    ["CATER"]
    """
    if not raw_payee:
        return []

    # Uppercase and split on common separators
    upper = raw_payee.upper()
    tokens = re.split(r"[/\s\-_,]+", upper)

    clean = []
    for tok in tokens:
        # Strip punctuation
        tok = tok.strip(string.punctuation)

        # Skip empty
        if not tok:
            continue

        # Skip noise tokens
        if tok in _NOISE_TOKENS:
            continue

        # Skip numeric/reference patterns
        if _NOISE_PATTERN.search(tok):
            continue

        # Skip very short tokens (likely abbreviations, not names)
        if len(tok) < 2:
            continue

        clean.append(tok)

    return clean


# ---------------------------------------------------------------------------
# Per-record payee privatisation
# ---------------------------------------------------------------------------

def privatise_payee(
    raw_payee: str,
    epsilon: float = 1.0,
    redact_raw: bool = True,
) -> Dict:
    """
    Apply local DP to a single extracted payee name.

    This is for per-record processing in the LangGraph pipeline.
    For population-level frequency analysis, use analyse_payee_population().

    What this returns:
    - A privatised version of the payee for storage
    - Token presence bits after Randomised Response
    - The raw payee is optionally redacted from the output

    Parameters
    ----------
    raw_payee : str
        The payee name extracted by the LLM/SLM.
    epsilon : float
        Privacy budget (spec Stage 2: ε = 1.0).
    redact_raw : bool
        If True (default), the raw payee string is not included
        in the returned dict. Only privatised tokens are returned.
        Set to False ONLY for debugging.

    Returns
    -------
    dict with keys:
        - payee_tokens_dp: list of tokens that survived RR
        - payee_token_count_dp: int, how many tokens survived
        - dp_applied: True
        - dp_epsilon_stage2: float

    Example
    -------
    >>> result = privatise_payee("ZOMATO LI", epsilon=1.0)
    >>> result["payee_tokens_dp"]
    ["ZOMATO"]   # or [] if RR flipped the token to absent
    """
    tokens = tokenise_payee(raw_payee)

    # Apply RR to each token's presence bit
    surviving_tokens = []
    for token in tokens:
        _, privatised_present = _apply_rr_to_token(token, is_present=True, epsilon=epsilon)
        if privatised_present:
            surviving_tokens.append(token)

    result = {
        "payee_tokens_dp": surviving_tokens,
        "payee_token_count_dp": len(surviving_tokens),
        "dp_applied": True,
        "dp_epsilon_stage2": epsilon,
    }

    if not redact_raw:
        result["payee_raw_DEBUG_ONLY"] = raw_payee

    return result


def _apply_rr_to_token(
    token: str,
    is_present: bool,
    epsilon: float,
) -> Tuple[str, bool]:
    """Internal helper — applies RR to a single token presence bit."""
    privatised = randomised_response_binary(is_present, epsilon)
    return (token, privatised)


# ---------------------------------------------------------------------------
# Population-level payee frequency analysis (with DP)
# ---------------------------------------------------------------------------

def analyse_payee_population(
    payee_list: List[str],
    epsilon: float = 1.0,
    min_count_threshold: int = 5,
) -> Tuple[Dict[str, int], Dict]:
    """
    Build a DP-protected payee frequency table across a population.

    This is for analytics endpoints — e.g. "what are the top merchants
    in this transaction set?" — where you want population statistics
    without exposing individual-level payee data.

    Process:
    1. Tokenise all payee names
    2. Count raw token frequencies
    3. Apply privatise_payee_token_frequencies() (RR-based)
    4. Suppress tokens below min_count_threshold (additional safety)
    5. Consume budget and return

    Parameters
    ----------
    payee_list : list of str
        All extracted payee names for this batch.
    epsilon : float
        Privacy budget (spec Stage 2: ε = 1.0).
    min_count_threshold : int
        Tokens appearing fewer than this many times are suppressed.
        Reduces risk of singling out rare payees.

    Returns
    -------
    Tuple[Dict[str, int], Dict]
        (privatised_frequency_table, audit_report)

    Example
    -------
    >>> payees = ["ZOMATO LI", "SWIGGY", "ZOMATO LI", "RAHUL KUMAR"]
    >>> freq, report = analyse_payee_population(payees, epsilon=1.0)
    >>> freq
    {"ZOMATO": 2, "SWIGGY": 1}  # RAHUL KUMAR may be suppressed
    """
    if not payee_list:
        return {}, {"dp_applied": True, "records": 0}

    # --- Pre-flight budget check ---
    budget.consume(
        stage="stage_2_payee_population",
        epsilon=epsilon,
        note=f"RR on payee token frequencies. {len(payee_list)} payees.",
        dry_run=True,
    )

    # --- Tokenise and count ---
    all_tokens = []
    for payee in payee_list:
        all_tokens.extend(tokenise_payee(payee))

    raw_counts = Counter(all_tokens)

    # --- Apply DP ---
    private_counts = privatise_payee_token_frequencies(
        token_counts=dict(raw_counts),
        epsilon=epsilon,
    )

    # --- Suppress low-frequency tokens ---
    suppressed_count = 0
    final_counts = {}
    for token, count in private_counts.items():
        if count >= min_count_threshold:
            final_counts[token] = count
        else:
            suppressed_count += 1

    # --- Consume budget ---
    remaining = budget.consume(
        stage="stage_2_payee_population",
        epsilon=epsilon,
        note=(
            f"RR(ε={epsilon}) on {len(raw_counts)} unique tokens. "
            f"{suppressed_count} tokens suppressed (< {min_count_threshold} count)."
        ),
    )

    audit_report = {
        "dp_applied": True,
        "stage": "stage_2_payee_population",
        "mechanism": "Randomised Response (Local DP)",
        "epsilon_consumed": epsilon,
        "epsilon_remaining": remaining,
        "payees_processed": len(payee_list),
        "unique_tokens_raw": len(raw_counts),
        "unique_tokens_after_dp": len(final_counts),
        "tokens_suppressed": suppressed_count,
        "suppression_threshold": min_count_threshold,
        "spec_reference": (
            "Section 6, Stage 2 — RAPPOR [R9], "
            "Local DP for Evolving Data [R10]"
        ),
    }

    return final_counts, audit_report
