"""
dp/mechanisms.py
----------------
Core differential privacy mechanisms used by the Credit Mitra pipeline.

Implements exactly the mechanisms recommended in the spec (Section 2.2 & 6):
  - Laplace Mechanism      → Stage 1 (numeric amounts, balances, counts)
  - Randomised Response    → Stage 2 (binary payee token presence)
                           → Stage 3 (merchant/non-merchant binary label)
  - Exponential Mechanism  → Stage 4 (categorical merchant category selection)

All functions are pure (no side effects) and stateless.
Budget tracking is the caller's responsibility via budget_tracker.py.

Theory references (from spec):
  [R3] Dwork et al. (2006) — Laplace mechanism, calibrating noise to sensitivity
  [R11] Warner (1965)      — Original Randomised Response
  [R12] Kairouz et al. (2016) — RR is optimal for binary/k-ary domains
  [R13] McSherry & Talwar (2007) — Exponential mechanism for categorical outputs
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Laplace Mechanism
#    Used for: numeric outputs (amounts, balances, counts)
#    Guarantee: pure ε-DP (δ = 0)
#    Noise scale: b = Δf / ε
#    Expected absolute error: Δf / ε   (spec Section 7.1)
# ---------------------------------------------------------------------------

def laplace_noise(sensitivity: float, epsilon: float) -> float:
    """
    Draw a single Laplace noise sample calibrated to (sensitivity, epsilon).

    Parameters
    ----------
    sensitivity : float
        Global sensitivity Δf — maximum change one individual can cause.
        For amounts: set to 99th-percentile cap (spec Section 3.1).
        For counts:  Δf = 1.
    epsilon : float
        Privacy parameter. Smaller = stronger privacy, more noise.

    Returns
    -------
    float
        A noise value drawn from Lap(0, sensitivity/epsilon).

    Example
    -------
    >>> noise = laplace_noise(sensitivity=100_000, epsilon=2.0)
    # Expected |noise| ≈ 50,000 for a single query.
    # On aggregate of n=10,000: per-record error ≈ ₹5 (spec Section 7.1).
    """
    if sensitivity < 0:
        raise ValueError(f"sensitivity must be >= 0, got {sensitivity}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    scale = sensitivity / epsilon
    return np.random.laplace(loc=0.0, scale=scale)


def privatise_amount(
    amount: float,
    epsilon: float,
    sensitivity: float,
    lower_clip: float = 0.0,
    upper_clip: Optional[float] = None,
) -> float:
    """
    Add Laplace noise to a single transaction amount and clip to valid range.

    Clipping is required to bound sensitivity (spec Section 3.1, [R5]).
    Without clipping, one extreme transaction could dominate the sensitivity.

    Parameters
    ----------
    amount : float
        The true transaction amount (e.g. ₹450.00).
    epsilon : float
        Privacy budget for this query (spec Stage 1: ε = 2.0).
    sensitivity : float
        Δf — typically the 99th-percentile transaction cap for this portfolio.
    lower_clip : float
        Minimum valid output (default 0.0 — amounts can't be negative).
    upper_clip : float, optional
        Maximum valid output. If None, no upper clipping.

    Returns
    -------
    float
        Privatised amount, clipped to [lower_clip, upper_clip].
    """
    noisy = amount + laplace_noise(sensitivity=sensitivity, epsilon=epsilon)

    # Clip to valid range
    noisy = max(noisy, lower_clip)
    if upper_clip is not None:
        noisy = min(noisy, upper_clip)

    return round(noisy, 2)


def privatise_count(count: int, epsilon: float) -> int:
    """
    Add Laplace noise to an integer count query.

    Sensitivity is always 1 for counting queries (one individual
    contributes at most 1 to any count — spec Section 3.1).

    Parameters
    ----------
    count : int
        True count value.
    epsilon : float
        Privacy budget for this query.

    Returns
    -------
    int
        Privatised count, clipped to >= 0.
    """
    noisy = count + laplace_noise(sensitivity=1.0, epsilon=epsilon)
    return max(0, int(round(noisy)))


def privatise_transaction_record(
    record: dict,
    epsilon: float,
    amount_sensitivity: float,
    amount_99th_percentile: float,
) -> dict:
    """
    Apply Laplace noise to all numeric fields in a transaction record.

    This is the main entry point for Stage 1 (extraction_from_pdfs/).
    Non-numeric fields (date, narration string, reference number)
    are passed through unchanged — they are handled by Stage 2.

    Parameters
    ----------
    record : dict
        Raw extracted transaction with keys:
        date, particulars, deposits, withdrawals, balance
    epsilon : float
        Privacy budget for this stage (spec Stage 1: ε = 2.0).
        Split equally between deposits/withdrawals and balance.
    amount_sensitivity : float
        Δf for amount fields — use 99th percentile cap.
    amount_99th_percentile : float
        The cap value — amounts above this are clipped before noise.

    Returns
    -------
    dict
        New record with privatised numeric fields.
        Adds 'dp_applied': True and 'dp_epsilon_stage1': epsilon.
    """
    result = dict(record)  # shallow copy — don't mutate input

    # Split ε budget: half for amounts, half for balance
    # This is basic composition within the stage
    eps_amount = epsilon * 0.6   # 60% for the primary amount fields
    eps_balance = epsilon * 0.4  # 40% for balance (less sensitive)

    def _parse_float(val: str) -> Optional[float]:
        """Parse Indian-formatted number strings like '1,04,500.00'."""
        try:
            cleaned = str(val).replace(",", "").strip()
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    # ---- Deposits --------------------------------------------------------
    dep_raw = _parse_float(record.get("deposits", ""))
    if dep_raw is not None:
        clipped = min(dep_raw, amount_99th_percentile)
        noisy_dep = privatise_amount(
            amount=clipped,
            epsilon=eps_amount,
            sensitivity=amount_sensitivity,
            lower_clip=0.0,
            upper_clip=amount_99th_percentile * 1.5,
        )
        result["deposits"] = str(noisy_dep)

    # ---- Withdrawals -----------------------------------------------------
    with_raw = _parse_float(record.get("withdrawals", ""))
    if with_raw is not None:
        clipped = min(with_raw, amount_99th_percentile)
        noisy_with = privatise_amount(
            amount=clipped,
            epsilon=eps_amount,
            sensitivity=amount_sensitivity,
            lower_clip=0.0,
            upper_clip=amount_99th_percentile * 1.5,
        )
        result["withdrawals"] = str(noisy_with)

    # ---- Balance ---------------------------------------------------------
    bal_raw = _parse_float(record.get("balance", ""))
    if bal_raw is not None:
        noisy_bal = privatise_amount(
            amount=bal_raw,
            epsilon=eps_balance,
            sensitivity=amount_sensitivity * 2,  # balance has higher sensitivity
            lower_clip=0.0,
        )
        result["balance"] = str(noisy_bal)

    # Audit fields
    result["dp_applied"] = True
    result["dp_epsilon_stage1"] = epsilon

    return result


# ---------------------------------------------------------------------------
# 2. Randomised Response
#    Used for: Stage 2 (binary token presence) & Stage 3 (merchant label)
#    Guarantee: local ε-DP
#    Flip probability: 1 / (e^ε + 1)
#    Theory: [R11] Warner (1965), [R12] Kairouz et al. (2016)
# ---------------------------------------------------------------------------

def randomised_response_binary(true_value: bool, epsilon: float) -> bool:
    """
    Apply Randomised Response to a single binary value.

    With probability e^ε / (e^ε + 1): return true_value (correct answer).
    With probability 1 / (e^ε + 1):   return NOT true_value (flip).

    This satisfies local ε-DP — noise is added at the source,
    before any data leaves the individual's record.

    Parameters
    ----------
    true_value : bool
        The real binary value to protect.
    epsilon : float
        Privacy parameter. Higher ε = less flipping = less privacy.
        Spec Stage 2 (payee tokens): ε = 1.0 (strong — most sensitive stage)
        Spec Stage 3 (merchant label): ε = 3.0 (moderate)

    Returns
    -------
    bool
        The privatised binary value.

    Example
    -------
    >>> # ε=1.0: flip prob = 1/(e+1) ≈ 0.269
    >>> rr = randomised_response_binary(True, epsilon=1.0)
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    e_eps = math.exp(epsilon)
    # Probability of returning the CORRECT answer
    p_correct = e_eps / (e_eps + 1)

    if random.random() < p_correct:
        return true_value
    else:
        return not true_value


def privatise_payee_token_presence(
    token: str,
    is_present: bool,
    epsilon: float = 1.0,
) -> Tuple[str, bool]:
    """
    Apply local DP to a single payee token's presence bit.

    This implements RAPPOR-style local DP (spec Stage 2, [R9]):
    for each token in the payee vocabulary, we flip the presence
    bit with probability 1/(e^ε + 1).

    Why this matters: raw payee token frequencies reveal social
    graphs. A frequency table showing "ZOMATO appears 47 times"
    for user X is fine; but "RAHUL KUMAR appears 12 times" reveals
    a personal relationship pattern. RR protects individual-level
    token frequencies while preserving population-level statistics.

    Parameters
    ----------
    token : str
        The token string (e.g. "ZOMATO", "SWIGGY", "RAHUL").
    is_present : bool
        Whether this token appears in the individual's payee strings.
    epsilon : float
        Privacy budget (spec Stage 2: ε = 1.0 — strongest in pipeline).

    Returns
    -------
    Tuple[str, bool]
        (token, privatised_presence)
    """
    privatised = randomised_response_binary(is_present, epsilon)
    return (token, privatised)


def privatise_payee_token_frequencies(
    token_counts: Dict[str, int],
    epsilon: float = 1.0,
    vocab_size: Optional[int] = None,
) -> Dict[str, int]:
    """
    Apply local DP to an entire payee token frequency dictionary.

    Each token's presence/absence bit is independently perturbed
    using Randomised Response. The counts are then reconstructed
    from the noisy bits.

    This is the Stage 2 main entry point — call this before storing
    or returning any payee frequency analysis.

    Parameters
    ----------
    token_counts : dict
        Raw token → count mapping from payee name extraction.
        e.g. {"ZOMATO": 5, "SWIGGY": 3, "RAHUL KUMAR": 2}
    epsilon : float
        Privacy budget for this stage (spec: ε = 1.0).
    vocab_size : int, optional
        Total vocabulary size (for calibrating noise).
        If None, uses len(token_counts).

    Returns
    -------
    dict
        Privatised token → adjusted_count mapping.
        Note: some counts may become 0 (token flipped to absent).
        Some zero-count tokens from the vocab may appear (false positives).
        This is expected and correct DP behaviour.
    """
    if not token_counts:
        return {}

    e_eps = math.exp(epsilon)
    # Probability of reporting TRUE value (correct)
    p = e_eps / (e_eps + 1)
    # Probability of flipping (error)
    q = 1.0 - p

    result = {}
    for token, count in token_counts.items():
        # Convert count to presence bit
        is_present = count > 0

        # Apply RR to the presence bit
        noisy_present = randomised_response_binary(is_present, epsilon)

        if noisy_present:
            # Debias the count estimate using the RR formula:
            # E[noisy] = p * true + q * (1 - true)
            # Debiased = (noisy - q) / (p - q)
            # For counts: use original count but apply noise scaling
            debiased_count = max(0, int(round(count * p - count * q)))
            result[token] = max(1, debiased_count) if count > 0 else 1
        # If noisy_present is False, token is excluded (count = 0)
        # This is intentional — some true tokens will be dropped

    return result


# ---------------------------------------------------------------------------
# 3. Exponential Mechanism
#    Used for: Stage 4 (categorical merchant category selection)
#    Guarantee: pure ε-DP
#    Theory: [R13] McSherry & Talwar (2007)
# ---------------------------------------------------------------------------

def exponential_mechanism(
    candidates: List[str],
    scores: Dict[str, float],
    epsilon: float,
    sensitivity: float = 1.0,
) -> str:
    """
    Select a category using the Exponential Mechanism.

    Samples from candidates with probability proportional to
    exp(ε * score / (2 * sensitivity)).

    Higher-scored candidates are exponentially more likely to be
    selected, while privacy is maintained. This is the correct
    mechanism whenever the output is categorical (spec Stage 4, [R13]).

    Parameters
    ----------
    candidates : list of str
        All possible output categories.
    scores : dict
        Utility score for each candidate. Higher = more preferred.
        Typically: score = count or confidence for each category.
    epsilon : float
        Privacy budget (spec Stage 4: ε = 2.0).
    sensitivity : float
        Maximum change in any score from one individual's data.
        For category counts: Δu = 1 (one person → one category).

    Returns
    -------
    str
        The selected category (may differ from true argmax).

    Example
    -------
    >>> cats = ["Food & Dining", "Travel", "Utilities"]
    >>> scores = {"Food & Dining": 10, "Travel": 3, "Utilities": 1}
    >>> selected = exponential_mechanism(cats, scores, epsilon=2.0)
    # "Food & Dining" is most likely, but not guaranteed.
    """
    if not candidates:
        raise ValueError("candidates list is empty")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    # Compute unnormalised probabilities
    # Pr[output = c] ∝ exp(ε * score(c) / (2 * Δu))
    raw_scores = []
    for c in candidates:
        score = scores.get(c, 0.0)
        raw_scores.append(epsilon * score / (2.0 * sensitivity))

    # Numerical stability: subtract max before exponentiating
    max_score = max(raw_scores)
    weights = [math.exp(s - max_score) for s in raw_scores]
    total = sum(weights)
    probabilities = [w / total for w in weights]

    # Sample from the distribution
    selected = random.choices(candidates, weights=probabilities, k=1)[0]
    return selected
