"""
dp/tests.py
-----------
Test suite for the Credit Mitra DP module.

Run with:  python dp/tests.py
All tests should pass with no errors before committing.
"""

import sys
import os
import math
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp.budget_tracker import PrivacyBudgetTracker, BudgetExhaustedError
from dp.mechanisms import (
    laplace_noise,
    privatise_amount,
    privatise_count,
    privatise_transaction_record,
    randomised_response_binary,
    privatise_payee_token_frequencies,
    exponential_mechanism,
)
from dp.stage1_extraction import extract_with_dp
from dp.stage2_payee import tokenise_payee, privatise_payee, analyse_payee_population


PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"


def run(name, fn):
    try:
        fn()
        print(f"{PASS}: {name}")
    except Exception as e:
        print(f"{FAIL}: {name}")
        print(f"       {type(e).__name__}: {e}")


# ── Budget Tracker Tests ───────────────────────────────────────────────────

def test_budget_basic():
    t = PrivacyBudgetTracker(epsilon_limit=5.0)
    t.consume("a", epsilon=2.0)
    t.consume("b", epsilon=2.0)
    assert abs(t.consumed() - 4.0) < 1e-9
    assert abs(t.remaining() - 1.0) < 1e-9


def test_budget_exhausted():
    t = PrivacyBudgetTracker(epsilon_limit=3.0)
    t.consume("x", epsilon=2.0)
    try:
        t.consume("y", epsilon=2.0)
        assert False, "Should have raised BudgetExhaustedError"
    except BudgetExhaustedError:
        pass


def test_budget_dry_run():
    t = PrivacyBudgetTracker(epsilon_limit=5.0)
    t.consume("a", epsilon=2.0, dry_run=True)
    assert t.consumed() == 0.0, "Dry run should not consume budget"


def test_budget_report_structure():
    t = PrivacyBudgetTracker(epsilon_limit=8.0)
    t.consume("stage_1", epsilon=2.0, note="test")
    r = t.report()
    assert "epsilon_limit" in r
    assert "epsilon_consumed" in r
    assert "entries" in r
    assert len(r["entries"]) == 1
    assert r["entries"][0]["stage"] == "stage_1"


# ── Laplace Mechanism Tests ────────────────────────────────────────────────

def test_laplace_noise_mean_zero():
    """Laplace noise should have mean ≈ 0 over many samples."""
    samples = [laplace_noise(sensitivity=1000.0, epsilon=2.0) for _ in range(10_000)]
    mean = statistics.mean(samples)
    assert abs(mean) < 50.0, f"Mean too far from zero: {mean:.2f}"


def test_laplace_scale():
    """Noise scale should be sensitivity/epsilon."""
    sensitivity, epsilon = 1000.0, 2.0
    expected_mean_abs = sensitivity / epsilon  # = 500.0
    samples = [abs(laplace_noise(sensitivity, epsilon)) for _ in range(10_000)]
    empirical_mean = statistics.mean(samples)
    # Should be within 10% of expected
    assert abs(empirical_mean - expected_mean_abs) < expected_mean_abs * 0.1


def test_privatise_amount_clip():
    """Privatised amount should stay within clipped range."""
    for _ in range(100):
        result = privatise_amount(
            amount=500.0,
            epsilon=2.0,
            sensitivity=10_000.0,
            lower_clip=0.0,
            upper_clip=200_000.0,
        )
        assert 0.0 <= result <= 200_000.0, f"Out of range: {result}"


def test_privatise_count_nonnegative():
    """Privatised counts should always be >= 0."""
    for count in [0, 1, 5, 100]:
        for _ in range(50):
            result = privatise_count(count, epsilon=1.0)
            assert result >= 0, f"Negative count: {result}"


def test_privatise_transaction_record():
    """Stage 1 record privatisation should add dp_applied flag."""
    record = {
        "date": "15-01-2026",
        "particulars": "UPI/DR/601591718046/SWIGGY LI/UTIB/...",
        "deposits": "",
        "withdrawals": "172.00",
        "balance": "1,041.95",
    }
    result = privatise_transaction_record(
        record=record,
        epsilon=2.0,
        amount_sensitivity=100_000.0,
        amount_99th_percentile=100_000.0,
    )
    assert result["dp_applied"] is True
    assert result["dp_epsilon_stage1"] == 2.0
    assert result["particulars"] == record["particulars"]  # string unchanged
    assert result["date"] == record["date"]  # date unchanged
    # Withdrawal should be privatised (different from 172.00 usually)
    # (very rarely might be exactly same due to noise rounding)


# ── Randomised Response Tests ──────────────────────────────────────────────

def test_rr_flip_rate():
    """
    At ε=1.0, flip prob = 1/(e+1) ≈ 0.269.
    Over 10K samples, empirical rate should be within 2% of theoretical.
    """
    epsilon = 1.0
    expected_flip = 1.0 / (math.exp(epsilon) + 1)
    n = 10_000
    flips = sum(
        1 for _ in range(n)
        if randomised_response_binary(True, epsilon) is False
    )
    empirical = flips / n
    assert abs(empirical - expected_flip) < 0.02, (
        f"Flip rate {empirical:.3f} too far from expected {expected_flip:.3f}"
    )


def test_rr_high_epsilon_rarely_flips():
    """At ε=10 (very weak privacy), almost never flips."""
    flips = sum(
        1 for _ in range(1000)
        if randomised_response_binary(True, epsilon=10.0) is False
    )
    assert flips < 50, f"Too many flips at high ε: {flips}"


def test_rr_low_epsilon_flips_often():
    """At ε=0.1 (very strong privacy), flips ≈ 50% of the time."""
    flips = sum(
        1 for _ in range(1000)
        if randomised_response_binary(True, epsilon=0.1) is False
    )
    # Should be between 35% and 65%
    assert 350 < flips < 650, f"Unexpected flip count at low ε: {flips}"


# ── Tokenisation Tests ─────────────────────────────────────────────────────

def test_tokenise_merchant():
    tokens = tokenise_payee("ZOMATO LI")
    assert "ZOMATO" in tokens
    # "LI" might be filtered as noise; that's ok


def test_tokenise_removes_upi_noise():
    tokens = tokenise_payee("UPI/DR/601591718046/SWIGGY LI/UTIB")
    assert "UPI" not in tokens
    assert "DR" not in tokens
    assert "SWIGGY" in tokens


def test_tokenise_person_name():
    tokens = tokenise_payee("RAHUL KUMAR")
    assert "RAHUL" in tokens
    assert "KUMAR" in tokens


def test_tokenise_empty():
    assert tokenise_payee("") == []
    assert tokenise_payee("   ") == []


def test_tokenise_removes_phone_numbers():
    tokens = tokenise_payee("8971829355")
    assert tokens == []


# ── Stage 2 Payee Tests ────────────────────────────────────────────────────

def test_privatise_payee_structure():
    result = privatise_payee("ZOMATO LI", epsilon=1.0)
    assert "payee_tokens_dp" in result
    assert "payee_token_count_dp" in result
    assert result["dp_applied"] is True
    assert result["dp_epsilon_stage2"] == 1.0
    assert "payee_raw_DEBUG_ONLY" not in result  # raw redacted by default


def test_privatise_payee_no_raw_in_output():
    """Raw payee name must not appear in output when redact_raw=True."""
    result = privatise_payee("RAHUL KUMAR PRIVATE", epsilon=1.0, redact_raw=True)
    result_str = str(result)
    # The concatenated raw string should not appear
    assert "RAHUL KUMAR PRIVATE" not in result_str


def test_analyse_payee_population():
    payees = ["ZOMATO LI"] * 20 + ["SWIGGY LTD"] * 10 + ["FLIPKART"] * 8
    freq, report = analyse_payee_population(payees, epsilon=1.0, min_count_threshold=3)
    assert isinstance(freq, dict)
    assert report["dp_applied"] is True
    assert report["payees_processed"] == 38


# ── Stage 1 Integration Test ───────────────────────────────────────────────

def test_extract_with_dp_integration():
    """Full Stage 1 integration test with real-format records."""
    records = [
        {
            "date": "15-01-2026",
            "particulars": "UPI/DR/601591718046/SWIGGY LI/UTIB/**.GPAY@OKPAYAXIS/UPI",
            "deposits": "",
            "withdrawals": "172.00",
            "balance": "1,041.95",
        },
        {
            "date": "17-01-2026",
            "particulars": "UPI/CR/601732821168/GAJNANI P/KKBK/**60222@PTHDFC/NA",
            "deposits": "2,880.00",
            "withdrawals": "",
            "balance": "3,831.95",
        },
        {
            "date": "",
            "particulars": "Chq: 601865247681",
            "deposits": "",
            "withdrawals": "",
            "balance": "",
        },
    ]

    # Use fresh tracker to avoid budget exhaustion from other tests
    from dp.budget_tracker import PrivacyBudgetTracker
    import dp.stage1_extraction as s1
    original_budget = s1.budget
    s1.budget = PrivacyBudgetTracker(epsilon_limit=10.0)

    try:
        private_records, report = extract_with_dp(records, epsilon=2.0)
        assert len(private_records) == 3
        assert report["dp_applied"] is True
        assert report["records_privatised"] == 2   # 2 have amounts
        assert report["records_passed_through"] == 1  # the empty one
        assert "sensitivity_cap_inr" in report
        # Narration strings must be untouched
        assert private_records[0]["particulars"] == records[0]["particulars"]
    finally:
        s1.budget = original_budget


def test_exponential_mechanism():
    """Exponential mechanism should heavily favour highest-scoring category."""
    from collections import Counter
    candidates = ["Food & Dining", "Travel", "Utilities", "Healthcare"]
    scores = {"Food & Dining": 100, "Travel": 1, "Utilities": 1, "Healthcare": 1}

    wins = Counter()
    for _ in range(1000):
        selected = exponential_mechanism(candidates, scores, epsilon=2.0)
        wins[selected] += 1

    # Food & Dining should win >> 90% with such a score gap
    assert wins["Food & Dining"] > 800, f"Expected dominant winner: {dict(wins)}"


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Credit Mitra — Differential Privacy Module Tests")
    print("=" * 60 + "\n")

    print("── Budget Tracker ──────────────────────────────────────")
    run("Basic budget consumption", test_budget_basic)
    run("Budget exhaustion raises error", test_budget_exhausted)
    run("Dry run does not consume budget", test_budget_dry_run)
    run("Report has correct structure", test_budget_report_structure)

    print("\n── Laplace Mechanism ───────────────────────────────────")
    run("Noise mean ≈ 0 over 10K samples", test_laplace_noise_mean_zero)
    run("Noise scale = sensitivity/epsilon", test_laplace_scale)
    run("Privatised amount stays in clipped range", test_privatise_amount_clip)
    run("Privatised count always >= 0", test_privatise_count_nonnegative)
    run("Stage 1 record privatisation", test_privatise_transaction_record)

    print("\n── Randomised Response ─────────────────────────────────")
    run("Flip rate matches theoretical (ε=1.0)", test_rr_flip_rate)
    run("High ε rarely flips", test_rr_high_epsilon_rarely_flips)
    run("Low ε flips ≈ 50%", test_rr_low_epsilon_flips_often)

    print("\n── Tokenisation ────────────────────────────────────────")
    run("Merchant token extracted", test_tokenise_merchant)
    run("UPI noise tokens removed", test_tokenise_removes_upi_noise)
    run("Person names tokenised", test_tokenise_person_name)
    run("Empty string → empty list", test_tokenise_empty)
    run("Phone numbers filtered", test_tokenise_removes_phone_numbers)

    print("\n── Stage 2 Payee ───────────────────────────────────────")
    run("Privatise payee has correct structure", test_privatise_payee_structure)
    run("Raw payee not in output (redacted)", test_privatise_payee_no_raw_in_output)
    run("Population frequency analysis", test_analyse_payee_population)

    print("\n── Stage 1 Integration ─────────────────────────────────")
    run("extract_with_dp full integration", test_extract_with_dp_integration)
    run("Exponential mechanism favours top score", test_exponential_mechanism)

    print("\n" + "=" * 60)
    print("  Tests complete.")
    print("=" * 60 + "\n")
