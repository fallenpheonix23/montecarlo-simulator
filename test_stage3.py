"""Stage 3 validation: portfolio risk + retirement planning.

Runs five tests, prints PASS/FAIL and a summary, and confirms both output
figures were written. Requires network access (live portfolio data for risk).
"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")          # set headless backend before pyplot is imported

from risk import RiskEstimator
from retirement import RetirementPlanner

SEED = 42
results = {}

# 60/40 portfolio used across the risk tests.
TICKERS = ["SPY", "TLT"]
WEIGHTS = [0.6, 0.4]


# ====================================================================== #
# Test 1 -- VaR / CVaR ordering
# ====================================================================== #
def test_var_ordering():
    print("\n--- Test 1: VaR/CVaR ordering (60/40 SPY/TLT, 1-day) ---")
    est = RiskEstimator(seed=SEED)
    res = est.estimate_var_cvar(TICKERS, WEIGHTS, horizon_days=1)

    var, cvar = res["var"], res["cvar"]
    cond_var = abs(var[0.99]) > abs(var[0.95])           # 99% VaR > 95% VaR
    cond_cvar_99 = abs(cvar[0.99]) > abs(var[0.99])      # CVaR > VaR at 99%
    cond_cvar_95 = abs(cvar[0.95]) > abs(var[0.95])      # CVaR > VaR at 95%
    passed = cond_var and cond_cvar_99 and cond_cvar_95

    print(f"95% VaR : {var[0.95]*100:.3f}%   95% CVaR : {cvar[0.95]*100:.3f}%")
    print(f"99% VaR : {var[0.99]*100:.3f}%   99% CVaR : {cvar[0.99]*100:.3f}%")
    print(f"|VaR99| > |VaR95|   : {cond_var}")
    print(f"|CVaR99| > |VaR99|  : {cond_cvar_99}")
    print(f"|CVaR95| > |VaR95|  : {cond_cvar_95}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 1 (VaR ordering)"] = passed
    return res


# ====================================================================== #
# Test 2 -- Horizon scaling
# ====================================================================== #
def test_horizon_scaling():
    print("\n--- Test 2: Horizon scaling (1-day vs 10-day) ---")
    est = RiskEstimator(seed=SEED)
    res_1 = est.estimate_var_cvar(TICKERS, WEIGHTS, horizon_days=1)
    res_10 = est.estimate_var_cvar(TICKERS, WEIGHTS, horizon_days=10)

    var_1 = res_1["var"][0.95]
    var_10 = res_10["var"][0.95]
    passed = abs(var_10) > abs(var_1)                    # longer horizon = bigger VaR

    print(f"1-day  95% VaR : {var_1*100:.3f}%")
    print(f"10-day 95% VaR : {var_10*100:.3f}%")
    print(f"10-day VaR > 1-day VaR : {passed}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 2 (horizon scaling)"] = passed


# ====================================================================== #
# Test 3 -- Stress test sanity
# ====================================================================== #
def test_stress():
    print("\n--- Test 3: Stress test (SPY -30%, TLT +10%) ---")
    est = RiskEstimator(seed=SEED)
    shock = {"SPY": -0.30, "TLT": 0.10}
    loss = est.stress_test(TICKERS, WEIGHTS, shock)

    # Expected: 0.6*(-0.30) + 0.4*(0.10) = -0.14.
    cond_negative = loss < 0
    cond_range = -0.40 <= loss <= -0.05
    passed = cond_negative and cond_range

    print(f"Portfolio loss     : {loss*100:.2f}%")
    print(f"Negative           : {cond_negative}")
    print(f"Between -40% and -5%: {cond_range}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 3 (stress test)"] = passed


# ====================================================================== #
# Test 4 -- Retirement simulation sanity
# ====================================================================== #
def test_retirement_sanity():
    print("\n--- Test 4: Retirement sanity (1M start, 5k/mo, 30y) ---")
    planner = RetirementPlanner(seed=SEED)
    res = planner.simulate(current_savings=1_000_000,
                           monthly_contribution=5_000,
                           years_to_retirement=30)

    cond_median = res["median"] > 1_000_000
    cond_ruin = 0.0 <= res["prob_ruin"] <= 0.5
    cond_spread = res["p90"] > res["p10"]
    passed = cond_median and cond_ruin and cond_spread

    print(f"Median final wealth : ${res['median']:,.0f}")
    print(f"10th pct            : ${res['p10']:,.0f}")
    print(f"90th pct            : ${res['p90']:,.0f}")
    print(f"P(ruin)             : {res['prob_ruin']*100:.2f}%")
    print(f"Median > start      : {cond_median}")
    print(f"0 <= P(ruin) <= 0.5 : {cond_ruin}")
    print(f"p90 > p10           : {cond_spread}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 4 (retirement sanity)"] = passed
    return res


# ====================================================================== #
# Test 5 -- Contribution impact
# ====================================================================== #
def test_contribution_impact():
    print("\n--- Test 5: Contribution impact (0 vs 10k/mo) ---")
    planner = RetirementPlanner(seed=SEED)
    common = dict(current_savings=1_000_000, years_to_retirement=30)

    low = planner.simulate(monthly_contribution=0, **common)
    high = planner.simulate(monthly_contribution=10_000, **common)

    passed = high["median"] > low["median"]

    print(f"Median (0/mo)     : ${low['median']:,.0f}")
    print(f"Median (10k/mo)   : ${high['median']:,.0f}")
    print(f"Higher contribution -> higher median : {passed}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 5 (contribution impact)"] = passed


# ====================================================================== #
# Run everything, save figures, summarise
# ====================================================================== #
def main():
    risk_res = test_var_ordering()
    test_horizon_scaling()
    test_stress()
    retire_res = test_retirement_sanity()
    test_contribution_impact()

    # Produce the two required figures.
    RiskEstimator(seed=SEED).plot_risk(risk_res)
    RetirementPlanner(seed=SEED).plot_retirement(retire_res)

    n_passed = sum(results.values())
    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("-" * 52)
    print(f"  {n_passed} / {len(results)} tests passed")

    # Confirm both figures were saved.
    print("-" * 52)
    for fname in ("risk_results.png", "retirement_results.png"):
        status = "saved" if os.path.exists(fname) else "MISSING"
        print(f"  {fname}: {status}")
    print("=" * 52)


if __name__ == "__main__":
    main()
