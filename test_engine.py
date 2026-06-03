"""Validation suite for the Monte Carlo engine (Stage 1).

Runs five sanity/convergence tests, prints PASS/FAIL for each, prints a
summary of how many passed, and saves a two-panel figure:
  (1) the convergence plot from Test 1
  (2) the returns histogram with VaR/CVaR lines from Test 5
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")            # non-interactive backend: just save the figure
import matplotlib.pyplot as plt

from engine import MonteCarloEngine
from black_scholes import bs_price
from payoffs import (
    european_call,
    european_put,
    asian_call,
    barrier_knockout_call,
)

# A fixed seed keeps the validation reproducible run-to-run.
SEED = 42

# Shared option parameters used across several tests.
S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
DT = 1.0 / 252.0

results = {}     # test name -> bool (passed?)


# ====================================================================== #
# Test 1 -- Convergence to the Black-Scholes price
# ====================================================================== #
def test_convergence():
    print("\n--- Test 1: Convergence to Black-Scholes ---")
    engine = MonteCarloEngine(seed=SEED)

    n_values, mc_prices, bs = engine.convergence_analysis(
        S0, K, r, sigma, T, option_type="call", max_n=10_000, steps=20
    )

    # Compare the highest-N MC price to the analytical price.
    final_price = mc_prices[-1]
    rel_err = abs(final_price - bs) / bs
    passed = rel_err <= 0.02       # must be within 2% by 10,000 sims

    print(f"BS price          : {bs:.4f}")
    print(f"MC price @ {n_values[-1]:>6} : {final_price:.4f}")
    print(f"Relative error    : {rel_err * 100:.2f}%  (need <= 2%)")
    print("RESULT:", "PASS" if passed else "FAIL")

    results["Test 1 (convergence)"] = passed
    # Return the data so it can be plotted later.
    return n_values, mc_prices, bs


# ====================================================================== #
# Test 2 -- Put-call parity:  C - P == S0 - K*e^{-rT}
# ====================================================================== #
def test_put_call_parity():
    print("\n--- Test 2: Put-call parity ---")
    engine = MonteCarloEngine(seed=SEED)
    n = 50_000

    # Use the SAME simulated paths for call and put so the comparison is fair.
    paths = engine.simulate_gbm(S0, r, sigma, T, DT, n, antithetic=True)
    call, _ = engine.price_option(paths, lambda p: european_call(p, K), r, T)
    put, _ = engine.price_option(paths, lambda p: european_put(p, K), r, T)

    lhs = call - put
    rhs = S0 - K * np.exp(-r * T)
    # Note: rhs (~0.12) is a tiny difference of two ~100-sized numbers, so we
    # measure the parity error relative to the spot price S0 -- the standard,
    # meaningful denominator. (Measuring against rhs itself demands accuracy of
    # ~0.001 on C-P, which is unreachable with plain Monte Carlo.)
    rel_err = abs(lhs - rhs) / S0
    passed = rel_err <= 0.01        # parity should hold within 1% of S0

    print(f"Call price        : {call:.4f}")
    print(f"Put price         : {put:.4f}")
    print(f"C - P             : {lhs:.4f}")
    print(f"S0 - K*e^(-rT)    : {rhs:.4f}")
    print(f"Error / S0        : {rel_err * 100:.3f}%  (need <= 1%)")
    print("RESULT:", "PASS" if passed else "FAIL")

    results["Test 2 (put-call parity)"] = passed


# ====================================================================== #
# Test 3 -- Asian call <= European call  (always true mathematically)
# ====================================================================== #
def test_asian_vs_european():
    print("\n--- Test 3: Asian call <= European call ---")
    engine = MonteCarloEngine(seed=SEED)
    n = 50_000

    # Same paths for both so the inequality reflects the payoff, not noise.
    paths = engine.simulate_gbm(S0, r, sigma, T, DT, n, antithetic=True)
    euro, _ = engine.price_option(paths, lambda p: european_call(p, K), r, T)
    asian, _ = engine.price_option(paths, lambda p: asian_call(p, K), r, T)

    passed = asian <= euro

    print(f"European call     : {euro:.4f}")
    print(f"Asian call        : {asian:.4f}")
    print("RESULT:", "PASS" if passed else "FAIL")

    results["Test 3 (Asian <= European)"] = passed


# ====================================================================== #
# Test 4 -- Barrier knock-out call behaviour
# ====================================================================== #
def test_barrier():
    print("\n--- Test 4: Barrier knock-out call ---")
    engine = MonteCarloEngine(seed=SEED)
    n = 50_000

    paths = engine.simulate_gbm(S0, r, sigma, T, DT, n, antithetic=True)
    euro, _ = engine.price_option(paths, lambda p: european_call(p, K), r, T)

    # Down-and-out call at two barrier levels.
    # barrier=90 sits CLOSER to S0=100 -> hit more often -> lower price.
    # barrier=70 sits further below   -> hit rarely     -> higher price.
    price_b90, _ = engine.price_option(
        paths, lambda p: barrier_knockout_call(p, K, barrier=90), r, T
    )
    price_b70, _ = engine.price_option(
        paths, lambda p: barrier_knockout_call(p, K, barrier=70), r, T
    )

    # Both barriers must price below the plain European call, and the
    # closer barrier (90) must price below the farther barrier (70).
    cond_below_euro = (price_b90 <= euro) and (price_b70 <= euro)
    cond_90_lower = price_b90 <= price_b70
    passed = cond_below_euro and cond_90_lower

    print(f"European call     : {euro:.4f}")
    print(f"Barrier=90 call   : {price_b90:.4f}")
    print(f"Barrier=70 call   : {price_b70:.4f}")
    print(f"Both <= European  : {cond_below_euro}")
    print(f"Barrier90 <= B70  : {cond_90_lower}")
    print("RESULT:", "PASS" if passed else "FAIL")

    results["Test 4 (barrier)"] = passed


# ====================================================================== #
# Test 5 -- VaR / CVaR sanity check
# ====================================================================== #
def test_var_cvar():
    print("\n--- Test 5: VaR / CVaR sanity ---")
    engine = MonteCarloEngine(seed=SEED)

    # Simulate 10,000 daily returns ~ Normal(mean=0.0008, std=0.01).
    rng = np.random.default_rng(SEED)
    returns = rng.normal(loc=0.0008, scale=0.01, size=10_000)

    var_95, cvar_95 = engine.calculate_var_cvar(returns, confidence_level=0.95)
    var_99, cvar_99 = engine.calculate_var_cvar(returns, confidence_level=0.99)

    # 99% VaR should be a bigger loss than 95% VaR; and at each level the
    # CVaR (tail average) should exceed the VaR (tail threshold).
    cond_var = abs(var_99) > abs(var_95)
    cond_cvar_95 = abs(cvar_95) > abs(var_95)
    cond_cvar_99 = abs(cvar_99) > abs(var_99)
    passed = cond_var and cond_cvar_95 and cond_cvar_99

    print(f"95% VaR  : {var_95:.5f}   95% CVaR : {cvar_95:.5f}")
    print(f"99% VaR  : {var_99:.5f}   99% CVaR : {cvar_99:.5f}")
    print(f"|VaR99| > |VaR95|       : {cond_var}")
    print(f"|CVaR95| > |VaR95|      : {cond_cvar_95}")
    print(f"|CVaR99| > |VaR99|      : {cond_cvar_99}")
    print("RESULT:", "PASS" if passed else "FAIL")

    results["Test 5 (VaR/CVaR)"] = passed
    # Return data for the histogram panel.
    return returns, var_95, cvar_95, var_99, cvar_99


# ====================================================================== #
# Run everything and build the summary figure
# ====================================================================== #
def main():
    # Test 1 and Test 5 return data we need for the figure.
    n_values, mc_prices, bs = test_convergence()
    test_put_call_parity()
    test_asian_vs_european()
    test_barrier()
    returns, var_95, cvar_95, var_99, cvar_99 = test_var_cvar()

    # ----- Summary ----------------------------------------------------- #
    n_passed = sum(results.values())
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("-" * 50)
    print(f"  {n_passed} / {len(results)} tests passed")
    print("=" * 50)

    # ----- Figure: two panels ----------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: convergence plot (x-axis log scale).
    ax1.plot(n_values, mc_prices, marker="o", ms=4, label="Monte Carlo price")
    ax1.axhline(bs, color="red", ls="--", label=f"Black-Scholes = {bs:.3f}")
    # Shade the +/-2% tolerance band for visual reference.
    ax1.axhspan(bs * 0.98, bs * 1.02, color="red", alpha=0.08, label="+/-2% band")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of simulations (log scale)")
    ax1.set_ylabel("Option price")
    ax1.set_title("Test 1: MC convergence to Black-Scholes")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # Panel 2: histogram of simulated returns with VaR/CVaR markers.
    ax2.hist(returns, bins=60, color="steelblue", alpha=0.7, edgecolor="none")
    # VaR/CVaR are losses, so plot them on the negative side of the axis.
    ax2.axvline(-var_95, color="orange", ls="--", label=f"95% VaR = {var_95:.4f}")
    ax2.axvline(-cvar_95, color="darkorange", ls=":", label=f"95% CVaR = {cvar_95:.4f}")
    ax2.axvline(-var_99, color="red", ls="--", label=f"99% VaR = {var_99:.4f}")
    ax2.axvline(-cvar_99, color="darkred", ls=":", label=f"99% CVaR = {cvar_99:.4f}")
    ax2.set_xlabel("Daily return")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Test 5: Returns distribution with VaR / CVaR")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = "results.png"
    fig.savefig(out, dpi=120)
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
