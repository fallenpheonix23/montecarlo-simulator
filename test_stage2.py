"""Stage 2 validation: live data fetch + options pricer integration.

Runs five tests against live yfinance data, prints PASS/FAIL for each, and a
summary. Requires network access. Uses numpy, pandas, matplotlib, yfinance.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np

from data_fetch import DataFetcher
from options import OptionsPricer

SEED = 42                      # reproducible Monte Carlo draws
results = {}                   # test name -> bool


# ====================================================================== #
# Test 1 -- Data fetch sanity
# ====================================================================== #
def test_data_fetch():
    print("\n--- Test 1: Data fetch (SPY) ---")
    fetcher = DataFetcher()

    price = fetcher.get_stock_price("SPY")
    vol = fetcher.get_historical_volatility("SPY")
    rate = fetcher.get_risk_free_rate()

    print(f"Price : {price:.2f}")
    print(f"Vol   : {vol:.4f}")
    print(f"Rate  : {rate:.4f}")

    # All positive floats, with vol and rate in sensible ranges.
    cond_positive = price > 0 and vol > 0 and rate > 0
    cond_vol = 0.05 <= vol <= 1.0
    cond_rate = 0.0 <= rate <= 0.15
    passed = cond_positive and cond_vol and cond_rate

    print(f"All positive            : {cond_positive}")
    print(f"0.05 <= vol <= 1.0      : {cond_vol}")
    print(f"0.0 <= rate <= 0.15     : {cond_rate}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 1 (data fetch)"] = passed


# ====================================================================== #
# Test 2 -- Correlation matrix
# ====================================================================== #
def test_correlation_matrix():
    print("\n--- Test 2: Correlation matrix (SPY, TLT, GLD) ---")
    fetcher = DataFetcher()

    corr = fetcher.get_correlation_matrix(["SPY", "TLT", "GLD"])
    print(corr)

    cond_shape = corr.shape == (3, 3)
    # Diagonal must be 1 (allow tiny float error).
    cond_diag = np.allclose(np.diag(corr), 1.0)
    cond_range = np.all(corr >= -1.0) and np.all(corr <= 1.0)
    passed = cond_shape and cond_diag and cond_range

    print(f"Shape is 3x3            : {cond_shape}")
    print(f"Diagonal all 1.0        : {cond_diag}")
    print(f"All in [-1, 1]          : {cond_range}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 2 (correlation)"] = passed


# ====================================================================== #
# Test 3 -- Options pricer integration (European call, live data)
# ====================================================================== #
def test_pricer_integration():
    print("\n--- Test 3: Options pricer integration (SPY European call) ---")
    pricer = OptionsPricer(seed=SEED)

    spot = pricer.fetcher.get_stock_price("SPY")
    strike = spot * 1.05                       # 5% out-of-the-money

    res = pricer.price(
        ticker="SPY", strike=strike, expiry_years=0.5,
        option_type="call", style="european",
    )

    price = res["price"]
    se = res["std_error"]
    se_ratio = se / price if price > 0 else np.inf

    print(f"Strike            : {strike:.2f}")
    print(f"Option price      : {price:.4f}")
    print(f"Standard error    : {se:.4f}  ({se_ratio * 100:.2f}% of price)")

    cond_positive = isinstance(price, float) and price > 0
    cond_se = se_ratio < 0.05                   # SE under 5% of the price
    passed = cond_positive and cond_se

    print(f"Price positive float    : {cond_positive}")
    print(f"SE < 5% of price        : {cond_se}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 3 (pricer integration)"] = passed


# ====================================================================== #
# Test 4 -- Asian <= European (live data)
# ====================================================================== #
def test_asian_vs_european_live():
    print("\n--- Test 4: Asian <= European (SPY, live) ---")
    pricer = OptionsPricer(seed=SEED)

    spot = pricer.fetcher.get_stock_price("SPY")
    strike = spot * 1.05
    params = dict(ticker="SPY", strike=strike, expiry_years=0.5,
                  option_type="call")

    # Common random numbers: re-seed the engine before each leg so both options
    # are priced on the SAME simulated paths, making the comparison exact.
    pricer.engine.rng = np.random.default_rng(SEED)
    euro = pricer.price(style="european", **params)["price"]
    pricer.engine.rng = np.random.default_rng(SEED)
    asian = pricer.price(style="asian", **params)["price"]

    passed = asian <= euro

    print(f"European call     : {euro:.4f}")
    print(f"Asian call        : {asian:.4f}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 4 (Asian <= European)"] = passed


# ====================================================================== #
# Test 5 -- Barrier knock-out below European (live data)
# ====================================================================== #
def test_barrier_live():
    print("\n--- Test 5: Barrier knock-out <= European (SPY, live) ---")
    pricer = OptionsPricer(seed=SEED)

    spot = pricer.fetcher.get_stock_price("SPY")
    strike = spot * 1.05
    barrier = spot * 0.90                       # 10% below current price
    params = dict(ticker="SPY", strike=strike, expiry_years=0.5,
                  option_type="call")

    # Common random numbers: price both legs on identical paths so the
    # knock-out's effect (zeroing some paths) is isolated from MC noise.
    pricer.engine.rng = np.random.default_rng(SEED)
    euro = pricer.price(style="european", **params)["price"]
    pricer.engine.rng = np.random.default_rng(SEED)
    knockout = pricer.price(style="barrier", barrier=barrier, **params)["price"]

    passed = knockout <= euro

    print(f"Barrier level     : {barrier:.2f}")
    print(f"European call     : {euro:.4f}")
    print(f"Barrier knockout  : {knockout:.4f}")
    print("RESULT:", "PASS" if passed else "FAIL")
    results["Test 5 (barrier)"] = passed


# ====================================================================== #
# Run all tests + summary
# ====================================================================== #
def main():
    test_data_fetch()
    test_correlation_matrix()
    test_pricer_integration()
    test_asian_vs_european_live()
    test_barrier_live()

    n_passed = sum(results.values())
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("-" * 50)
    print(f"  {n_passed} / {len(results)} tests passed")
    print("=" * 50)


if __name__ == "__main__":
    main()
