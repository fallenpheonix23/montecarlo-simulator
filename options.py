"""High-level option pricer that wires live data into the Monte Carlo engine.

OptionsPricer pulls the current price, historical volatility, and risk-free
rate for a ticker, simulates risk-neutral GBM paths, applies the requested
payoff, and returns a tidy result dictionary.
"""

from engine import MonteCarloEngine
from data_fetch import DataFetcher
from payoffs import (
    european_call,
    european_put,
    asian_call,
    asian_put,
    barrier_knockout_call,
)

# Daily time step (252 trading days per year).
DT = 1.0 / 252.0


class OptionsPricer:
    """Prices options on real tickers using auto-fetched market data."""

    def __init__(self, seed=None):
        # Optional seed makes the Monte Carlo draw reproducible for testing.
        self.engine = MonteCarloEngine(seed=seed)
        self.fetcher = DataFetcher()

    def _select_payoff(self, style, option_type, strike, barrier):
        """Return a payoff(paths) closure for the requested style/type."""
        if style == "european":
            fn = european_call if option_type == "call" else european_put
            return lambda paths: fn(paths, strike)

        elif style == "asian":
            fn = asian_call if option_type == "call" else asian_put
            return lambda paths: fn(paths, strike)

        elif style == "barrier":
            # Only the down-and-out call is defined in Stage 1's payoffs.
            if option_type != "call":
                raise ValueError("barrier style currently supports calls only")
            if barrier is None:
                raise ValueError("barrier style requires a `barrier` level")
            return lambda paths: barrier_knockout_call(paths, strike, barrier)

        else:
            raise ValueError(f"unknown style: {style!r}")

    def price(self, ticker, strike, expiry_years, option_type, style,
              barrier=None, n_paths=50_000):
        """Price an option on `ticker` using live market data.

        Auto-fetches spot, volatility, and the risk-free rate; simulates GBM
        paths under the risk-neutral measure; applies the chosen payoff; and
        returns a dictionary with the price, standard error, the inputs used,
        and the simulated paths (for plotting).
        """
        # --- 1. Fetch live inputs ------------------------------------- #
        S0 = self.fetcher.get_stock_price(ticker)
        sigma = self.fetcher.get_historical_volatility(ticker)
        r = self.fetcher.get_risk_free_rate()
        print(f"Fetched: S0={S0:.2f}, sigma={sigma:.2f}, r={r:.3f}")

        # --- 2. Simulate risk-neutral price paths --------------------- #
        # Drift = r (risk-neutral). Antithetic variates tighten the estimate.
        paths = self.engine.simulate_gbm(
            S0, r, sigma, expiry_years, DT, n_paths, antithetic=True
        )

        # --- 3. Pick the payoff and price ----------------------------- #
        payoff_func = self._select_payoff(style, option_type, strike, barrier)
        option_price, std_error = self.engine.price_option(
            paths, payoff_func, r, expiry_years
        )

        # --- 4. Package the result ------------------------------------ #
        return {
            "price": option_price,
            "std_error": std_error,
            "stock_price": S0,
            "volatility": sigma,
            "risk_free_rate": r,
            "n_paths": n_paths,
            "paths": paths,
        }
