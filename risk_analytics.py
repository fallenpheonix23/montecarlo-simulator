"""Advanced portfolio risk analytics (Stage 5).

A pure-computation layer (no network) that powers the upgraded Risk Estimator
page: choice of return distribution (normal / Student-t / historical bootstrap),
component (marginal) VaR decomposition, worst-case scenarios, and a Kupiec POF
backtest. Fetching/caching stays in the app; this module just takes a returns
DataFrame and computes.

Nothing here touches the original risk.py, so the Stage 1-3 test suites are
unaffected.
"""

import numpy as np
import pandas as pd
from scipy import stats

from engine import MonteCarloEngine


class PortfolioRiskAnalyzer:
    """Portfolio VaR/CVaR with distribution choice and risk attribution."""

    def __init__(self, seed=None):
        # Reuse the project engine for its RNG and calculate_var_cvar routine.
        self.engine = MonteCarloEngine(seed=seed)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def analyze(self, asset_returns, weights, horizon_days=1,
                confidence_levels=(0.95, 0.99), n_paths=10_000,
                distribution="normal", portfolio_value=1.0):
        """Run the full risk analysis on a portfolio.

        Parameters
        ----------
        asset_returns : DataFrame   daily simple returns, one column per ticker.
        weights : array-like        portfolio weights (assumed already summing to 1).
        distribution : str          'normal', 't', or 'historical'.

        Returns a dict with VaR/CVaR per level, the simulated and historical
        horizon-return distributions, component-VaR fractions, the worst
        simulated scenarios, and Kupiec backtest results.
        """
        weights = np.asarray(weights, dtype=float)
        confidence_levels = list(confidence_levels)
        tickers = list(asset_returns.columns)

        # Historical daily portfolio returns (weighted sum of asset returns).
        port_hist = asset_returns.values @ weights
        mu = float(port_hist.mean())
        sigma = float(port_hist.std(ddof=1))

        # 1. Simulate horizon returns under the chosen distribution.
        sim_returns = self._simulate(port_hist, mu, sigma,
                                     horizon_days, n_paths, distribution)

        # 2. VaR / CVaR at each confidence level (positive loss magnitudes).
        var, cvar = {}, {}
        for cl in confidence_levels:
            v, c = self.engine.calculate_var_cvar(sim_returns, cl)
            var[cl] = v
            cvar[cl] = c

        # 3. Worst 5 simulated scenarios (most negative horizon returns).
        worst = np.sort(sim_returns)[:5]

        # 4. Component (marginal) VaR fractions -- where risk comes from.
        component = self._component_var(asset_returns, weights, horizon_days)

        # 5. Historical horizon returns (rolling sum) for the model-vs-data overlay.
        hist_series = pd.Series(port_hist, index=asset_returns.index)
        hist_horizon = hist_series.rolling(horizon_days).sum().dropna().values

        # 6. Kupiec POF backtest on the 1-day historical returns.
        backtest = self._kupiec(port_hist, mu, sigma,
                                distribution, confidence_levels)

        return {
            "var": var,
            "cvar": cvar,
            "sim_returns": sim_returns,
            "hist_horizon": hist_horizon,
            "component": component,
            "worst": worst,
            "backtest": backtest,
            "port_hist": port_hist,
            "hist_index": list(asset_returns.index),
            "mu": mu,
            "sigma": sigma,
            "horizon_days": horizon_days,
            "confidence_levels": confidence_levels,
            "distribution": distribution,
            "tickers": tickers,
        }

    # ------------------------------------------------------------------ #
    # Distribution sampling
    # ------------------------------------------------------------------ #
    def _simulate(self, hist, mu, sigma, horizon_days, n_paths, distribution):
        """Draw n_paths horizon returns (sum of horizon_days daily draws)."""
        rng = self.engine.rng

        if distribution == "normal":
            # Plain Gaussian fit to the historical mean/std.
            daily = rng.normal(mu, sigma, size=(n_paths, horizon_days))

        elif distribution == "t":
            # Student-t captures fat tails -> larger, more realistic tail risk.
            df, loc, scale = stats.t.fit(hist)
            df = max(df, 2.05)                       # keep variance finite
            daily = stats.t.rvs(df, loc=loc, scale=scale,
                                size=(n_paths, horizon_days), random_state=rng)

        elif distribution == "historical":
            # Non-parametric bootstrap: resample actual past daily returns.
            idx = rng.integers(0, len(hist), size=(n_paths, horizon_days))
            daily = np.asarray(hist)[idx]

        else:
            raise ValueError(f"unknown distribution: {distribution!r}")

        # Total return over the horizon = sum of the daily returns.
        return daily.sum(axis=1)

    # ------------------------------------------------------------------ #
    # Component / marginal VaR
    # ------------------------------------------------------------------ #
    def _component_var(self, asset_returns, weights, horizon_days):
        """Euler decomposition of portfolio risk into per-asset contributions.

        Under an elliptical assumption, each asset's share of total VaR is
        f_i = w_i * (Sigma w)_i / (w' Sigma w). These fractions sum to 1; a
        negative fraction means the holding is a net diversifier.
        """
        # Sample covariance of daily asset returns, scaled to the horizon.
        cov = np.atleast_2d(np.cov(asset_returns.values.T, ddof=1))
        cov_h = cov * horizon_days

        port_var = float(weights @ cov_h @ weights)
        port_sigma = float(np.sqrt(port_var)) if port_var > 0 else 0.0

        if port_var > 0:
            # Marginal contribution to portfolio sigma, and Euler fractions.
            marginal = (cov_h @ weights) / port_sigma
            fractions = weights * (cov_h @ weights) / port_var
        else:
            marginal = np.zeros_like(weights)
            fractions = np.zeros_like(weights)

        return {
            "fractions": fractions,           # share of total VaR per asset (sum=1)
            "marginal": marginal,             # marginal VaR per unit weight
            "port_sigma": port_sigma,
            "tickers": list(asset_returns.columns),
        }

    # ------------------------------------------------------------------ #
    # Kupiec POF backtest
    # ------------------------------------------------------------------ #
    def _one_day_var(self, hist, mu, sigma, distribution, cl):
        """Model-implied 1-day VaR (positive loss) at confidence cl."""
        alpha = 1.0 - cl
        if distribution == "normal":
            q = mu + sigma * stats.norm.ppf(alpha)
        elif distribution == "t":
            df, loc, scale = stats.t.fit(hist)
            df = max(df, 2.05)
            q = stats.t.ppf(alpha, df, loc=loc, scale=scale)
        else:  # historical
            q = np.percentile(hist, alpha * 100.0)
        return -float(q)

    @staticmethod
    def _kupiec_lr(n, x, p):
        """Likelihood-ratio statistic for the Kupiec POF (coverage) test."""
        if n == 0:
            return 0.0
        pi = x / n
        # Log-likelihood under the expected breach rate p.
        ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
        # Log-likelihood under the observed rate (guard the 0% / 100% cases).
        if x == 0:
            ll_alt = 0.0
        elif x == n:
            ll_alt = 0.0
        else:
            ll_alt = (n - x) * np.log(1 - pi) + x * np.log(pi)
        return float(-2.0 * (ll_null - ll_alt))

    def _kupiec(self, hist, mu, sigma, distribution, confidence_levels):
        """Backtest each VaR level against the historical daily returns."""
        hist = np.asarray(hist)
        n = len(hist)
        out = {}
        for cl in confidence_levels:
            var_1d = self._one_day_var(hist, mu, sigma, distribution, cl)
            # A breach is a day whose loss exceeded the VaR estimate.
            breaches = hist < -var_1d
            x = int(breaches.sum())
            p = 1.0 - cl
            lr = self._kupiec_lr(n, x, p)
            pval = float(1.0 - stats.chi2.cdf(lr, df=1))
            out[cl] = {
                "n": n,
                "exceptions": x,
                "expected": p * n,
                "rate": x / n if n else 0.0,
                "var_1d": var_1d,
                "LR": lr,
                "pval": pval,
                # Well-calibrated if we cannot reject correct coverage at 5%.
                "pass": pval > 0.05,
                "breaches": breaches,
            }
        return out
