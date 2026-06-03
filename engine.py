"""Core Monte Carlo simulation engine for finance.

Stage 1: the engine and its validation only -- no interface, API, or app.

The engine simulates asset price paths under Geometric Brownian Motion (GBM),
prices options by Monte Carlo, and computes risk metrics (VaR / CVaR).
"""

import numpy as np

from black_scholes import bs_price
from payoffs import european_call, european_put


class MonteCarloEngine:
    """A small Monte Carlo engine for simulating prices and pricing options."""

    def __init__(self, seed=None):
        # An optional seed makes runs reproducible (handy for tests).
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Path simulation
    # ------------------------------------------------------------------ #
    def simulate_gbm(self, S0, mu, sigma, T, dt, n_paths, antithetic=False):
        """Simulate price paths under Geometric Brownian Motion.

        Uses the exact (log-Euler) GBM solution, so each step is unbiased
        regardless of dt:

            S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

        If `antithetic` is True, each random shock Z is paired with its mirror
        -Z. This is a standard variance-reduction trick: the two paths are
        negatively correlated, so their average converges faster without adding
        bias. It noticeably tightens Monte Carlo pricing error for a given
        number of paths.

        Returns an array of shape (n_paths, n_steps + 1). Column 0 is S0, so
        the path includes its starting point (needed for Asian/barrier payoffs).
        """
        n_steps = int(round(T / dt))                       # number of time steps

        if antithetic:
            # Draw half the shocks, then append their negatives as mirror paths.
            half = (n_paths + 1) // 2
            Z_half = self.rng.standard_normal((half, n_steps))
            Z = np.concatenate([Z_half, -Z_half], axis=0)[:n_paths]
        else:
            # Standard normal shocks, one per (path, step).
            Z = self.rng.standard_normal((n_paths, n_steps))

        # Per-step drift and diffusion of the log-price.
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        log_increments = drift + diffusion * Z             # log-returns each step

        # Cumulative sum of log-returns gives the log-price relative to S0.
        log_paths = np.cumsum(log_increments, axis=1)

        # Build the full path array with S0 prepended as the first column.
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_paths)
        return paths

    def simulate_correlated_assets(self, S0_list, mu_list, sigma_list,
                                   correlation_matrix, T, dt, n_paths):
        """Simulate multiple correlated assets under GBM.

        Correlation is imposed via the Cholesky factor of the correlation
        matrix: if L L^T = C and Z is i.i.d. standard normal, then L @ Z has
        the desired correlation structure.

        Returns an array of shape (n_assets, n_paths, n_steps + 1).
        """
        S0_list = np.asarray(S0_list, dtype=float)
        mu_list = np.asarray(mu_list, dtype=float)
        sigma_list = np.asarray(sigma_list, dtype=float)
        correlation_matrix = np.asarray(correlation_matrix, dtype=float)

        n_assets = len(S0_list)
        n_steps = int(round(T / dt))

        # Cholesky decomposition: lower-triangular L with L @ L.T == C.
        L = np.linalg.cholesky(correlation_matrix)

        # Independent shocks: shape (n_assets, n_paths, n_steps).
        Z = self.rng.standard_normal((n_assets, n_paths, n_steps))

        # Mix the asset dimension with L to introduce the correlation.
        # 'ij,jpk->ipk': for every (path p, step k) we compute L @ Z[:, p, k].
        correlated = np.einsum("ij,jpk->ipk", L, Z)

        paths = np.empty((n_assets, n_paths, n_steps + 1))
        for a in range(n_assets):
            # Same GBM log-step as the single-asset case, per asset.
            drift = (mu_list[a] - 0.5 * sigma_list[a] ** 2) * dt
            diffusion = sigma_list[a] * np.sqrt(dt)
            log_increments = drift + diffusion * correlated[a]
            log_paths = np.cumsum(log_increments, axis=1)

            paths[a, :, 0] = S0_list[a]
            paths[a, :, 1:] = S0_list[a] * np.exp(log_paths)
        return paths

    # ------------------------------------------------------------------ #
    # Option pricing
    # ------------------------------------------------------------------ #
    def price_option(self, paths, payoff_func, r, T):
        """Price an option by averaging discounted payoffs.

        `payoff_func` maps a paths array to a 1-D array of payoffs (e.g. a
        lambda wrapping one of the functions in payoffs.py).

        Returns (price, standard_error), where the standard error quantifies
        the Monte Carlo sampling uncertainty of the price estimate.
        """
        payoffs = payoff_func(paths)                       # undiscounted payoff
        discounted = np.exp(-r * T) * payoffs              # present value
        price = discounted.mean()
        # Standard error of the mean = sample std / sqrt(N).
        std_err = discounted.std(ddof=1) / np.sqrt(len(discounted))
        return price, std_err

    # ------------------------------------------------------------------ #
    # Risk metrics
    # ------------------------------------------------------------------ #
    def calculate_var_cvar(self, portfolio_returns, confidence_level):
        """Compute Value-at-Risk and Conditional VaR.

        VaR at confidence c is the loss not exceeded with probability c.
        CVaR (a.k.a. Expected Shortfall) is the average loss in the worst
        (1 - c) tail. Both are returned as positive loss magnitudes.
        """
        portfolio_returns = np.asarray(portfolio_returns, dtype=float)
        alpha = 1.0 - confidence_level                     # tail probability

        # The (alpha)-quantile of returns marks the VaR threshold.
        var_threshold = np.percentile(portfolio_returns, alpha * 100.0)
        VaR = -var_threshold                               # report as a loss (>0)

        # CVaR averages the returns at or below that threshold (the tail).
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        CVaR = -tail_losses.mean()
        return VaR, CVaR

    # ------------------------------------------------------------------ #
    # Convergence analysis
    # ------------------------------------------------------------------ #
    def convergence_analysis(self, S0, K, r, sigma, T, option_type,
                             max_n, steps):
        """Price a European option at increasing path counts and compare to BS.

        Runs the MC pricer at `steps` path counts spaced logarithmically from
        100 to `max_n`. Returns (n_values, mc_prices, bs_analytical) so the
        caller can plot convergence against the exact Black-Scholes price.

        Note: pricing is done under the risk-neutral measure, so the drift is
        the risk-free rate r (not a real-world mu).
        """
        # Log-spaced, de-duplicated, integer path counts from 100 to max_n.
        n_values = np.unique(
            np.logspace(np.log10(100), np.log10(max_n), steps).astype(int)
        )

        dt = 1.0 / 252.0                                   # daily steps
        payoff_func = european_call if option_type == "call" else european_put

        mc_prices = []
        for n in n_values:
            # Risk-neutral simulation: drift = r. Antithetic variates reduce
            # the Monte Carlo error so convergence to BS is clean.
            paths = self.simulate_gbm(S0, r, sigma, T, dt, int(n), antithetic=True)
            price, _ = self.price_option(paths, lambda p: payoff_func(p, K), r, T)
            mc_prices.append(price)

        bs_analytical = bs_price(S0, K, r, sigma, T, option_type)
        return n_values, np.array(mc_prices), bs_analytical
