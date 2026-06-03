"""Portfolio risk estimation (Stage 3).

RiskEstimator turns a portfolio (tickers + weights) into Monte Carlo VaR/CVaR
estimates, a risk plot, and a deterministic stress-test scenario.
"""

import matplotlib

matplotlib.use("Agg")              # headless: we only ever save figures
import matplotlib.pyplot as plt
import numpy as np

from engine import MonteCarloEngine
from data_fetch import DataFetcher


class RiskEstimator:
    """Estimates portfolio VaR/CVaR and runs stress scenarios."""

    def __init__(self, seed=None):
        # Reuse the project's Monte Carlo engine (its RNG + VaR/CVaR routine).
        self.engine = MonteCarloEngine(seed=seed)
        self.fetcher = DataFetcher()

    # ------------------------------------------------------------------ #
    # VaR / CVaR
    # ------------------------------------------------------------------ #
    def estimate_var_cvar(self, tickers, weights, horizon_days=1,
                          confidence_levels=(0.95, 0.99), n_paths=10_000):
        """Monte Carlo VaR/CVaR for a portfolio over `horizon_days`.

        Fits a normal distribution to the historical daily portfolio returns,
        simulates `n_paths` horizon returns (sum of `horizon_days` daily draws),
        and reports VaR/CVaR (positive loss magnitudes) at each confidence level.
        """
        confidence_levels = list(confidence_levels)

        # 1. Historical daily portfolio returns, then fit Normal(mu, sigma).
        port_returns = self.fetcher.get_portfolio_data(tickers, weights)
        mu = float(port_returns.mean())
        sigma = float(port_returns.std(ddof=1))

        # 2. Simulate horizon returns. Each path sums `horizon_days` daily
        #    normal returns, so a 10-day return aggregates 10 daily shocks.
        daily_draws = self.engine.rng.normal(
            mu, sigma, size=(n_paths, horizon_days)
        )
        sim_returns = daily_draws.sum(axis=1)      # total return over horizon

        # 3. VaR/CVaR at each confidence level via the engine's routine.
        var = {}
        cvar = {}
        for cl in confidence_levels:
            v, c = self.engine.calculate_var_cvar(sim_returns, cl)
            var[cl] = v
            cvar[cl] = c

        return {
            "var": var,                            # {cl: VaR loss (>0)}
            "cvar": cvar,                          # {cl: CVaR loss (>0)}
            "distribution": sim_returns,           # simulated horizon returns
            "mu": mu,
            "sigma": sigma,
            "horizon_days": horizon_days,
            "confidence_levels": confidence_levels,
        }

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot_risk(self, results_dict):
        """Two-panel risk figure: return histogram + metrics table.

        Saves the figure as risk_results.png.
        """
        dist = results_dict["distribution"]
        cls = results_dict["confidence_levels"]
        var = results_dict["var"]
        cvar = results_dict["cvar"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Left: histogram with VaR/CVaR markers --------------------- #
        ax1.hist(dist, bins=60, color="steelblue", alpha=0.7, edgecolor="none")
        # VaR/CVaR are losses, so they sit on the negative-return side.
        var_colors = {0.95: "orange", 0.99: "red"}
        cvar_colors = {0.95: "darkorange", 0.99: "darkred"}
        for cl in cls:
            ax1.axvline(-var[cl], color=var_colors.get(cl, "orange"),
                        ls="--", lw=1.8,
                        label=f"{int(cl * 100)}% VaR = {var[cl] * 100:.2f}%")
            ax1.axvline(-cvar[cl], color=cvar_colors.get(cl, "darkred"),
                        ls=":", lw=1.8,
                        label=f"{int(cl * 100)}% CVaR = {cvar[cl] * 100:.2f}%")
        ax1.set_xlabel(f"Simulated {results_dict['horizon_days']}-day return")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Simulated portfolio returns with VaR / CVaR")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # --- Right: clean metrics table -------------------------------- #
        ax2.axis("off")
        rows = [["Metric", "Value (loss)"]]
        for cl in cls:
            rows.append([f"{int(cl * 100)}% VaR", f"{var[cl] * 100:.2f}%"])
            rows.append([f"{int(cl * 100)}% CVaR", f"{cvar[cl] * 100:.2f}%"])
        table = ax2.table(cellText=rows, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        # Bold the header row.
        for j in range(2):
            table[(0, j)].set_facecolor("#40466e")
            table[(0, j)].set_text_props(color="white", fontweight="bold")
        ax2.set_title(f"Risk metrics ({results_dict['horizon_days']}-day horizon)")

        fig.tight_layout()
        fig.savefig("risk_results.png", dpi=120)
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------ #
    # Stress test
    # ------------------------------------------------------------------ #
    def stress_test(self, tickers, weights, shock_dict, n_paths=10_000):
        """Apply a manual per-ticker shock and return the portfolio loss (%).

        `shock_dict` maps ticker -> fractional shock (e.g. {"SPY": -0.30}).
        Starting from a unit portfolio, each asset's value becomes
        weight * (1 + shock); the loss is the new value minus 1, i.e. the
        weighted sum of shocks. Tickers absent from shock_dict are unshocked.
        """
        weights = np.asarray(weights, dtype=float)
        # Look up each ticker's shock (default 0 if not specified).
        shocks = np.array([shock_dict.get(t, 0.0) for t in tickers])
        # New unit-portfolio value = sum(weight * (1 + shock)); loss = value - 1.
        portfolio_loss = float(np.sum(weights * shocks))
        return portfolio_loss
