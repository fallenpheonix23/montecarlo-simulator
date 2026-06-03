"""Retirement wealth projection via Monte Carlo (Stage 3).

RetirementPlanner simulates monthly wealth accumulation (contributions +
random market returns) and visualises the distribution of outcomes.
"""

import matplotlib

matplotlib.use("Agg")              # headless: we only ever save figures
import matplotlib.pyplot as plt
import numpy as np

from engine import MonteCarloEngine

# S&P 500 long-run historical defaults.
DEFAULT_ANNUAL_RETURN = 0.10
DEFAULT_ANNUAL_VOL = 0.15


class RetirementPlanner:
    """Projects retirement wealth paths under random monthly returns."""

    def __init__(self, seed=None):
        # Borrow the engine's RNG so draws are reproducible when seeded.
        self.engine = MonteCarloEngine(seed=seed)

    def simulate(self, current_savings, monthly_contribution,
                 years_to_retirement, annual_return=None,
                 annual_volatility=None, n_paths=10_000):
        """Simulate monthly wealth paths until retirement.

        Each month we add the contribution, then apply a random monthly return
        drawn from Normal(annual_return / 12, annual_volatility / sqrt(12)).

        Returns a dict with the full paths array (n_paths, months), the final
        wealth distribution, its median / 10th / 90th percentiles, and the
        probability of ruin (fraction of paths that ever hit zero or below).
        """
        # Fall back to S&P 500 historical defaults if not supplied.
        if annual_return is None:
            annual_return = DEFAULT_ANNUAL_RETURN
        if annual_volatility is None:
            annual_volatility = DEFAULT_ANNUAL_VOL

        months = int(round(years_to_retirement * 12))

        # Convert annualised parameters to monthly.
        monthly_mean = annual_return / 12.0
        monthly_vol = annual_volatility / np.sqrt(12.0)

        # Pre-draw all monthly returns: shape (n_paths, months).
        returns = self.engine.rng.normal(
            monthly_mean, monthly_vol, size=(n_paths, months)
        )

        paths = np.empty((n_paths, months))
        wealth = np.full(n_paths, float(current_savings))
        ever_ruined = np.zeros(n_paths, dtype=bool)

        # March month by month: contribute first, then apply the return.
        for t in range(months):
            wealth = (wealth + monthly_contribution) * (1.0 + returns[:, t])
            ever_ruined |= wealth <= 0.0            # track if it ever hit <= 0
            paths[:, t] = wealth

        final_wealth = paths[:, -1]
        return {
            "paths": paths,
            "final_wealth": final_wealth,
            "median": float(np.median(final_wealth)),
            "p10": float(np.percentile(final_wealth, 10)),
            "p90": float(np.percentile(final_wealth, 90)),
            "prob_ruin": float(ever_ruined.mean()),
            "years": years_to_retirement,
            "months": months,
        }

    def plot_retirement(self, results_dict):
        """Two-panel retirement figure: fan chart + final-wealth histogram.

        Saves the figure as retirement_results.png.
        """
        paths = results_dict["paths"]
        n_paths, months = paths.shape
        # X-axis in years from now.
        t_years = np.arange(1, months + 1) / 12.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Left: fan chart ------------------------------------------ #
        # Plot a sample of paths in light grey to show the spread.
        n_show = min(300, n_paths)
        for i in range(n_show):
            ax1.plot(t_years, paths[i], color="grey", alpha=0.05, lw=0.5)

        # Per-month median and 10th/90th percentile bands across paths.
        median_path = np.median(paths, axis=0)
        p10_path = np.percentile(paths, 10, axis=0)
        p90_path = np.percentile(paths, 90, axis=0)
        ax1.plot(t_years, median_path, color="blue", lw=2, label="Median")
        ax1.plot(t_years, p90_path, color="red", ls="--", lw=1.5,
                 label="90th percentile")
        ax1.plot(t_years, p10_path, color="red", ls="--", lw=1.5,
                 label="10th percentile")
        ax1.set_xlabel("Years from now")
        ax1.set_ylabel("Wealth ($)")
        ax1.set_title("Simulated wealth paths to retirement")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Right: final-wealth histogram ---------------------------- #
        final = results_dict["final_wealth"]
        median = results_dict["median"]
        p10 = results_dict["p10"]
        p90 = results_dict["p90"]
        prob_ruin = results_dict["prob_ruin"]

        ax2.hist(final, bins=60, color="seagreen", alpha=0.7, edgecolor="none")
        ax2.axvline(median, color="blue", lw=2,
                    label=f"Median = ${median:,.0f}")
        ax2.set_xlabel("Final wealth at retirement ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of final wealth")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Text annotations for the key statistics.
        annotation = (
            f"Median:  ${median:,.0f}\n"
            f"10th pct: ${p10:,.0f}\n"
            f"90th pct: ${p90:,.0f}\n"
            f"P(ruin): {prob_ruin * 100:.2f}%"
        )
        ax2.text(0.97, 0.97, annotation, transform=ax2.transAxes,
                 ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        fig.tight_layout()
        fig.savefig("retirement_results.png", dpi=120)
        plt.close(fig)
        return fig
