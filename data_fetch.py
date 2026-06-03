"""Live market-data access for the Monte Carlo engine (Stage 2).

Wraps yfinance so the rest of the project can request a current price, a
historical-volatility estimate, the risk-free rate, a correlation matrix, or a
weighted portfolio return series -- all keyed by ticker.
"""

import datetime
import warnings

warnings.filterwarnings("ignore")          # quiet yfinance/urllib3 chatter

import numpy as np
import pandas as pd
import yfinance as yf


class DataFetcher:
    """Fetches market data from yfinance and shapes it for the engine."""

    # ------------------------------------------------------------------ #
    # Internal helper
    # ------------------------------------------------------------------ #
    def _download_closes(self, tickers, window=252):
        """Download daily closing prices and return a tidy DataFrame.

        Returns a DataFrame whose columns are the requested tickers (in order)
        and whose rows are trading days. We pull a bit more calendar history
        than `window` trading days and trim later, since weekends/holidays mean
        calendar days > trading days.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # ~1.7 calendar days per trading day, plus a buffer, covers `window`.
        lookback_days = int(window * 1.7) + 15
        start = datetime.date.today() - datetime.timedelta(days=lookback_days)

        data = yf.download(
            tickers, start=start.isoformat(),
            progress=False, auto_adjust=True,
        )
        closes = data["Close"]
        # A single ticker can come back as a Series; normalise to a DataFrame.
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=tickers[0])
        return closes.dropna()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_stock_price(self, ticker):
        """Return the most recent closing price for `ticker` as a float."""
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]          # collapse single-column frame
        return float(close.dropna().iloc[-1])

    def get_historical_volatility(self, ticker, window=252):
        """Annualised historical volatility from daily log returns.

        Takes the last `window` closing prices, computes daily log returns,
        and annualises their standard deviation by sqrt(252).
        """
        closes = self._download_closes(ticker, window)[ticker].tail(window)
        # Daily log returns: ln(P_t / P_{t-1}).
        log_returns = np.log(closes / closes.shift(1)).dropna()
        # Annualise: daily std * sqrt(trading days per year).
        vol = log_returns.std(ddof=1) * np.sqrt(252)
        return float(vol)

    def get_risk_free_rate(self):
        """Current 3-month T-bill yield (^IRX) as a decimal.

        ^IRX is quoted in percent (e.g. 3.62 means 3.62%), so we divide by 100.
        """
        data = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return float(close.dropna().iloc[-1]) / 100.0

    def get_correlation_matrix(self, tickers, window=252):
        """Correlation matrix of daily returns for `tickers` as a numpy array."""
        closes = self._download_closes(tickers, window)[tickers]
        # Daily simple returns over the last `window` days.
        returns = closes.pct_change().dropna().tail(window)
        # pandas .corr() gives the Pearson correlation matrix.
        return returns.corr().values

    def get_portfolio_data(self, tickers, weights, window=252):
        """Weighted daily portfolio return series for `tickers`/`weights`.

        `weights` must align with `tickers` and sum to 1. Returns a pandas
        Series indexed by date of the portfolio's daily return.
        """
        weights = np.asarray(weights, dtype=float)
        closes = self._download_closes(tickers, window)[tickers]
        returns = closes.pct_change().dropna().tail(window)
        # Weighted sum across the ticker columns -> single portfolio return.
        portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
        portfolio_returns.name = "portfolio_return"
        return portfolio_returns
