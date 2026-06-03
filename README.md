# Monte Carlo Simulator

An interactive finance toolkit built with Streamlit and a custom Monte Carlo
engine. It prices options, estimates portfolio risk (VaR / CVaR), and projects
retirement wealth using thousands of simulated market scenarios, pulling live
market data automatically via yfinance.

## Tools

- **Options Pricer** — European / Asian / Barrier options on any ticker, with
  live spot, historical volatility, and risk-free rate; sample-path and
  convergence charts.
- **Risk Estimator** — Monte Carlo VaR / CVaR with a choice of return
  distribution (Normal / Student-t / Historical bootstrap), marginal (component)
  VaR attribution, worst-case scenarios, a Kupiec backtest, and stress testing.
- **Retirement Planner** — monthly wealth projection with multiple currencies,
  index-based return presets, inflation-adjusted (real) vs nominal views, and a
  "what if I save more" explorer.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit web interface |
| `engine.py` | Core Monte Carlo engine (GBM, pricing, VaR/CVaR) |
| `black_scholes.py` | Analytical Black-Scholes (validation) |
| `payoffs.py` | Option payoff functions |
| `data_fetch.py` | Live market data via yfinance |
| `options.py` | Options pricer wiring data → engine |
| `risk.py` | Portfolio risk estimator + stress test |
| `risk_analytics.py` | Advanced risk: distributions, component VaR, backtest |
| `retirement.py` | Retirement wealth simulation |
| `test_*.py` | Validation test suites |

## Notes

- Live data depends on Yahoo Finance and can occasionally be rate-limited on
  shared cloud IPs; results are cached for one hour to reduce calls.
