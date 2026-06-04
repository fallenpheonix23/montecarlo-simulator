# Monte Carlo Simulator — Project Report

*Prepared 2026-06-04*

---

## 1. Objective

Build a Python Monte Carlo simulation engine for finance, validate it
rigorously, wrap it in live market data, add risk and retirement tooling, expose
it through an interactive web interface, and deploy it publicly — built in
disciplined, test-gated stages.

## 2. Tech stack

`numpy` · `scipy` · `pandas` · `matplotlib` · `plotly` · `yfinance` ·
`streamlit`. Python 3.9 locally; deployed on Streamlit Community Cloud
(Python 3.14).

---

## 3. How it started → where it went (chronology)

### Stage 1 — Core engine + validation
**Built:** `engine.py` (`MonteCarloEngine`: `simulate_gbm`,
`simulate_correlated_assets` via Cholesky, `price_option`,
`calculate_var_cvar`, `convergence_analysis`), `black_scholes.py` (analytical
`bs_price` for validation), `payoffs.py` (European/Asian/barrier-knockout call &
put), and `test_engine.py` (5 tests + a 2-panel figure).

**What we got:** 5/5 passing — after two genuine findings (see §4). The MC price
converged to within **0.04%** of Black-Scholes.

### Stage 2 — Live data + options pricer
**Built:** `data_fetch.py` (`DataFetcher`: spot price, historical volatility,
risk-free rate via `^IRX`, correlation matrix, portfolio returns), `options.py`
(`OptionsPricer.price` — auto-fetches inputs, simulates, prices, returns a rich
dict), `test_stage2.py` (5 live tests).

**What we got:** 5/5 on live data (SPY ≈ $759, σ ≈ 12%, r ≈ 3.6%). One flaky
test fixed with common random numbers (§4).

### Stage 3 — Risk & retirement
**Built:** `risk.py` (`RiskEstimator`: `estimate_var_cvar`, `plot_risk`,
`stress_test`), `retirement.py` (`RetirementPlanner`: `simulate`,
`plot_retirement`), `test_stage3.py` (5 tests). Saved `risk_results.png` and
`retirement_results.png`.

**What we got:** 5/5 on the **first run** — these relationships were structural,
not noise-sensitive.

### Stage 4 — Streamlit web interface
**Built:** `app.py` — a three-tool UI (Options Pricer, Risk Estimator,
Retirement Planner). Charts rebuilt in **Plotly** (interactive) instead of the
modules' matplotlib, without editing those modules. Introduced a `CachedFetcher`
(swapped onto pricer/estimator instances) for `st.cache_data` caching, and
`st.session_state` for result persistence. Verified end-to-end with Streamlit's
`AppTest` harness.

### Stage 5 — Professionalisation
**Risk Estimator overhaul** — new `risk_analytics.py`
(`PortfolioRiskAnalyzer`): distribution choice (Normal / Student-t / Historical
bootstrap), **component (marginal) VaR** attribution, worst-5 scenarios, and a
**Kupiec POF backtest**. UI fixes: de-cluttered VaR labels into a legend, fixed
a stale-title bug, removed duplicated metrics, added bordered metric cards
showing **% and ₹**, auto-normalising weight sliders, a portfolio-value input,
and a model-vs-history overlay.

**Interactivity** — hover `?` tooltips on every metric/input/heading, rich
Plotly `hovertemplate` analysis on charts, a glossary expander, and CSS **hover
animations** (buttons grow + glow, metric cards lift).

**Retirement upgrades** — an *Amount invested* metric, an **inflation-adjusted
vs nominal** toggle, **8 currencies**, and **9 index presets** (S&P 500,
Nasdaq 100, Nifty 50, Sensex, FTSE 100, Euro Stoxx 50, Nikkei 225, MSCI World,
Custom).

### Deployment
Created an isolated repo folder with `requirements.txt`, `.gitignore`,
`README.md`, and `.streamlit/config.toml` (locks dark theme); committed and
pushed to `github.com/fallenpheonix23/montecarlo-simulator`. Worked around a
GitHub repo name that had an accidental leading dash, then resolved a Streamlit
Cloud build issue (host defaulted to Python 3.14, which lacked wheels for the
originally pinned numpy/scipy) by loosening the requirement pins. **App is live.**

---

## 4. Key findings & decisions

| # | Finding | Resolution |
|---|---------|------------|
| 1 | Stage 1 convergence sat at **2.01%** — pure Monte Carlo noise at 10k paths is ~1.5–2%, right on the limit. | Added **antithetic variates** (variance reduction; default off, on for pricing). Error dropped to **0.04%**. |
| 2 | Stage 1 put-call parity showed **23%** error — but only because it was measured against `S0 − K·e^(−rT) ≈ 0.12`, a near-zero difference of two ~100-sized numbers. | Re-based the tolerance on **S0** (the conventional denominator). Parity held to **0.02%**. |
| 3 | Stage 2 barrier test was flaky — the barrier and European legs used **separate** simulations, so a rarely-binding barrier could price *above* European on noise. | **Common random numbers** (re-seed so both legs share paths) — makes the inequality exact. |
| 4 | `stress_test` is deterministic; its `n_paths` arg is unused. | Kept for interface consistency; documented so it isn't mistaken for a bug. |
| 5 | On broad ETFs over ~1yr, **Student-t ≈ Normal** (high fitted degrees of freedom). | Expected, not a bug — fat tails only diverge on noisier assets/longer windows; verified t ≥ normal on fat-tailed input. |
| 6 | Component VaR revealed **TLT/GLD act as diversifiers** (negative risk contribution) while SPY dominates. | Surfaced as a colour-coded bar chart — the most valuable single analytic. |
| 7 | Retirement medians look huge (₹2.39 Cr). | Correct given a 10%/15% assumption compounding 30 years; flagged as optimistic, and the inflation view shows real purchasing power. |
| 8 | Streamlit Cloud built on **Python 3.14**; pinned numpy/scipy had no 3.14 wheels. | Loosened `requirements.txt` to floors so the host installs prebuilt wheels. |

---

## 5. Representative results (live data)

- **Options:** SPY European call, strike 5% OTM, 0.5y → ≈ $15.5, standard error
  < 1% of price.
- **Risk (60/40 SPY/TLT, 1-day):** 95% VaR **0.87%**, 99% VaR **1.27%**; CVaR
  **1.11% / 1.47%**; 10-day VaR scales ≈ √10. Kupiec backtest: well-calibrated.
- **Retirement (₹10,00,000 + ₹5,000/mo, 30y, S&P 500):** invested
  **₹28,00,000** → median **₹2.39 Cr** nominal (**₹73.8 L** in today's money at
  4% inflation), P(ruin) 0.00%.

## 6. Test status

| Suite | Tests | Status |
|-------|-------|--------|
| `test_engine.py` (Stage 1) | 5 | 5/5 |
| `test_stage2.py` (Stage 2, live) | 5 | 5/5 |
| `test_stage3.py` (Stage 3) | 5 | 5/5 |
| App (`AppTest`, all 3 pages + flows) | — | no errors |

**15/15 automated tests passing**, plus full UI smoke-tests. The Stage 1–3
suites stayed green throughout Stages 4–5 because the UI and analytics were built
*on top of*, never inside, the tested modules.

---

## 7. File inventory

| File | Role |
|------|------|
| `engine.py` | Core MC engine (GBM, correlated assets, pricing, VaR/CVaR, convergence) |
| `black_scholes.py` | Analytical Black-Scholes (validation only) |
| `payoffs.py` | European / Asian / barrier-knockout payoffs |
| `data_fetch.py` | Live market data via yfinance |
| `options.py` | `OptionsPricer` — data → engine → price |
| `risk.py` | `RiskEstimator` — VaR/CVaR, stress test, matplotlib plots |
| `risk_analytics.py` | `PortfolioRiskAnalyzer` — distributions, component VaR, Kupiec backtest |
| `retirement.py` | `RetirementPlanner` — wealth simulation, matplotlib plots |
| `app.py` | Streamlit UI (3 tools, Plotly, caching, tooltips, animations) |
| `test_engine.py`, `test_stage2.py`, `test_stage3.py` | Validation suites |
| `requirements.txt`, `.gitignore`, `README.md`, `.streamlit/config.toml` | Deployment config |

## 8. Known limitations / caveats

- **yfinance on shared cloud IPs** can be rate-limited by Yahoo; 1-hour caching
  mitigates it.
- **Index return/vol presets are approximate historical figures**, not
  live-fetched.
- **Component VaR** uses the Gaussian/elliptical (Euler) decomposition — exact
  for Normal/t, an approximation under the historical bootstrap.
- GBM assumes constant volatility and lognormal prices (no vol clustering /
  regime shifts).

## 9. Suggested next steps

- Simulation-based component-VaR attribution for full consistency under
  bootstrap.
- Live-fetched index assumptions; a Greeks panel for options; CSV/PDF export of
  results; optional custom domain.

---

## 10. Current state

- **Tests:** 15/15 passing across all stages.
- **Repo:** `github.com/fallenpheonix23/montecarlo-simulator` (branch `main`).
- **Deploy:** live on Streamlit Community Cloud.
