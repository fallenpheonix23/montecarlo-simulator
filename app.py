"""Monte Carlo Simulator -- Streamlit web interface (Stage 4).

A thin, interactive UI layer on top of the existing engine/options/risk/
retirement modules. None of those modules are modified here; this file only
calls into them and renders the results with Plotly.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from options import OptionsPricer
from risk import RiskEstimator
from retirement import RetirementPlanner
from risk_analytics import PortfolioRiskAnalyzer
from data_fetch import DataFetcher
from payoffs import (
    european_call,
    european_put,
    asian_call,
    asian_put,
    barrier_knockout_call,
)

# --------------------------------------------------------------------- #
# Constants / page config
# --------------------------------------------------------------------- #
SEED = 42                 # reproducible Monte Carlo draws
DT = 1.0 / 252.0          # daily time step (matches options.py)

st.set_page_config(page_title="Monte Carlo Simulator", layout="wide")


# --------------------------------------------------------------------- #
# Cached data access
#
# DataFetcher calls hit yfinance. We wrap each call in a module-level
# st.cache_data function (ttl 1h), then expose them through a CachedFetcher
# that has the SAME method names as DataFetcher. We assign an instance of it
# onto each pricer/estimator so their internal `self.fetcher.<call>()` lookups
# transparently use the cache -- no edits to the underlying modules.
# --------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_stock_price(ticker):
    return DataFetcher().get_stock_price(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_volatility(ticker, window=252):
    return DataFetcher().get_historical_volatility(ticker, window)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_risk_free_rate():
    return DataFetcher().get_risk_free_rate()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_portfolio_data(tickers, weights, window=252):
    return DataFetcher().get_portfolio_data(list(tickers), list(weights), window)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_correlation(tickers, window=252):
    return DataFetcher().get_correlation_matrix(list(tickers), window)


class CachedFetcher:
    """Drop-in replacement for DataFetcher backed by st.cache_data."""

    def get_stock_price(self, ticker):
        return _cached_stock_price(ticker)

    def get_historical_volatility(self, ticker, window=252):
        return _cached_volatility(ticker, window)

    def get_risk_free_rate(self):
        return _cached_risk_free_rate()

    def get_portfolio_data(self, tickers, weights, window=252):
        # Tuples so the cache key is hashable.
        return _cached_portfolio_data(tuple(tickers), tuple(weights), window)

    def get_correlation_matrix(self, tickers, window=252):
        return _cached_correlation(tuple(tickers), window)


# --------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------- #
def _discounted_payoff_per_path(res, style, option_type, strike, barrier, T):
    """Recompute the discounted payoff for every simulated path.

    Used only to build the convergence chart (running mean of the MC price).
    Reuses the same payoff functions the engine used.
    """
    paths = res["paths"]
    if style == "european":
        payoff = (european_call(paths, strike) if option_type == "call"
                  else european_put(paths, strike))
    elif style == "asian":
        payoff = (asian_call(paths, strike) if option_type == "call"
                  else asian_put(paths, strike))
    else:  # barrier
        payoff = barrier_knockout_call(paths, strike, barrier)
    return np.exp(-res["risk_free_rate"] * T) * payoff


# Canonical default weights for the demo tickers; others fall back to equal.
DEFAULT_WEIGHTS = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}


def fmt_inr(x):
    """Format a number as Indian-grouped rupees, e.g. 2150000 -> ₹21,50,000."""
    sign = "-" if x < 0 else ""
    n = int(round(abs(x)))
    s = str(n)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:                      # group the rest in pairs
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}₹{grouped}"


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_single_returns(ticker, window=252):
    """Daily simple-return series for a single ticker (cached)."""
    s = DataFetcher().get_portfolio_data([ticker], [1.0], window)
    return s.rename(ticker)


def fetch_asset_returns(tickers, window=252):
    """Aligned DataFrame of daily returns, one column per ticker (cached)."""
    series = [_cached_single_returns(t, window) for t in tickers]
    df = pd.concat(series, axis=1, join="inner").dropna()
    df.columns = list(tickers)
    return df


# Plain-language explanations surfaced as hover tooltips throughout the app,
# so a first-time user can learn the jargon without leaving the page.
EXPLAIN = {
    "var": "Value at Risk — the most you'd expect to lose on a typical bad "
           "day at this confidence level. A 95% VaR of 2% means losses stay "
           "under 2% on about 95% of days; roughly 1 day in 20 is worse.",
    "cvar": "Conditional VaR (Expected Shortfall) — the AVERAGE loss on the "
            "bad days that breach VaR. It answers 'when it goes wrong, how bad "
            "is it on average?'. Always at least as large as VaR.",
    "horizon": "How far ahead we measure risk. Risk grows with time (roughly "
               "with the square root of time), so a 10-day loss can be much "
               "larger than a 1-day loss.",
    "dist": "How we model daily returns. Normal = simple bell curve. "
            "Student-t = fatter tails (more extreme days), usually more "
            "realistic. Historical = resamples your actual past returns.",
    "confidence": "How strict the estimate is. 99% looks at rarer, more "
                  "extreme losses than 95%.",
    "pv": "Your total holdings in rupees. Lets us turn percentage risk into "
          "real money.",
    "weights": "How your money is split across holdings. These auto-adjust to "
               "always total 100%.",
    "sims": "How many random future scenarios we simulate. More = smoother, "
            "more stable numbers, but a little slower.",
    "overlay": "Blue = simulated possible futures. Orange = what actually "
               "happened in the past. When they overlap, the model is grounded "
               "in real data.",
    "component": "Splits your TOTAL risk across holdings — the most useful "
                 "view here. A green (negative) bar means that holding is a "
                 "diversifier that REDUCES overall risk.",
    "worst": "The five nastiest outcomes out of every simulated scenario — a "
             "gut-check on tail risk, in both % and rupees.",
    "backtest": "Reality-check: counts how often real past returns would have "
                "breached this VaR. If breaches are close to expected, the "
                "model is well-calibrated and trustworthy.",
    "stress": "Manually shock each holding by a % to see the immediate "
              "portfolio impact — for 'what if the market drops 30%?' questions.",
    "opt_price": "The simulation's estimate of the option's fair value today.",
    "opt_se": "How precise that estimate is — smaller is more confident. It "
              "shrinks as you add more simulations.",
    "ret_median": "The middle outcome: half the simulations finish above this, "
                  "half below.",
    "ret_p10": "A pessimistic case — only 10% of simulations finish below this.",
    "ret_p90": "An optimistic case — only 10% of simulations finish above this.",
    "ret_ruin": "The chance your savings ever hit zero at any point on the way.",
}

# Consistent dark tooltip styling for all Plotly charts.
HOVER_STYLE = dict(bgcolor="rgba(15,17,26,0.92)", font_size=12,
                   bordercolor="rgba(120,120,140,0.6)")


def fmt_money(x, symbol="$"):
    """Currency-aware formatting: Indian grouping for ₹, Western otherwise."""
    if symbol == "₹":
        return fmt_inr(x)                          # already handles sign + grouping
    sign = "-" if x < 0 else ""
    return f"{sign}{symbol}{int(round(abs(x))):,}"


# Display currencies (label -> symbol used in formatting).
CURRENCIES = {
    "₹ Indian Rupee": "₹",
    "$ US Dollar": "$",
    "€ Euro": "€",
    "£ British Pound": "£",
    "¥ Japanese Yen": "¥",
    "A$ Australian Dollar": "A$",
    "C$ Canadian Dollar": "C$",
    "S$ Singapore Dollar": "S$",
}

# Approximate long-run historical (nominal) annual return / volatility per index.
# Picking one auto-fills the assumptions; "Custom" unlocks the sliders.
INDEX_PRESETS = {
    "S&P 500 (US)": (0.10, 0.15),
    "Nasdaq 100 (US tech)": (0.13, 0.20),
    "Nifty 50 (India)": (0.12, 0.18),
    "Sensex (India)": (0.12, 0.19),
    "FTSE 100 (UK)": (0.07, 0.13),
    "Euro Stoxx 50 (EU)": (0.08, 0.17),
    "Nikkei 225 (Japan)": (0.08, 0.18),
    "MSCI World (Global)": (0.08, 0.15),
    "Custom": None,
}

# Lightweight CSS: buttons grow/glow and metric cards lift on hover.
ANIM_CSS = """
<style>
div.stButton > button, div.stDownloadButton > button {
    transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 18px rgba(90,130,255,0.40);
}
div.stButton > button:active { transform: scale(0.97); }

div[data-testid="stMetric"] {
    transition: transform .15s ease, box-shadow .15s ease;
    border-radius: 10px;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}

details > summary:hover { color: #6ea8fe; }
</style>
"""


# ===================================================================== #
# Options Pricer page
# ===================================================================== #
def render_options():
    st.sidebar.header("Options Pricer inputs")
    ticker = st.sidebar.text_input("Ticker symbol", value="SPY").strip().upper()
    strike = st.sidebar.number_input("Strike price", min_value=0.01,
                                      value=100.0, step=1.0)
    expiry = st.sidebar.slider("Expiry (years)", 0.1, 3.0, 1.0, 0.1)
    option_type = st.sidebar.radio("Option type", ["Call", "Put"])
    style = st.sidebar.radio("Option style", ["European", "Asian", "Barrier"])

    # Barrier level only appears for the Barrier style.
    barrier = None
    if style == "Barrier":
        barrier = st.sidebar.number_input("Barrier level", min_value=0.01,
                                          value=90.0, step=1.0)

    n_paths = st.sidebar.slider("Number of simulations",
                                10_000, 100_000, 50_000, 10_000)

    if st.sidebar.button("Run", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Fetching {ticker} data and simulating "
                            f"{n_paths:,} paths..."):
                pricer = OptionsPricer(seed=SEED)
                pricer.fetcher = CachedFetcher()          # use the cache
                res = pricer.price(
                    ticker=ticker, strike=strike, expiry_years=expiry,
                    option_type=option_type.lower(), style=style.lower(),
                    barrier=barrier, n_paths=n_paths,
                )
            # Stash result + the inputs needed to redraw the charts.
            st.session_state["opt"] = {
                "res": res, "ticker": ticker, "strike": strike,
                "expiry": expiry, "option_type": option_type.lower(),
                "style": style.lower(), "barrier": barrier,
            }
        except Exception as exc:                          # graceful failure
            st.session_state.pop("opt", None)
            st.error(f"Could not price this option. "
                     f"Check the ticker has data on yfinance. (Details: {exc})")

    # ---- Render stored result (survives slider/widget reruns) -------- #
    if "opt" not in st.session_state:
        st.info("Set your parameters in the sidebar and press **Run**.")
        return

    s = st.session_state["opt"]
    res, strike = s["res"], s["strike"]

    # Fetched inputs info box.
    st.info(
        f"**Auto-fetched for {s['ticker']}** — "
        f"Stock price: **${res['stock_price']:.2f}**  |  "
        f"Volatility: **{res['volatility'] * 100:.1f}%**  |  "
        f"Risk-free rate: **{res['risk_free_rate'] * 100:.2f}%**"
    )

    # Headline metrics.
    m1, m2 = st.columns(2)
    m1.metric(f"{s['style'].title()} {s['option_type'].title()} price",
              f"${res['price']:.2f}", help=EXPLAIN["opt_price"])
    m2.metric("Standard error", f"±{res['std_error']:.4f}",
              help=EXPLAIN["opt_se"])

    # Two charts side by side.
    c1, c2 = st.columns(2)

    # Left: 200 sample paths with the strike line.
    paths = res["paths"]
    t_axis = np.arange(paths.shape[1]) * DT
    fig_paths = go.Figure()
    for i in range(min(200, paths.shape[0])):
        fig_paths.add_trace(go.Scatter(
            x=t_axis, y=paths[i], mode="lines",
            line=dict(width=0.5, color="rgba(70,90,180,0.18)"),
            hoverinfo="skip", showlegend=False,
        ))
    fig_paths.add_hline(y=strike, line_dash="dash", line_color="red",
                        annotation_text=f"Strike ${strike:.0f}")
    fig_paths.update_layout(title="200 sample simulated paths",
                            xaxis_title="Years", yaxis_title="Price",
                            margin=dict(l=10, r=10, t=40, b=10), height=420)
    c1.plotly_chart(fig_paths, use_container_width=True)

    # Right: convergence of the running MC mean from 1,000 to n_paths.
    disc = _discounted_payoff_per_path(
        res, s["style"], s["option_type"], strike, s["barrier"], s["expiry"])
    running_mean = np.cumsum(disc) / np.arange(1, len(disc) + 1)
    n_total = len(disc)
    xs = np.unique(np.logspace(np.log10(1000), np.log10(n_total),
                               60).astype(int))
    xs = xs[xs <= n_total]
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=xs, y=running_mean[xs - 1], mode="lines",
        line=dict(color="royalblue"), name="MC running price",
        hovertemplate="After %{x:,} sims<br>Price: $%{y:.2f}<extra></extra>"))
    fig_conv.add_hline(y=res["price"], line_dash="dash", line_color="green",
                       annotation_text=f"Final ${res['price']:.2f}")
    fig_conv.update_layout(title="Convergence of MC price",
                           xaxis_title="Number of simulations",
                           yaxis_title="Option price", xaxis_type="log",
                           hoverlabel=HOVER_STYLE,
                           margin=dict(l=10, r=10, t=40, b=10), height=420)
    c2.plotly_chart(fig_conv, use_container_width=True)


# ===================================================================== #
# Risk Estimator page
# ===================================================================== #
def render_risk():
    st.sidebar.header("Risk Estimator inputs")

    tickers_str = st.sidebar.text_input(
        "Tickers (comma separated)", value="SPY, TLT, GLD",
        help="The holdings in your portfolio, e.g. SPY, TLT, GLD. Any symbol "
             "available on Yahoo Finance works.")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    portfolio_value = st.sidebar.number_input(
        "Portfolio value (₹)", min_value=0.0, value=1_000_000.0, step=50_000.0,
        help=EXPLAIN["pv"])

    # ---- Auto-normalising weight sliders (no error-prone text box) ---- #
    st.sidebar.markdown("**Weights** — auto-normalised",
                        help=EXPLAIN["weights"])
    raw = []
    for t in tickers:
        default = DEFAULT_WEIGHTS.get(t, 1.0 / max(len(tickers), 1))
        raw.append(st.sidebar.slider(t, 0.0, 1.0, float(round(default, 2)),
                                     0.05, key=f"w_{t}"))
    raw = np.array(raw, dtype=float)
    total = raw.sum()
    if total <= 0:
        weights = np.ones(len(tickers)) / max(len(tickers), 1)
        st.sidebar.markdown(":red[Σ = 0.00 — using equal weights]")
    else:
        weights = raw / total
        # Live indicator: green tick since we always normalise to 1.
        st.sidebar.markdown(
            ":green[Σ = 1.00 ✓]  ·  "
            + ", ".join(f"{t} {w*100:.0f}%" for t, w in zip(tickers, weights)))

    dist_label = st.sidebar.selectbox(
        "Return distribution",
        ["Normal", "Student-t (fat tails)", "Historical bootstrap"],
        help=EXPLAIN["dist"])
    dist_key = {"Normal": "normal", "Student-t (fat tails)": "t",
                "Historical bootstrap": "historical"}[dist_label]

    horizon_label = st.sidebar.radio("Horizon", ["1 day", "5 days", "10 days"],
                                     help=EXPLAIN["horizon"])
    horizon_days = {"1 day": 1, "5 days": 5, "10 days": 10}[horizon_label]

    st.sidebar.write("Confidence levels")
    use_95 = st.sidebar.checkbox("95%", value=True, help=EXPLAIN["confidence"])
    use_99 = st.sidebar.checkbox("99%", value=True, help=EXPLAIN["confidence"])

    n_paths = st.sidebar.slider("Number of simulations",
                                10_000, 100_000, 10_000, 10_000,
                                help=EXPLAIN["sims"])

    if st.sidebar.button("Run", type="primary", use_container_width=True):
        cls = [c for c, on in [(0.95, use_95), (0.99, use_99)] if on]
        if len(tickers) < 1:
            st.error("Enter at least one ticker.")
        elif not cls:
            st.error("Select at least one confidence level.")
        else:
            try:
                with st.spinner("Fetching data and running risk analysis..."):
                    rets = fetch_asset_returns(tuple(tickers))
                    analyzer = PortfolioRiskAnalyzer(seed=SEED)
                    res = analyzer.analyze(
                        rets, weights, horizon_days=horizon_days,
                        confidence_levels=cls, n_paths=n_paths,
                        distribution=dist_key, portfolio_value=portfolio_value)
                st.session_state["risk"] = {
                    "res": res, "tickers": tickers, "weights": weights,
                    "pv": portfolio_value, "dist": dist_label,
                    "horizon": horizon_label}
            except Exception as exc:
                st.session_state.pop("risk", None)
                st.error(f"Could not run the risk analysis. Check the tickers "
                         f"have data on yfinance. (Details: {exc})")

    if "risk" not in st.session_state:
        st.info("Configure your portfolio in the sidebar and press **Run**.")
        return

    R = st.session_state["risk"]
    res, pv = R["res"], R["pv"]
    tickers, weights = R["tickers"], R["weights"]
    var, cvar = res["var"], res["cvar"]
    cls = res["confidence_levels"]
    primary = max(cls)                                  # most conservative level

    st.markdown(f"### Portfolio risk — {R['horizon']} horizon · "
                f"{R['dist']} · {fmt_inr(pv)}")

    # Friendly glossary for newcomers (hover the ? on each card too).
    with st.expander("📖 New to these terms? Quick 20-second glossary"):
        st.markdown(
            f"- **VaR** — {EXPLAIN['var']}\n"
            f"- **CVaR** — {EXPLAIN['cvar']}\n"
            f"- **Component VaR** — {EXPLAIN['component']}\n"
            f"- **Backtest** — {EXPLAIN['backtest']}\n\n"
            "💡 *Tip: hover the little **?** next to any number or heading for "
            "a quick explanation.*")

    # ---- Metric cards: percentage AND absolute money ----------------- #
    # Each card carries a hover ? explaining the metric in plain language.
    cards = [("95% VaR", var, 0.95, EXPLAIN["var"]),
             ("99% VaR", var, 0.99, EXPLAIN["var"]),
             ("95% CVaR", cvar, 0.95, EXPLAIN["cvar"]),
             ("99% CVaR", cvar, 0.99, EXPLAIN["cvar"])]
    for col, (label, d, cl, tip) in zip(st.columns(4), cards):
        with col.container(border=True):
            if cl in d:
                st.metric(label, f"{d[cl] * 100:.2f}%", help=tip)
                st.caption(f"≈ {fmt_inr(d[cl] * pv)} loss")
            else:
                st.metric(label, "—", help=tip)
                st.caption("not selected")

    # ---- Model-vs-data overlay --------------------------------------- #
    st.subheader("Simulated vs historical returns", help=EXPLAIN["overlay"])
    st.caption("The orange historical distribution shows the model is grounded "
               "in real data; the blue is the Monte Carlo simulation. "
               "*Hover the bars for details.*")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=res["sim_returns"], nbinsx=60, histnorm="probability density",
        name="Simulated", marker_color="steelblue", opacity=0.75,
        hovertemplate="<b>Simulated future</b><br>Return: %{x:.2%}"
                      "<br>Likelihood: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Histogram(
        x=res["hist_horizon"], nbinsx=40, histnorm="probability density",
        name="Historical", marker_color="orange", opacity=0.45,
        hovertemplate="<b>Actual past</b><br>Return: %{x:.2%}"
                      "<br>Likelihood: %{y:.2f}<extra></extra>"))
    # VaR/CVaR as clean full-height lines; labels live in the legend (no clutter).
    line_colors = {0.95: "orange", 0.99: "red"}
    for cl in cls:
        c = line_colors.get(cl, "white")
        fig.add_vline(x=-var[cl], line_dash="dash", line_color=c)
        fig.add_vline(x=-cvar[cl], line_dash="dot", line_color=c)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(dash="dash", color=c),
            name=f"{int(cl*100)}% VaR ({var[cl]*100:.2f}%)"))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(dash="dot", color=c),
            name=f"{int(cl*100)}% CVaR ({cvar[cl]*100:.2f}%)"))
    fig.update_layout(barmode="overlay", height=420,
                      xaxis_title=f"{res['horizon_days']}-day return",
                      yaxis_title="Likelihood", hovermode="x unified",
                      hoverlabel=HOVER_STYLE,
                      margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Marginal risk contribution ---------------------------------- #
    st.subheader("Where is the risk coming from?", help=EXPLAIN["component"])
    frac = res["component"]["fractions"]
    var_money = var[primary] * pv
    comp_money = frac * var_money
    bar_colors = ["#d62728" if f >= 0 else "#2ca02c" for f in frac]
    # Per-bar hover spells out the contribution in plain words.
    roles = ["adds risk" if f >= 0 else "reduces risk (diversifier)"
             for f in frac]
    customdata = np.array([[fmt_inr(m), role]
                           for m, role in zip(comp_money, roles)])
    figc = go.Figure(go.Bar(
        x=frac * 100, y=tickers, orientation="h", marker_color=bar_colors,
        text=[f"{f*100:.0f}%  ·  {fmt_inr(m)}" for f, m in zip(frac, comp_money)],
        textposition="auto", customdata=customdata,
        hovertemplate="<b>%{y}</b><br>Share of total risk: %{x:.1f}%"
                      "<br>Risk in money: %{customdata[0]}"
                      "<br>Effect: %{customdata[1]}<extra></extra>"))
    figc.update_layout(
        height=90 + 60 * len(tickers), hoverlabel=HOVER_STYLE,
        title=f"Component VaR (basis: {int(primary*100)}% VaR = {fmt_inr(var_money)})",
        xaxis_title="Share of total portfolio VaR (%)",
        margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figc, use_container_width=True)
    st.caption("Contributions sum to 100%. A **negative** (green) share means "
               "the holding is a net **diversifier** — it lowers total risk.")

    # ---- Worst-N scenarios ------------------------------------------- #
    st.subheader("5 worst simulated scenarios", help=EXPLAIN["worst"])
    worst = res["worst"]
    wdf = pd.DataFrame({
        "Scenario": [f"#{i+1}" for i in range(len(worst))],
        "Return": [f"{w*100:.2f}%" for w in worst],
        "Profit / Loss": [fmt_inr(w * pv) for w in worst],
    })
    st.table(wdf)

    # ---- Kupiec VaR backtest ----------------------------------------- #
    st.subheader("VaR backtest (Kupiec POF test)", help=EXPLAIN["backtest"])
    st.caption("How often actual historical returns would have breached this "
               "VaR. A calibrated model breaches roughly (1 − confidence) of "
               "the time. *Hover the red dots to see each breach.*")
    bt = res["backtest"]
    for col, cl in zip(st.columns(len(cls)), cls):
        b = bt[cl]
        with col.container(border=True):
            st.markdown(f"**{int(cl*100)}% VaR backtest**")
            st.write(f"Breaches: **{b['exceptions']}** / {b['n']} "
                     f"(expected {b['expected']:.0f})")
            st.write(f"Breach rate: **{b['rate']*100:.2f}%** "
                     f"(target {(1-cl)*100:.0f}%)")
            st.write(f"Kupiec LR = {b['LR']:.2f},  p = {b['pval']:.3f}")
            if b["pass"]:
                st.markdown(":green[✓ Well-calibrated]")
            else:
                st.markdown(":red[⚠ Miscalibrated]")

    # Breach time-series for the primary level.
    b = bt[primary]
    idx = res["hist_index"]
    daily = np.asarray(res["port_hist"])
    mask = b["breaches"]
    figb = go.Figure()
    figb.add_trace(go.Scatter(
        x=idx, y=daily, mode="lines", name="Daily return",
        line=dict(color="rgba(150,150,150,0.6)", width=1),
        hovertemplate="%{x|%d %b %Y}<br>Return: %{y:.2%}<extra></extra>"))
    figb.add_hline(y=-b["var_1d"], line_dash="dash", line_color="red",
                   annotation_text=f"{int(primary*100)}% VaR")
    figb.add_trace(go.Scatter(
        x=np.array(idx)[mask], y=daily[mask], mode="markers", name="Breach",
        marker=dict(color="red", size=6),
        hovertemplate="<b>VaR breach</b><br>%{x|%d %b %Y}"
                      "<br>Loss: %{y:.2%}<extra></extra>"))
    figb.update_layout(height=320, yaxis_title="Daily return",
                       hoverlabel=HOVER_STYLE,
                       margin=dict(l=10, r=10, t=20, b=10),
                       legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    figb.update_yaxes(tickformat=".0%")
    st.plotly_chart(figb, use_container_width=True)

    # ---- Stress test ------------------------------------------------- #
    st.subheader("Stress test", help=EXPLAIN["stress"])
    st.caption("Apply a manual price shock to each holding.")
    shocks = {}
    for col, tk in zip(st.columns(max(1, len(tickers))), tickers):
        shocks[tk] = col.slider(f"{tk} shock (%)", -50, 50, 0, 1,
                                key=f"shock_{tk}") / 100.0

    if st.button("Run Stress Test", use_container_width=True):
        est = RiskEstimator(seed=SEED)            # stress_test is deterministic
        loss = est.stress_test(tickers, list(weights), shocks)
        s1, s2 = st.columns(2)
        s1.metric("Portfolio impact", f"{loss * 100:.2f}%",
                  delta=f"{loss * 100:.2f}%")        # negative renders red
        s2.metric("In money terms", fmt_inr(loss * pv))


# ===================================================================== #
# Retirement Planner page
# ===================================================================== #
def _simulate_retirement(current_savings, monthly_contribution, years,
                         annual_return, annual_vol, n_paths):
    planner = RetirementPlanner(seed=SEED)
    return planner.simulate(
        current_savings=current_savings,
        monthly_contribution=monthly_contribution,
        years_to_retirement=years,
        annual_return=annual_return, annual_volatility=annual_vol,
        n_paths=n_paths)


def render_retirement():
    st.sidebar.header("Retirement Planner inputs")

    currency_label = st.sidebar.selectbox(
        "Currency", list(CURRENCIES.keys()),
        help="Display currency for all amounts on this page.")
    cur = CURRENCIES[currency_label]

    current_savings = st.sidebar.number_input(
        f"Current savings ({cur})", min_value=0.0, value=1_000_000.0,
        step=10_000.0, help="What you have saved so far.")
    monthly_contribution = st.sidebar.number_input(
        f"Monthly contribution ({cur})", min_value=0.0, value=5_000.0,
        step=500.0, help="How much you add every month.")
    years = st.sidebar.slider("Years to retirement", 5, 50, 30)

    # ---- Index preset drives the return/vol assumptions -------------- #
    index_choice = st.sidebar.selectbox(
        "Return assumptions (index)", list(INDEX_PRESETS.keys()),
        help="Pick an index to auto-fill its long-run historical return and "
             "volatility, or choose Custom to set your own.")
    preset = INDEX_PRESETS[index_choice]
    is_custom = preset is None

    if is_custom:
        annual_return_pct = st.sidebar.slider("Expected annual return (%)",
                                              1, 25, 10, key="ret_custom")
        annual_vol_pct = st.sidebar.slider("Expected annual volatility (%)",
                                           1, 50, 15, key="vol_custom")
    else:
        # Disabled sliders show the locked preset; dynamic keys refresh them
        # when the index changes.
        r0, v0 = int(round(preset[0] * 100)), int(round(preset[1] * 100))
        annual_return_pct = st.sidebar.slider(
            "Expected annual return (%)", 1, 25, r0, disabled=True,
            key=f"ret_{index_choice}")
        annual_vol_pct = st.sidebar.slider(
            "Expected annual volatility (%)", 1, 50, v0, disabled=True,
            key=f"vol_{index_choice}")
        st.sidebar.caption(f"Using {index_choice}: ~{r0}% return, ~{v0}% "
                           f"volatility (approx. historical).")

    inflation_pct = st.sidebar.slider(
        "Expected inflation (%)", 0, 12, 4,
        help="Used to also show results in today's money (real purchasing "
             "power), not just future rupees/dollars.")

    n_paths = st.sidebar.slider("Number of simulations", 1_000, 20_000,
                                10_000, 1_000, help=EXPLAIN["sims"])

    ann_ret = annual_return_pct / 100.0
    ann_vol = annual_vol_pct / 100.0
    inflation = inflation_pct / 100.0

    if st.sidebar.button("Run", type="primary", use_container_width=True):
        with st.spinner(f"Simulating {n_paths:,} wealth paths over "
                        f"{years} years..."):
            res = _simulate_retirement(current_savings, monthly_contribution,
                                       years, ann_ret, ann_vol, n_paths)
        st.session_state["retire"] = {
            "res": res, "cur": cur, "current_savings": current_savings,
            "monthly_contribution": monthly_contribution, "years": years,
            "ann_ret": ann_ret, "ann_vol": ann_vol, "n_paths": n_paths,
            "inflation": inflation, "index": index_choice,
        }

    if "retire" not in st.session_state:
        st.info("Set your plan in the sidebar and press **Run**.")
        return

    p = st.session_state["retire"]
    res, cur = p["res"], p["cur"]
    infl, yrs, months = p["inflation"], p["years"], res["months"]

    # ---- Nominal vs inflation-adjusted view -------------------------- #
    view = st.radio(
        "Show values in",
        [f"Nominal (future {cur})", f"Today's money (−{infl*100:.0f}% inflation)"],
        horizontal=True,
        help="Nominal = the actual future amount. Today's money = adjusted for "
             "inflation, i.e. what it could actually buy in today's terms.")
    real_view = view.startswith("Today")

    # Deflators: one for the final horizon, one per month for the path.
    final_deflator = (1 + infl) ** yrs
    month_years = np.arange(1, months + 1) / 12.0
    path_deflator = (1 + infl) ** month_years

    def adj(v):
        return v / final_deflator if real_view else v

    # Amount actually invested is a nominal cash figure: it does NOT change
    # with the inflation view (you literally contributed this many rupees).
    invested = p["current_savings"] + p["monthly_contribution"] * months

    median, p10, p90 = adj(res["median"]), adj(res["p10"]), adj(res["p90"])
    growth = median - invested

    # ---- Five headline metrics (now including Amount invested) ------- #
    cols = st.columns(5)
    cols[0].metric("Amount invested", fmt_money(invested, cur),
                   help="Total cash you put in: starting savings + every "
                        "monthly contribution, before any market growth. This "
                        "is a nominal figure and doesn't change with the "
                        "inflation view.")
    cols[1].metric("Median final wealth", fmt_money(median, cur),
                   delta=f"{fmt_money(growth, cur)} vs invested",
                   help=EXPLAIN["ret_median"])
    cols[2].metric("10th percentile", fmt_money(p10, cur),
                   help=EXPLAIN["ret_p10"])
    cols[3].metric("90th percentile", fmt_money(p90, cur),
                   help=EXPLAIN["ret_p90"])
    cols[4].metric("Probability of ruin", f"{res['prob_ruin']*100:.2f}%",
                   help=EXPLAIN["ret_ruin"])

    # Always show the other (adjusted/unadjusted) figure for context.
    if real_view:
        st.caption(f"Shown in **today's money** (adjusted for {infl*100:.0f}% "
                   f"inflation). The nominal future median would be "
                   f"**{fmt_money(res['median'], cur)}**.")
    else:
        st.caption(f"Shown as **future {cur}**. In today's money the median is "
                   f"**{fmt_money(res['median'] / final_deflator, cur)}** after "
                   f"{infl*100:.0f}% inflation over {yrs} years.")

    # ---- Fan chart (deflated if real view) --------------------------- #
    paths = res["paths"]
    disp_paths = paths / path_deflator if real_view else paths
    n_show = min(200, disp_paths.shape[0])
    t_years = np.arange(1, disp_paths.shape[1] + 1) / 12.0
    money_lbl = f"{cur} (today's money)" if real_view else cur

    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=t_years, y=disp_paths[i], mode="lines",
            line=dict(width=0.4, color="rgba(150,150,150,0.15)"),
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t_years, y=np.median(disp_paths, axis=0), mode="lines",
        line=dict(color="blue", width=2.5), name="Median",
        hovertemplate="Year %{x:.1f}<br>Median: " + cur +
                      "%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t_years, y=np.percentile(disp_paths, 90, axis=0), mode="lines",
        line=dict(color="red", dash="dash"), name="90th percentile",
        hovertemplate="Year %{x:.1f}<br>Optimistic (90th): " + cur +
                      "%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t_years, y=np.percentile(disp_paths, 10, axis=0), mode="lines",
        line=dict(color="red", dash="dash"), name="10th percentile",
        hovertemplate="Year %{x:.1f}<br>Pessimistic (10th): " + cur +
                      "%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Simulated wealth paths to retirement",
                      xaxis_title="Years from now",
                      yaxis_title=f"Wealth ({money_lbl})", height=460,
                      hoverlabel=HOVER_STYLE,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- What-if section (auto re-runs on slider change) ------------- #
    st.subheader("What if I save more?",
                 help="See instantly how an extra monthly contribution changes "
                      "your outcome.")
    extra = st.slider("Additional monthly contribution", 0, 50_000, 0, 1_000,
                      key="retire_extra")
    if extra > 0:
        with st.spinner("Re-running with the extra contribution..."):
            new_res = _simulate_retirement(
                p["current_savings"], p["monthly_contribution"] + extra,
                p["years"], p["ann_ret"], p["ann_vol"], p["n_paths"])
        new_median = adj(new_res["median"])
        med_delta = new_median - median
        ruin_delta = (new_res["prob_ruin"] - res["prob_ruin"]) * 100
        d1, d2 = st.columns(2)
        d1.metric("Median final wealth", fmt_money(new_median, cur),
                  delta=f"{fmt_money(med_delta, cur)}")
        d2.metric("Probability of ruin", f"{new_res['prob_ruin']*100:.2f}%",
                  delta=f"{ruin_delta:.2f}%", delta_color="inverse")
    else:
        st.caption("Drag the slider above to see the impact of saving more.")


# ===================================================================== #
# Landing page + routing
# ===================================================================== #
def main():
    st.markdown(ANIM_CSS, unsafe_allow_html=True)   # hover animations
    st.title("Monte Carlo Simulator")
    st.write(
        "This tool prices options, estimates portfolio risk (VaR / CVaR), and "
        "projects retirement wealth by running thousands of simulated market "
        "scenarios. It pulls live market data automatically, so you only need "
        "to pick a tool and enter your parameters."
    )
    st.caption("💡 New here? Hover the small **?** icons next to any number or "
               "heading, and hover over the charts, for plain-language "
               "explanations.")

    mode = st.selectbox(
        "Choose a tool",
        ["Options Pricer", "Risk Estimator", "Retirement Planner"],
    )
    st.divider()

    if mode == "Options Pricer":
        render_options()
    elif mode == "Risk Estimator":
        render_risk()
    else:
        render_retirement()


if __name__ == "__main__":
    main()
