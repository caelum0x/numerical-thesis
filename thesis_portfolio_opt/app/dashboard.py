"""
Streamlit interactive dashboard for portfolio optimization.
Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import (
    RAW_DIR, PROCESSED_DIR, RESULTS_DIR,
    TICKERS, TICKER_LIST, FRED_SERIES,
    RISK_AVERSION_RANGE, STRESS_SCENARIOS, ASSET_CLASSES,
    PREDICTION_HORIZON,
)
from src.optimization.optimizer import (
    mean_variance_optimize, max_sharpe_optimize,
    minimum_variance_optimize, risk_parity_optimize,
    inverse_volatility_weights, estimate_covariance,
    efficient_frontier,
)
from src.optimization.backtester import (
    compute_portfolio_metrics, backtest_strategy,
    benchmark_equal_weight, stress_test, rolling_sharpe,
)

# --- Page Config ---
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# --- Sidebar ---
st.sidebar.title("📊 Portfolio Optimizer")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Optimization",
    "Backtesting",
    "Stress Testing",
    "Model Insights",
])

st.sidebar.markdown("---")
st.sidebar.header("Global Parameters")
risk_aversion = st.sidebar.slider("Risk Aversion (λ)", 0.1, 20.0, 2.0, 0.1)
lookback_years = st.sidebar.slider("Lookback (years)", 1, 10, 3)
cov_method = st.sidebar.selectbox("Covariance Method", ["sample", "shrinkage"])
selected_tickers = st.sidebar.multiselect(
    "Assets", list(TICKERS.keys()), default=list(TICKERS.keys())
)


# --- Data Loading ---
@st.cache_data
def load_prices():
    try:
        return pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None


@st.cache_data
def load_macro():
    try:
        return pd.read_csv(RAW_DIR / "macro.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None


all_prices = load_prices()
macro = load_macro()

if all_prices is None:
    st.error("Price data not found. Run `python run_pipeline.py --step fetch` first.")
    st.stop()

if not selected_tickers:
    st.warning("Select at least one asset.")
    st.stop()

prices = all_prices[selected_tickers]
returns = prices.pct_change().dropna()
lookback = lookback_years * 252
recent_returns = returns.iloc[-lookback:]
mu = recent_returns.mean().values * 252
cov = estimate_covariance(recent_returns, method=cov_method)


# =====================================================================
# PAGE: OVERVIEW
# =====================================================================
if page == "Overview":
    st.title("Market Overview")

    # Price chart
    st.subheader("Asset Prices (Normalized)")
    norm = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=f"{col} ({TICKERS[col]})", mode="lines"))
    fig.update_layout(yaxis_title="Indexed Value (Base=100)", yaxis_type="log", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.subheader("Summary Statistics (Annualized)")
    summary = pd.DataFrame({
        "Asset Class": [TICKERS[t] for t in selected_tickers],
        "Ann. Return": returns.mean() * 252,
        "Ann. Volatility": returns.std() * np.sqrt(252),
        "Sharpe": (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        "Skewness": returns.skew(),
        "Max Drawdown": pd.Series({
            col: ((prices[col] / prices[col].cummax()) - 1).min()
            for col in prices.columns
        }),
    }, index=selected_tickers)

    fmt = {
        "Ann. Return": "{:.2%}", "Ann. Volatility": "{:.2%}",
        "Sharpe": "{:.3f}", "Skewness": "{:.3f}", "Max Drawdown": "{:.2%}",
    }
    st.dataframe(summary.style.format(fmt), use_container_width=True)

    # Correlation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        corr = returns.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="equal")
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.subheader("Return Distributions")
        fig_dist = go.Figure()
        for col in returns.columns:
            fig_dist.add_trace(go.Histogram(x=returns[col], name=col, opacity=0.6, nbinsx=80))
        fig_dist.update_layout(barmode="overlay", height=500, xaxis_title="Daily Return")
        st.plotly_chart(fig_dist, use_container_width=True)

    # Macro
    if macro is not None:
        st.subheader("Macroeconomic Indicators")
        selected_macro = st.multiselect("Select Indicators", macro.columns.tolist(),
                                        default=macro.columns[:4].tolist())
        if selected_macro:
            fig_macro = make_subplots(rows=len(selected_macro), cols=1, shared_xaxes=True,
                                      subplot_titles=[FRED_SERIES.get(s, s) for s in selected_macro])
            for i, col in enumerate(selected_macro):
                fig_macro.add_trace(
                    go.Scatter(x=macro.index, y=macro[col], name=col, line=dict(width=1)),
                    row=i + 1, col=1,
                )
            fig_macro.update_layout(height=200 * len(selected_macro), showlegend=False)
            st.plotly_chart(fig_macro, use_container_width=True)


# =====================================================================
# PAGE: OPTIMIZATION
# =====================================================================
elif page == "Optimization":
    st.title("Portfolio Optimization")

    # Compute portfolios
    strategies = {}
    w_mv = mean_variance_optimize(mu, cov, risk_aversion=risk_aversion)
    if w_mv is not None:
        strategies[f"Mean-Variance (λ={risk_aversion})"] = w_mv

    w_sharpe = max_sharpe_optimize(mu, cov)
    if w_sharpe is not None:
        strategies["Max Sharpe"] = w_sharpe

    w_minvar = minimum_variance_optimize(cov)
    if w_minvar is not None:
        strategies["Min Variance"] = w_minvar

    w_rp = risk_parity_optimize(cov)
    strategies["Risk Parity"] = w_rp

    w_iv = inverse_volatility_weights(cov)
    strategies["Inv. Volatility"] = w_iv

    n = len(selected_tickers)
    strategies["Equal Weight"] = np.ones(n) / n

    # Weights comparison
    st.subheader("Optimal Weights")
    weights_df = pd.DataFrame(strategies, index=selected_tickers)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(weights_df.style.format("{:.1%}").background_gradient(cmap="Blues", axis=1),
                      use_container_width=True)

    with col2:
        fig_w = go.Figure()
        for strat in strategies:
            fig_w.add_trace(go.Bar(x=selected_tickers, y=strategies[strat] * 100, name=strat))
        fig_w.update_layout(barmode="group", yaxis_title="Weight (%)", height=400)
        st.plotly_chart(fig_w, use_container_width=True)

    # Efficient Frontier
    st.subheader("Efficient Frontier")
    ef = efficient_frontier(mu, cov, n_points=80)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=ef["volatility"] * 100, y=ef["return"] * 100,
        mode="lines", name="Efficient Frontier", line=dict(width=3),
    ))

    # Individual assets
    for i, ticker in enumerate(selected_tickers):
        vol = np.sqrt(cov[i, i]) * 100
        ret = mu[i] * 100
        fig_ef.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", name=ticker,
            text=[ticker], textposition="top center", marker=dict(size=8),
        ))

    # Named portfolios
    markers = {"Max Sharpe": "star", "Min Variance": "diamond", "Risk Parity": "square"}
    for name, marker in markers.items():
        if name in strategies:
            w = strategies[name]
            vol = np.sqrt(w @ cov @ w) * 100
            ret = (mu @ w) * 100
            fig_ef.add_trace(go.Scatter(
                x=[vol], y=[ret], mode="markers", name=name,
                marker=dict(size=14, symbol=marker, line=dict(width=2, color="black")),
            ))

    fig_ef.update_layout(
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        height=600,
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # Risk-return table
    st.subheader("Expected Risk-Return")
    port_stats = []
    for name, w in strategies.items():
        ret = mu @ w
        vol = np.sqrt(w @ cov @ w)
        port_stats.append({"Strategy": name, "Exp. Return": ret, "Exp. Vol": vol, "Sharpe": ret / vol})
    stats_df = pd.DataFrame(port_stats).set_index("Strategy")
    st.dataframe(stats_df.style.format({
        "Exp. Return": "{:.2%}", "Exp. Vol": "{:.2%}", "Sharpe": "{:.3f}"
    }), use_container_width=True)

    # Sensitivity
    st.subheader("Sensitivity: Weights vs Risk Aversion (λ)")
    lambdas = np.linspace(0.1, 20, 40)
    weight_paths = {t: [] for t in selected_tickers}
    for lam in lambdas:
        w = mean_variance_optimize(mu, cov, risk_aversion=lam)
        for i, t in enumerate(selected_tickers):
            weight_paths[t].append(w[i] if w is not None else np.nan)

    fig_sens = go.Figure()
    for t in selected_tickers:
        fig_sens.add_trace(go.Scatter(x=lambdas, y=weight_paths[t], name=t, mode="lines"))
    fig_sens.update_layout(xaxis_title="Risk Aversion (λ)", yaxis_title="Weight", height=400)
    st.plotly_chart(fig_sens, use_container_width=True)


# =====================================================================
# PAGE: BACKTESTING
# =====================================================================
elif page == "Backtesting":
    st.title("Backtesting Results")

    rebalance_freq = st.sidebar.slider("Rebalance Frequency (days)", 5, 63, 21)
    tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)

    st.info("Running backtests... this may take a moment.")

    # Run strategies
    results = {}
    for lam in [2.0, 5.0, 10.0]:
        name = f"MV (λ={lam})"
        results[name] = backtest_strategy(
            prices, risk_aversion=lam,
            rebalance_freq=rebalance_freq,
            transaction_cost_bps=tc_bps,
        )

    results["Equal Weight"] = benchmark_equal_weight(prices)

    # Metrics
    st.subheader("Performance Metrics")
    metrics_rows = []
    for name, r in results.items():
        m = r["metrics"]
        m["strategy"] = name
        if r["turnover"] is not None and len(r["turnover"]) > 0:
            m["avg_turnover"] = r["turnover"]["turnover"].mean()
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows).set_index("strategy")

    fmt = {
        "annualized_return": "{:.2%}", "annualized_volatility": "{:.2%}",
        "sharpe_ratio": "{:.3f}", "max_drawdown": "{:.2%}",
        "calmar_ratio": "{:.3f}", "total_return": "{:.2%}", "avg_turnover": "{:.4f}",
    }
    st.dataframe(metrics_df.style.format(fmt, na_rep="—"), use_container_width=True)

    # Cumulative returns
    st.subheader("Cumulative Returns")
    fig_cum = go.Figure()
    for name, r in results.items():
        cum = (1 + r["returns"]).cumprod()
        fig_cum.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name, mode="lines"))
    fig_cum.update_layout(yaxis_title="Growth of $1", yaxis_type="log", height=500)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Drawdowns
    st.subheader("Drawdowns")
    selected_strat = st.selectbox("Strategy", list(results.keys()))
    ret = results[selected_strat]["returns"]
    cum = (1 + ret).cumprod()
    dd = (cum / cum.cummax()) - 1

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, fill="tozeroy",
                                 fillcolor="rgba(255,0,0,0.3)", line=dict(color="red", width=1)))
    fig_dd.update_layout(yaxis_title="Drawdown (%)", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Weights over time
    st.subheader("Portfolio Weights Over Time")
    w_df = results[selected_strat]["weights"]
    fig_wt = go.Figure()
    for col in w_df.columns:
        fig_wt.add_trace(go.Scatter(
            x=w_df.index, y=w_df[col], name=col, stackgroup="one", mode="lines",
        ))
    fig_wt.update_layout(yaxis_title="Weight", yaxis_range=[0, 1], height=400)
    st.plotly_chart(fig_wt, use_container_width=True)

    # Rolling Sharpe
    st.subheader("Rolling Sharpe Ratio (1-Year)")
    fig_rs = go.Figure()
    for name, r in results.items():
        rs = rolling_sharpe(r["returns"], window=252)
        if len(rs) > 0:
            fig_rs.add_trace(go.Scatter(x=rs.index, y=rs.values, name=name, mode="lines"))
    fig_rs.add_hline(y=0, line_dash="dash", line_color="black")
    fig_rs.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
    fig_rs.update_layout(yaxis_title="Sharpe Ratio", height=400)
    st.plotly_chart(fig_rs, use_container_width=True)


# =====================================================================
# PAGE: STRESS TESTING
# =====================================================================
elif page == "Stress Testing":
    st.title("Stress Testing")

    st.subheader("Historical Stress Scenarios")
    scenario_df = pd.DataFrame([
        {"Scenario": name, "Start": start, "End": end}
        for name, (start, end) in STRESS_SCENARIOS.items()
    ])
    st.dataframe(scenario_df, use_container_width=True)

    # Run backtest for stress analysis
    lam = risk_aversion
    result = backtest_strategy(prices, risk_aversion=lam)
    eq_result = benchmark_equal_weight(prices)

    stress_df = stress_test(result["returns"], benchmark_returns=eq_result["returns"])

    if len(stress_df) > 0:
        st.subheader(f"Stress Performance (λ={lam})")

        fmt = {
            "total_return": "{:.2%}", "annualized_vol": "{:.2%}",
            "max_drawdown": "{:.2%}", "sharpe": "{:.3f}",
            "benchmark_return": "{:.2%}", "excess_return": "{:.2%}",
        }
        st.dataframe(stress_df.style.format(fmt, na_rep="—"), use_container_width=True)

        # Bar chart
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            x=stress_df.index, y=stress_df["total_return"] * 100,
            name=f"MV (λ={lam})", marker_color="steelblue",
        ))
        if "benchmark_return" in stress_df.columns:
            fig_stress.add_trace(go.Bar(
                x=stress_df.index, y=stress_df["benchmark_return"] * 100,
                name="Equal Weight", marker_color="coral",
            ))
        fig_stress.update_layout(
            barmode="group", yaxis_title="Total Return (%)", height=450,
        )
        fig_stress.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_stress, use_container_width=True)

        # Crisis deep-dive
        st.subheader("Crisis Deep-Dive")
        selected_crisis = st.selectbox("Select Scenario", list(STRESS_SCENARIOS.keys()))
        start, end = STRESS_SCENARIOS[selected_crisis]
        mask = (result["returns"].index >= pd.Timestamp(start)) & (result["returns"].index <= pd.Timestamp(end))
        crisis_ret = result["returns"][mask]
        crisis_eq = eq_result["returns"][mask]

        if len(crisis_ret) > 0:
            fig_crisis = go.Figure()
            fig_crisis.add_trace(go.Scatter(
                x=crisis_ret.index, y=(1 + crisis_ret).cumprod().values,
                name=f"MV (λ={lam})", mode="lines",
            ))
            fig_crisis.add_trace(go.Scatter(
                x=crisis_eq.index, y=(1 + crisis_eq).cumprod().values,
                name="Equal Weight", mode="lines",
            ))
            fig_crisis.update_layout(
                yaxis_title="Growth of $1",
                title=f"{selected_crisis}: {start} to {end}",
                height=400,
            )
            st.plotly_chart(fig_crisis, use_container_width=True)
    else:
        st.warning("No stress periods overlap with the backtest period.")


# =====================================================================
# PAGE: MODEL INSIGHTS
# =====================================================================
elif page == "Model Insights":
    st.title("Predictive Model Insights")

    # Load model comparison
    try:
        model_comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
        avg_comp = model_comp.groupby("model")[["rmse_mean", "r2_mean", "mae_mean"]].mean().sort_values("rmse_mean")

        st.subheader("Model Comparison (Averaged Across Assets)")
        st.dataframe(avg_comp.style.format("{:.6f}").background_gradient(
            subset=["rmse_mean"], cmap="RdYlGn_r"
        ).background_gradient(subset=["r2_mean"], cmap="RdYlGn"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_rmse = px.bar(avg_comp.reset_index(), x="model", y="rmse_mean",
                              title="RMSE by Model", color="rmse_mean", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col2:
            fig_r2 = px.bar(avg_comp.reset_index(), x="model", y="r2_mean",
                            title="R² by Model", color="r2_mean", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_r2, use_container_width=True)

        # Per-asset breakdown
        st.subheader("Per-Asset Model Performance")
        selected_model = st.selectbox("Model", model_comp["model"].unique())
        asset_perf = model_comp[model_comp["model"] == selected_model].sort_values("r2_mean", ascending=False)

        fig_asset = px.bar(asset_perf, x="ticker", y="r2_mean", title=f"{selected_model} — R² by Asset",
                           color="r2_mean", color_continuous_scale="RdYlGn")
        fig_asset.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_asset, use_container_width=True)

    except FileNotFoundError:
        st.warning("Model comparison data not found. Run `python run_pipeline.py --step train` first.")

    # Feature importance
    try:
        import pickle
        from src.models.trainer import get_feature_importance

        st.subheader("Feature Importance")
        model_files = list(RESULTS_DIR.glob("model_xgboost_SPY*.pkl"))
        if model_files:
            with open(model_files[0], "rb") as f:
                result = pickle.load(f)

            features_df = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)
            feature_cols = [c for c in features_df.columns if not c.endswith(f"_ret_{PREDICTION_HORIZON}d")]

            fi = get_feature_importance(result["model"], feature_cols, "xgboost")
            top_n = st.slider("Top N Features", 10, 50, 20)
            top = fi.head(top_n)

            fig_fi = px.bar(top.iloc[::-1], x="importance_pct", y="feature", orientation="h",
                            title=f"Top {top_n} Features (XGBoost — SPY)")
            fig_fi.update_layout(height=max(400, top_n * 25))
            st.plotly_chart(fig_fi, use_container_width=True)
    except Exception:
        pass


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Thesis: Macroeconomic Factor-Based Dynamic Portfolio Optimization")
