"""
End-to-end pipeline: Data -> Features -> Models -> Optimization -> Backtest.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --step fetch       # only fetch data
    python run_pipeline.py --step features    # only build features
    python run_pipeline.py --step train       # only train models
    python run_pipeline.py --step backtest    # only run backtests
    python run_pipeline.py --step all         # everything
"""

import argparse
import time
import pandas as pd
import numpy as np

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TICKER_LIST,
    TICKERS,
    PREDICTION_HORIZON,
    RISK_AVERSION_RANGE,
)


def step_fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: Fetch raw data from FRED and YFinance."""
    from src.data.fetcher import fetch_all

    print("=" * 60)
    print("STEP 1: FETCHING DATA")
    print("=" * 60)
    prices, macro = fetch_all(save=True)
    print(f"  Prices: {prices.shape}, Macro: {macro.shape}")
    return prices, macro


def step_features(prices: pd.DataFrame | None = None, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Step 2: Build feature matrix."""
    from src.data.preprocessor import build_features, stationarity_report

    print("\n" + "=" * 60)
    print("STEP 2: BUILDING FEATURES")
    print("=" * 60)

    if prices is None:
        prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True)
    if macro is None:
        macro = pd.read_csv(RAW_DIR / "macro.csv", index_col=0, parse_dates=True)

    features = build_features(prices, macro, save=True)

    # Stationarity report on returns
    daily_returns = prices.pct_change().dropna()
    print("\nStationarity of daily returns:")
    stationarity_report(daily_returns)

    return features


def step_train(features: pd.DataFrame | None = None) -> None:
    """Step 3: Train predictive models for each asset."""
    from src.models.trainer import train_all_models, save_model

    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MODELS")
    print("=" * 60)

    if features is None:
        features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col=0, parse_dates=True)

    horizon = PREDICTION_HORIZON
    all_comparisons = []

    for ticker in TICKER_LIST:
        target_col = f"{ticker}_ret_{horizon}d"
        if target_col not in features.columns:
            print(f"  Skipping {ticker}: target column {target_col} not found")
            continue

        print(f"\n--- Training models for {ticker} ({TICKERS[ticker]}) ---")

        # Separate features from target
        target = features[target_col]
        # Use non-target columns as features (exclude all return targets for this horizon)
        feature_cols = [c for c in features.columns if not c.endswith(f"_ret_{horizon}d")]
        X = features[feature_cols]

        # Align and drop NaN
        mask = X.notna().all(axis=1) & target.notna()
        X_clean = X[mask]
        y_clean = target[mask]

        result = train_all_models(X_clean, y_clean)

        # Save best model
        best_model_name = result["comparison"].iloc[0]["model"]
        best_result = result["models"][best_model_name]
        save_model(best_result, target_col)

        # Also save all models
        for model_name, model_result in result["models"].items():
            save_model(model_result, target_col)

        comp = result["comparison"].copy()
        comp["ticker"] = ticker
        all_comparisons.append(comp)

    # Save overall comparison
    if all_comparisons:
        full_comparison = pd.concat(all_comparisons, ignore_index=True)
        full_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        print(f"\nSaved model comparison to {RESULTS_DIR / 'model_comparison.csv'}")

        # Print summary
        print("\n--- MODEL COMPARISON SUMMARY ---")
        summary = full_comparison.groupby("model")[["rmse_mean", "r2_mean"]].mean()
        summary = summary.sort_values("rmse_mean")
        print(summary.to_string())


def step_backtest(prices: pd.DataFrame | None = None) -> dict:
    """Step 4: Run backtests with multiple strategies."""
    from src.optimization.backtester import (
        backtest_strategy,
        benchmark_equal_weight,
        compare_strategies,
        stress_test,
    )
    from src.optimization.optimizer import estimate_covariance

    print("\n" + "=" * 60)
    print("STEP 4: BACKTESTING")
    print("=" * 60)

    if prices is None:
        prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True)

    # Define strategies
    strategies = {}
    for lam in RISK_AVERSION_RANGE:
        strategies[f"MV (λ={lam})"] = {"risk_aversion": lam}

    # Run comparison
    comparison = compare_strategies(prices, strategies=strategies)

    # Print metrics
    print("\n--- BACKTEST RESULTS ---")
    metrics_df = comparison["metrics"]
    print(metrics_df.to_string())

    # Save metrics
    metrics_df.to_csv(RESULTS_DIR / "backtest_metrics.csv")
    print(f"\nSaved to {RESULTS_DIR / 'backtest_metrics.csv'}")

    # Stress test best strategy
    best_strategy = metrics_df["sharpe_ratio"].idxmax()
    print(f"\nBest strategy by Sharpe: {best_strategy}")

    best_returns = comparison["results"][best_strategy]["returns"]
    eq_returns = comparison["results"]["Equal Weight"]["returns"]

    stress_results = stress_test(best_returns, benchmark_returns=eq_returns)
    if len(stress_results) > 0:
        print("\n--- STRESS TEST ---")
        print(stress_results.to_string())
        stress_results.to_csv(RESULTS_DIR / "stress_test.csv")

    # Save portfolio returns for plotting
    for name, result in comparison["results"].items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        result["returns"].to_csv(RESULTS_DIR / f"returns_{safe_name}.csv")

    # Save weights of best strategy
    best_weights = comparison["results"][best_strategy]["weights"]
    best_weights.to_csv(RESULTS_DIR / "best_weights.csv")

    return comparison


def step_figures(prices: pd.DataFrame | None = None, comparison: dict | None = None) -> None:
    """Step 5: Generate thesis figures."""
    from src.visualization.plots import (
        plot_cumulative_returns,
        plot_weights_over_time,
        plot_drawdowns,
        plot_correlation_heatmap,
        plot_stress_test_results,
        plot_metrics_table,
        plot_rolling_sharpe,
    )
    from src.optimization.backtester import rolling_sharpe

    print("\n" + "=" * 60)
    print("STEP 5: GENERATING FIGURES")
    print("=" * 60)

    if prices is None:
        prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True)

    daily_returns = prices.pct_change().dropna()

    # Correlation heatmap
    print("  Correlation heatmap...")
    plot_correlation_heatmap(daily_returns)

    if comparison is not None:
        results = comparison["results"]
        metrics_df = comparison["metrics"]

        # Cumulative returns
        print("  Cumulative returns...")
        returns_dict = {name: r["returns"] for name, r in results.items()}
        plot_cumulative_returns(returns_dict)

        # Best strategy details
        best = metrics_df["sharpe_ratio"].idxmax()
        print(f"  Weights over time ({best})...")
        plot_weights_over_time(results[best]["weights"])

        print("  Drawdowns...")
        plot_drawdowns(results[best]["returns"])

        # Rolling Sharpe
        print("  Rolling Sharpe...")
        sharpe_dict = {}
        for name, r in results.items():
            rs = rolling_sharpe(r["returns"])
            if len(rs) > 0:
                sharpe_dict[name] = rs
        if sharpe_dict:
            plot_rolling_sharpe(sharpe_dict)

        # Metrics table
        print("  Metrics table...")
        plot_metrics_table(metrics_df)

        # Stress test figure
        stress_path = RESULTS_DIR / "stress_test.csv"
        if stress_path.exists():
            print("  Stress test chart...")
            stress_df = pd.read_csv(stress_path, index_col=0)
            plot_stress_test_results(stress_df)

    print("\nAll figures saved to", RESULTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis Portfolio Optimization Pipeline")
    parser.add_argument("--step", default="all", choices=["fetch", "features", "train", "backtest", "figures", "all"])
    args = parser.parse_args()

    start = time.time()

    if args.step == "fetch":
        step_fetch()
    elif args.step == "features":
        step_features()
    elif args.step == "train":
        step_train()
    elif args.step == "backtest":
        step_backtest()
    elif args.step == "figures":
        step_figures()
    elif args.step == "all":
        prices, macro = step_fetch()
        features = step_features(prices, macro)
        step_train(features)
        comparison = step_backtest(prices)
        step_figures(prices, comparison)

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
