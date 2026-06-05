# Thesis Defense Storyline

## Slide 1: Title and Core Claim

Macroeconomic Factor-Based Dynamic Portfolio Optimization with Autonomous Research and Swarm Intelligence.

Core claim: a closed-loop industrial engineering system can convert macroeconomic state variables into dynamic portfolio decisions, outperforming static benchmarks in economic terms while honestly reporting statistical limitations.

## Slide 2: Problem Definition

Static portfolio optimization assumes stable expected returns and covariances. The 2022--2024 period shows why this is fragile: interest rates, inflation, volatility, and equity-bond correlation shifted quickly.

Research problem: build a repeatable system that predicts returns, solves constrained allocations, uses autonomous search to improve configurations, and tests whether swarm consensus improves risk management.

## Slide 3: Literature and Background

The thesis connects five bodies of work:

- Markowitz mean-variance optimization.
- Macroeconomic factor models.
- Machine learning return prediction.
- Convex optimization with CVXPY.
- Autonomous research and swarm intelligence.

The gap is the integration of these ideas into one reproducible closed-loop framework.

## Slide 4: System Architecture

Three systems interact:

- AutoResearch searches models, features, and optimization parameters.
- MiroFish simulates 14 financial agents and produces an agreement-based risk overlay.
- thesis_portfolio_opt runs the integrated walk-forward backtest and feedback loop.

The loop is: predict -> optimize -> backtest -> compare -> feedback -> next experiment.

## Slide 5: Data and Methodology

Universe: 12 liquid ETFs across equities, bonds, commodities, real estate, and TIPS.

Data: 2005--2024 daily prices plus FRED macro indicators.

Evaluation: train through 2021, test out-of-sample on 2022--2024.

Optimization: long-only mean-variance portfolio with position limits, Ledoit-Wolf covariance, transaction costs, and monthly rebalance.

## Slide 6: Main Result

Best integrated strategy: ML-only LightGBM macro strategy.

- Sharpe: 1.33 at 10 bps, 1.41 at 3 bps.
- Annualized return: +19.6% to +20.7%.
- Max drawdown: -18.1% to -17.8%.
- SPY benchmark: Sharpe 0.57, return +10.0%, max drawdown -24.5%.

Economic result is strong; statistical significance is limited by the three-year OOS window.

## Slide 7: AutoResearch Findings

AutoResearch ran 72 experiments.

Key findings:

- LightGBM macro-only features dominate.
- Concentrated max-weight constraints perform best.
- SVR has the highest information coefficient and was discovered through feedback.
- PCA, regime splitting, and ensembles generally underperform.

## Slide 8: MiroFish Finding

MiroFish is useful as a risk overlay, not as an alpha engine.

ML+Swarm reduces volatility and improves drawdown, but lowers Sharpe. This supports a risk-averse institutional interpretation: use swarm disagreement to scale down risk, not to replace the return model.

## Slide 9: Phase 5 Robustness

The feedback-driven extension tested three ideas.

- Low-lambda swarm: Sharpe 0.813, still below baseline.
- Tight constraints: Sharpe 1.307 with maxW=0.3 and shrinkage=0.2, improving drawdown to -16.0%.
- IC-weighted ensemble: Sharpe 0.126, rejected.

Most important: the strategy remains strong under tighter institutional constraints.

## Slide 10: Limitations and Conclusion

Limitations:

- OOS sample is only 2022--2024.
- Backtests cannot capture every live trading friction.
- Macro data vintage effects are not fully modeled.
- The search space is broad but not exhaustive.

Conclusion: the thesis provides a reproducible industrial engineering framework for dynamic financial decision-making under uncertainty, combining prediction, optimization, simulation, and autonomous feedback.
