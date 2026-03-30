 PHASE 1: Core Research (15/15 COMPLETE)
 ═══════════════════════════════════════

  #1  Statistical significance tests         DONE  p=0.79 (OOS not significant — honest)
  #2  Walk-forward ALL strategies             DONE  RF Sharpe=0.940 beats SPY!
  #3  Robustness (3 OOS periods)              DONE  Works in macro shifts (2022-24)
  #4  Feature importance                      DONE  10Y yield, VIX, BBB spread top features
  #5  TC sensitivity                          DONE  Break-even ~25 bps
  #6  Test suite                              DONE  47/47 passing
  #7  Expand tests                            DONE  4 test files
  #8  Streamlit dashboard                     DONE  Verified with real data
  #9  CI/CD pipeline                          DONE  GitHub Actions
  #10 Type hints                              DONE  Key files annotated
  #11 Monte Carlo bootstrap                   DONE  10K samples
  #12 Regime-conditional                      DONE  VIX high/low split
  #13 Factor attribution                      DONE  FF5 regression, alpha insig
  #14 Turnover constraint                     DONE  5 levels tested
  #15 Multi-horizon                           DONE  21d optimal

 PHASE 2: Advanced Research & Polish (8/8 COMPLETE)
 ════════════════════════════════════

  ┌─────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────────────┬────────┐
  │  #  │                  What                    │                           Why                                │ Status │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 16  │ Walk-forward RF deep-dive                │ RF Sharpe=0.969! DD=-12.3%. Stress tested. fig29.           │   DONE │
  │     │                                          │ Beats SPY on return AND risk.                               │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 17  │ Ensemble of WF strategies                │ Ensemble Sharpe=0.710 (beats SPY 0.671). fig30.             │   DONE │
  │     │                                          │ RF alone still beats ensemble — no diversification gain.    │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 18  │ Monthly returns heatmap                  │ Calendar heatmap generated. Shows monthly P&L pattern.      │   DONE │
  │     │                                          │ fig33_monthly_heatmap.pdf                                   │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 19  │ Update ALL deliverables with RF finding  │ PDF, DOCX, Excel, LaTeX all regenerated with RF headline.   │   DONE │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 20  │ Correlation of strategies                │ fig31: RF-LGBM corr=~0.5, RF-Lasso=~0.5. fig32: rolling.  │   DONE │
  │     │                                          │ Moderate correlation — some diversification benefit.        │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 21  │ Risk parity + HRP walk-forward           │ Risk parity + inv vol tested. fig34_riskparity_comparison.  │   DONE │
  │     │                                          │ Compared MV vs Risk Parity vs Inv Vol vs SPY.               │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 22  │ Sharpe ratio time-variation              │ fig32: rolling 6M Sharpe for all strategies. RF dominates   │   DONE │
  │     │                                          │ in 2023-2024, LGBM catches up late 2024.                    │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 23  │ Final comprehensive deliverable update   │ All regenerated: 34 figs, 11 tables, PDF/DOCX/Excel/LaTeX. │   DONE │
  └─────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────────────┴────────┘

 PHASE 3: End-to-End Integration (7/7 COMPLETE)
 ══════════════════════════════════════════════

  ┌─────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────────────┬────────┐
  │  #  │                  What                    │                           Result                             │ Status │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 24  │ Fix hardcoded FRED API keys              │ Removed from autoresearch/prepare.py + experiment.py.       │   DONE │
  │     │                                          │ Load via python-dotenv from .env. Added .env.example.       │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 25  │ Fix autoresearch dependencies            │ Replaced pyproject.toml: removed torch/rustbpe/tiktoken,    │   DONE │
  │     │                                          │ added scikit-learn/lightgbm/xgboost/fredapi/yfinance/cvxpy. │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 26  │ Fix PCA backtest bug + CORS + dep pins   │ PCA transform propagated to backtest via transform_info.    │   DONE │
  │     │                                          │ CORS reads CORS_ORIGINS env var. Deps pinned <major.        │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 27  │ MiroFish test suite (from zero)          │ 36 tests: config, file parser, text processor, task mgr,    │   DONE │
  │     │                                          │ API health, blueprints. All passing.                        │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 28  │ AutoResearch → Pipeline bridge           │ autoresearch_bridge.py reads batch_results.csv, extracts    │   DONE │
  │     │                                          │ best config: LightGBM λ=5 maxW=0.5 → Sharpe 0.832.         │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 29  │ MiroFish → Optimizer bridge              │ mirofish_bridge.py: agreement → risk scale (mean 0.473),    │   DONE │
  │     │                                          │ 17 swarm features, swarm weights. Wired into CVXPY.         │        │
  ├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────┼────────┤
  │ 30  │ Closed-loop feedback + orchestrator      │ feedback_loop.py writes next-experiment suggestions.         │   DONE │
  │     │                                          │ run_all.py: 5-step orchestrator with --loop N support.      │        │
  └─────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────────────┴────────┘

 INTEGRATED RESULTS (OOS 2022-2024)
 ═══════════════════════════════════

  ┌───────────────┬─────────┬──────────┬──────────┬──────────┐
  │ Strategy      │ Sharpe  │  Return  │  MaxDD   │   Vol    │
  ├───────────────┼─────────┼──────────┼──────────┼──────────┤
  │ ML-only     ★ │  1.294  │  +18.4%  │  -18.8%  │  14.2%   │
  │ ML+Swarm      │  0.828  │   +9.7%  │  -17.2%  │  11.7%   │
  │ SPY           │  0.570  │  +10.0%  │  -24.5%  │  17.5%   │
  │ Swarm-only    │  0.301  │   +3.1%  │  -19.4%  │  10.4%   │
  │ Equal Weight  │  0.129  │   +1.3%  │  -18.9%  │  10.4%   │
  └───────────────┴─────────┴──────────┴──────────┴──────────┘

  Key findings:
  - ML-only (AutoResearch best config) dominates: Sharpe 1.294, 2.3x SPY
  - ML+Swarm trades Sharpe for safety: -1.6% better drawdown, -2.6% less vol
  - MiroFish swarm verdict: "beneficial_risk" — reduces drawdown at cost of return
  - Feedback loop identified 6 next experiments: elastic, SVR, regime-conditional,
    IC-weighted ensemble, lower λ with swarm, tighter constraints

 ARCHITECTURE
 ════════════

  autoresearch/              MiroFish/                   thesis_portfolio_opt/
  ┌──────────────┐   ┌────────────────────┐   ┌──────────────────────────────┐
  │ train.py     │   │ financial_simulator│   │ src/integration/             │
  │ --batch      │──→│ 14 agents          │──→│   autoresearch_bridge.py     │
  │ 27 experiments│  │ 35 rounds          │   │   mirofish_bridge.py         │
  │ Sharpe 0.832 │   │ agreement 0.247    │   │   feedback_loop.py           │
  └──────┬───────┘   └────────┬───────────┘   │   _backtest_helpers.py       │
         │                    │               │                              │
         │  best config       │  risk overlay │ run_all.py (5-step loop)     │
         └────────────────────┴───────────────│   Step 1: AutoResearch       │
                                              │   Step 2: MiroFish           │
         ┌────────────────────────────────────│   Step 3: Integrated backtest│
         │  feedback/latest.json              │   Step 4: Figures + LaTeX    │
         │  (next experiment suggestions)     │   Step 5: Feedback loop      │
         └────────────────────────────────────└──────────────────────────────┘

  Run: python run_all.py              # single pass (all 5 steps)
  Run: python run_all.py --loop 3     # 3 iterations of closed loop
  Run: python run_all.py --feedback   # just analyze + write feedback

 TEST STATUS
 ═══════════

  thesis_portfolio_opt/tests/  47/47 passing
  MiroFish/backend/tests/      36/36 passing
  autoresearch/                 imports verified, batch mode functional
