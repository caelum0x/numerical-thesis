"""
Master runner — integrates all three repos for the thesis.

Usage:
    python run_all.py                 # run everything
    python run_all.py --autoresearch  # run autoresearch batch
    python run_all.py --mirofish      # run MiroFish simulation
    python run_all.py --pipeline      # run thesis pipeline only
    python run_all.py --figures       # generate all figures
"""

import os
import sys
import time
import argparse
import subprocess
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUTORESEARCH_DIR = os.path.join(PROJECT_ROOT, '..', 'autoresearch')
MIROFISH_DIR = os.path.join(PROJECT_ROOT, '..', 'MiroFish')
VENV_PYTHON = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')

sys.path.insert(0, PROJECT_ROOT)


def run_command(cmd: str, cwd: str | None = None, description: str = "") -> subprocess.CompletedProcess[str]:
    """Run a shell command and return output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    if result.returncode == 0:
        # Print last 30 lines
        lines = result.stdout.strip().split('\n')
        for line in lines[-30:]:
            print(f"  {line}")
        print(f"  [{elapsed:.1f}s]")
    else:
        print(f"  ERROR (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "no stderr")
    return result


def run_thesis_pipeline() -> None:
    """Run the main thesis data/model/backtest pipeline."""
    from src.config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR

    # Check if data exists
    prices_path = RAW_DIR / 'prices.csv'
    if not prices_path.exists():
        print("Data not found. Fetching...")
        run_command(f"{VENV_PYTHON} run_pipeline.py --step fetch", cwd=PROJECT_ROOT,
                    description="Fetching price and macro data")

    features_path = PROCESSED_DIR / 'features.csv'
    if not features_path.exists():
        print("Features not found. Building...")
        run_command(f"{VENV_PYTHON} run_pipeline.py --step features", cwd=PROJECT_ROOT,
                    description="Building feature matrix")

    print("\nThesis pipeline data ready.")


def run_autoresearch() -> None:
    """Run autoresearch batch experiments."""
    run_command(
        f"{VENV_PYTHON} train.py --batch",
        cwd=AUTORESEARCH_DIR,
        description="AutoResearch: Running batch experiments (12 strategies)"
    )


def run_mirofish() -> None:
    """Run MiroFish financial market simulation."""
    run_command(
        f"{VENV_PYTHON} backend/app/services/financial_simulator.py",
        cwd=MIROFISH_DIR,
        description="MiroFish: Running multi-agent market simulation"
    )


def generate_figures() -> None:
    """Generate all thesis figures from latest results."""
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.config import RAW_DIR, RESULTS_DIR, STRESS_SCENARIOS, TICKERS

    sns.set_theme(style='whitegrid', palette='deep')

    prices = pd.read_csv(RAW_DIR / 'prices.csv', index_col=0, parse_dates=True).ffill().bfill()

    # Load autoresearch batch results if available
    batch_path = os.path.join(AUTORESEARCH_DIR, 'batch_results.csv')
    if os.path.exists(batch_path):
        batch = pd.read_csv(batch_path)

        fig, ax = plt.subplots(figsize=(14, 6))
        batch_sorted = batch.sort_values('sharpe', ascending=True)
        colors = ['green' if s > 0.671 else 'steelblue' if s > 0.202 else 'coral'
                  for s in batch_sorted['sharpe']]
        ax.barh(batch_sorted['experiment'], batch_sorted['sharpe'], color=colors)
        ax.axvline(x=0.202, color='gray', linestyle='--', linewidth=1, label='Equal Weight')
        ax.axvline(x=0.671, color='black', linestyle='--', linewidth=1, label='SPY')
        ax.set_title('AutoResearch Batch Results — OOS Sharpe (2022-2024)', fontsize=14)
        ax.legend()
        fig.savefig(RESULTS_DIR / 'fig15_autoresearch_batch.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  fig15_autoresearch_batch.pdf")

    # Load MiroFish simulation results
    sim_path = RESULTS_DIR / 'mirofish_simulation.json'
    if sim_path.exists():
        import json
        with open(sim_path) as f:
            sim = json.load(f)

        agreements = [r['agent_agreement'] for r in sim['rounds']]
        dates = [pd.Timestamp(r['date']) for r in sim['rounds']]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dates, agreements, 'o-', color='steelblue', markersize=4, linewidth=1)
        ax.axhline(y=np.mean(agreements), color='red', linestyle='--', label=f'Mean: {np.mean(agreements):.3f}')
        ax.fill_between(dates, agreements, alpha=0.2, color='steelblue')
        ax.set_title('MiroFish Agent Agreement Over Time (2022-2024)', fontsize=14)
        ax.set_ylabel('Agent Agreement')
        ax.set_xlabel('Date')
        ax.legend()
        fig.savefig(RESULTS_DIR / 'fig16_mirofish_agreement.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  fig16_mirofish_agreement.pdf")

    print("\nAll integrated figures generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Master thesis runner')
    parser.add_argument('--autoresearch', action='store_true')
    parser.add_argument('--mirofish', action='store_true')
    parser.add_argument('--pipeline', action='store_true')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()

    run_specific = args.autoresearch or args.mirofish or args.pipeline or args.figures

    if not run_specific or args.pipeline:
        run_thesis_pipeline()

    if not run_specific or args.autoresearch:
        run_autoresearch()

    if not run_specific or args.mirofish:
        run_mirofish()

    if not run_specific or args.figures:
        generate_figures()

    # Summary
    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)

    from src.config import RESULTS_DIR
    pdfs = list(RESULTS_DIR.glob('fig*.pdf'))
    csvs = list(RESULTS_DIR.glob('*.csv'))
    jsons = list(RESULTS_DIR.glob('*.json'))
    pkls = list(RESULTS_DIR.glob('*.pkl'))
    print(f"  Figures: {len(pdfs)} PDFs")
    print(f"  Data: {len(csvs)} CSVs, {len(jsons)} JSONs")
    print(f"  Models: {len(pkls)} trained")
    print(f"\n  Results at: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
