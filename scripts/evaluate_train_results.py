# ============================================================
# MPF — EVALUATE TRAINING RESULTS
# ============================================================

from pathlib import Path
import argparse

import pandas as pd

from src.evaluation import rank_mpf_configurations


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate MPF training / hyperparameter-selection results."
    )

    parser.add_argument(
        "results_file",
        type=str,
        help="CSV result file to evaluate.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve input file
    # --------------------------------------------------------

    results_path = Path(args.results_file)

    # If only a filename was supplied, look in results/
    if not results_path.is_absolute():

        if not results_path.exists():
            results_path = RESULTS_DIR / results_path

    if not results_path.exists():

        raise FileNotFoundError(
            f"\nResults file not found:\n"
            f"{results_path}\n"
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    print("=" * 80)
    print("MPF — TRAINING RESULTS EVALUATION")
    print("=" * 80)

    print(f"\nInput file:")
    print(results_path)

    df = pd.read_csv(results_path)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # --------------------------------------------------------
    # Rank configurations
    # --------------------------------------------------------

    metrics_df, ranking_df = rank_mpf_configurations(df)

    # --------------------------------------------------------
    # Print overall ranking
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MPF — OVERALL RANKING")
    print("=" * 80)

    display_table = ranking_df.copy()

    display_table.columns = [
        "Overall Rank",
        "Components",
        "Method",
        "Average Rank",
        "Rank Positive Windows",
        "Rank Mean Excess",
        "Rank Compound Excess",
        "Rank Sharpe",
    ]

    display_table["Overall Rank"] = (
        display_table["Overall Rank"]
        .astype(int)
    )

    print(
        display_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Top configurations
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)

    print(
        display_table
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    stem = results_path.stem

    ranking_output = (
        RESULTS_DIR
        / f"{stem}_ranking.csv"
    )

    metrics_output = (
        RESULTS_DIR
        / f"{stem}_metrics.csv"
    )

    display_table.to_csv(
        ranking_output,
        index=False,
    )

    metrics_df.to_csv(
        metrics_output,
        index=False,
    )

    print("\n" + "=" * 80)
    print("EVALUATION FILES WRITTEN")
    print("=" * 80)

    print(f"\nRanking:")
    print(ranking_output)

    print(f"\nMetrics:")
    print(metrics_output)

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = display_table.iloc[0]

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)

    print(
        f"\nOverall Rank : {best['Overall Rank']}"
        f"\nComponents   : {best['Components']}"
        f"\nMethod       : {best['Method']}"
        f"\nAverage Rank : {best['Average Rank']:.2f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()