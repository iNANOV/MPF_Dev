# ============================================================
# MPF — EVALUATE TRAINING RESULTS
# ============================================================

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# PYTHON PATH
# ============================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation import rank_mpf_results


# ============================================================
# OUTPUT SUFFIXES
# ============================================================

CLASSICAL_SUFFIX = "_uniq_classical_ranking.csv"
IDR_SUFFIX = "_uniq_idr_ranking.csv"
FINAL_SUFFIX = "_uniq_final_selection.csv"


# ============================================================
# INPUT FILE DISCOVERY
# ============================================================

def find_result_files():
    """
    Find completed MPF optimization CSV files.

    JSON files without a corresponding CSV are ignored.

    Evaluation output files containing the '_uniq_' suffix
    are excluded from the input files.
    """

    csv_files = sorted(
        path
        for path in RESULTS_DIR.glob("*.csv")
        if "_uniq_" not in path.name
    )

    completed_files = []

    for csv_file in csv_files:

        # ----------------------------------------------------
        # Ignore empty CSV files
        # ----------------------------------------------------

        if csv_file.stat().st_size == 0:
            print(
                f"Skipping empty CSV: {csv_file.name}"
            )
            continue

        completed_files.append(csv_file)

    return completed_files


# ============================================================
# OUTPUT PATHS
# ============================================================

def get_output_paths(csv_file):
    """
    Return the three evaluation-output paths associated
    with one optimization result CSV.
    """

    stem = csv_file.stem

    return {
        "classical": (
            RESULTS_DIR
            / f"{stem}{CLASSICAL_SUFFIX}"
        ),
        "idr": (
            RESULTS_DIR
            / f"{stem}{IDR_SUFFIX}"
        ),
        "final": (
            RESULTS_DIR
            / f"{stem}{FINAL_SUFFIX}"
        ),
    }


# ============================================================
# CHECK WHETHER EVALUATION IS COMPLETE
# ============================================================

def evaluation_is_complete(output_paths):
    """
    Check whether all three evaluation files already exist.
    """

    return all(
        path.exists()
        for path in output_paths.values()
    )


# ============================================================
# EVALUATE ONE RESULT FILE
# ============================================================

def evaluate_result_file(csv_file):
    """
    Evaluate one completed MPF optimization result CSV.

    Only missing evaluation outputs are written.
    """

    output_paths = get_output_paths(
        csv_file
    )

    print("\n" + "=" * 80)
    print(
        f"INPUT: {csv_file.name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Check existing outputs
    # --------------------------------------------------------

    if evaluation_is_complete(
        output_paths
    ):

        print(
            "\nAll three evaluation files already exist."
        )

        for path in output_paths.values():
            print(
                f"  {path.name}"
            )

        print(
            "\nSkipping."
        )

        return False

    # --------------------------------------------------------
    # Load result CSV
    # --------------------------------------------------------

    print(
        "\nLoading result CSV..."
    )

    try:
        df = pd.read_csv(
            csv_file
        )
    except Exception as exc:

        print(
            f"\nWARNING: Could not read "
            f"{csv_file.name}: {exc}"
        )

        return False

    if df.empty:

        print(
            "\nWARNING: CSV is empty. Skipping."
        )

        return False

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\nCalculating rankings..."
    )

    try:

        (
            classical_ranking,
            idr_ranking,
            final_selection,
        ) = rank_mpf_results(df)

    except Exception as exc:

        print(
            f"\nWARNING: Evaluation failed for "
            f"{csv_file.name}: {exc}"
        )

        return False

    # --------------------------------------------------------
    # Save Classical ranking
    # --------------------------------------------------------

    if not output_paths["classical"].exists():

        classical_ranking.to_csv(
            output_paths["classical"],
            index=False,
        )

        print(
            "\nWritten:"
            f"\n  {output_paths['classical'].name}"
        )

    else:

        print(
            "\nAlready exists:"
            f"\n  {output_paths['classical'].name}"
        )

    # --------------------------------------------------------
    # Save IDR ranking
    # --------------------------------------------------------

    if not output_paths["idr"].exists():

        idr_ranking.to_csv(
            output_paths["idr"],
            index=False,
        )

        print(
            "\nWritten:"
            f"\n  {output_paths['idr'].name}"
        )

    else:

        print(
            "\nAlready exists:"
            f"\n  {output_paths['idr'].name}"
        )

    # --------------------------------------------------------
    # Save final selection
    # --------------------------------------------------------

    if not output_paths["final"].exists():

        final_selection.to_csv(
            output_paths["final"],
            index=False,
        )

        print(
            "\nWritten:"
            f"\n  {output_paths['final'].name}"
        )

    else:

        print(
            "\nAlready exists:"
            f"\n  {output_paths['final'].name}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nEvaluation complete."
    )

    print(
        f"  Classical candidates: "
        f"{len(classical_ranking):,}"
    )

    print(
        f"  IDR candidates: "
        f"{len(idr_ranking):,}"
    )

    print(
        f"  Final configurations: "
        f"{len(final_selection):,}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MPF — TRAINING RESULTS EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Check results directory
    # --------------------------------------------------------

    if not RESULTS_DIR.exists():

        raise FileNotFoundError(
            f"Results directory does not exist:\n"
            f"{RESULTS_DIR}"
        )

    # --------------------------------------------------------
    # Find completed CSV files
    # --------------------------------------------------------

    csv_files = find_result_files()

    print(
        f"\nFound {len(csv_files):,} "
        f"completed result CSV file(s)."
    )

    if not csv_files:

        print(
            "\nNo completed CSV files found."
        )

        return

    # --------------------------------------------------------
    # Process every completed result independently
    # --------------------------------------------------------

    processed = 0
    skipped = 0

    for csv_file in csv_files:

        output_paths = get_output_paths(
            csv_file
        )

        if evaluation_is_complete(
            output_paths
        ):
            skipped += 1
            continue

        if evaluate_result_file(
            csv_file
        ):
            processed += 1

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nCompleted CSV files found: "
        f"{len(csv_files):,}"
    )

    print(
        f"Files newly evaluated: "
        f"{processed:,}"
    )

    print(
        f"Files already evaluated: "
        f"{skipped:,}"
    )

    print(
        "\nEvaluation finished."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

