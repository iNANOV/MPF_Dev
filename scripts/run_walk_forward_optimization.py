import sys
import time
import os

import pandas as pd

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import (
    DATA_FILE,
    MEMBERSHIP_FILE,
    OUTPUT_FILE,

    STRATEGY,
    MA_COLUMN,
    RANKING_COLUMN,

    START_DATE,
    CALIBRATION_WINDOW,
    MOVING_PARAM,
    TEST_SPAN,

    N_SIMULATIONS,
    THRESHOLD_STEP,
    RANDOM_STATE,

    MIN_NUM_COMPONENTS,
    MAX_NUM_COMPONENTS,
    INITIAL_CAPITAL,
    ABS_COST_FOR_A_TRADE,
    PERCENT_COST_FOR_A_TRADE,
    MAX_INVESTMENT_SIZE_IN_PERCENT,
    MIN_CASH_IN_PERCENT,

    MAX_WORKERS
)

from src.walk_forward import run_single_walk_forward


# ============================================================
# WORKER
# ============================================================

def run_worker(max_num_components):
    """
    Run the complete walk-forward optimization for one value
    of max_num_components.

    Each worker loads its own data so that large DataFrames do
    not have to be serialized and transferred between processes.
    """

    worker_start = time.time()

    print(
        f"[START] max_num_components={max_num_components}",
        flush=True
    )

    try:

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        data = pd.read_parquet(
            DATA_FILE
        )

        membership = pd.read_csv(
            MEMBERSHIP_FILE,
            parse_dates=["from", "to"]
        )

        components = (
            membership["ticker"]
            .drop_duplicates()
            .tolist()
        )

        # ----------------------------------------------------
        # Run walk-forward optimization
        # ----------------------------------------------------

        result = run_single_walk_forward(

            data=data,

            components=components,

            membership=membership,

            strategy=STRATEGY,

            ma_column=MA_COLUMN,

            ranking_column=RANKING_COLUMN,

            max_num_components=max_num_components,

            start_date=START_DATE,

            window=CALIBRATION_WINDOW,

            moving_param=MOVING_PARAM,

            test_span=TEST_SPAN,

            n_simulations=N_SIMULATIONS,

            threshold_step=THRESHOLD_STEP,

            random_state=(
                RANDOM_STATE +
                max_num_components
            ),

            initial_capital=INITIAL_CAPITAL,

            abs_cost_for_a_trade=(
                ABS_COST_FOR_A_TRADE
            ),

            percent_cost_for_a_trade=(
                PERCENT_COST_FOR_A_TRADE
            ),

            max_investment_size_in_percent=(
                MAX_INVESTMENT_SIZE_IN_PERCENT
            ),

            min_cash_in_percent=(
                MIN_CASH_IN_PERCENT
            )
        )

        elapsed = (
            time.time() -
            worker_start
        )

        # ----------------------------------------------------
        # Check result
        # ----------------------------------------------------

        if result is None:

            print(
                f"[DONE ] max_num_components="
                f"{max_num_components} "
                f"| NO RESULT "
                f"| time={elapsed / 60:.2f} min",
                flush=True
            )

            return None

        try:

            n_windows = len(result)

        except TypeError:

            n_windows = 0

        print(
            f"[DONE ] max_num_components="
            f"{max_num_components} "
            f"| windows={n_windows} "
            f"| time={elapsed / 60:.2f} min",
            flush=True
        )

        return result

    except Exception as error:

        elapsed = (
            time.time() -
            worker_start
        )

        print(
            f"[ERROR] max_num_components="
            f"{max_num_components} "
            f"| {type(error).__name__}: {error} "
            f"| time={elapsed / 60:.2f} min",
            flush=True
        )

        raise


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        " WALK-FORWARD MONTE-CARLO OPTIMIZATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print(
        f"Strategy             : {STRATEGY}"
    )

    print(
        f"Components           : "
        f"{MIN_NUM_COMPONENTS} -> "
        f"{MAX_NUM_COMPONENTS}"
    )

    print(
        f"Start date           : "
        f"{START_DATE}"
    )

    print(
        f"Calibration window   : "
        f"{CALIBRATION_WINDOW} years"
    )

    print(
        f"Moving parameter     : "
        f"{MOVING_PARAM} months"
    )

    print(
        f"Test span            : "
        f"{TEST_SPAN} months"
    )

    print(
        f"MC simulations       : "
        f"{N_SIMULATIONS}"
    )

    print(
        f"Threshold step       : "
        f"{THRESHOLD_STEP}"
    )

    print(
        f"Initial capital      : "
        f"{INITIAL_CAPITAL:,.0f}"
    )

    print(
        f"Random state         : "
        f"{RANDOM_STATE}"
    )

    print(
        f"Max workers          : "
        f"{MAX_WORKERS or 'automatic'}"
    )

    print()

    print(
        f"Output file          : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Data file not found:\n"
            f"{DATA_FILE}"
        )

    if not MEMBERSHIP_FILE.exists():

        raise FileNotFoundError(
            f"Membership file not found:\n"
            f"{MEMBERSHIP_FILE}"
        )

    # --------------------------------------------------------
    # Load data once for diagnostics
    # --------------------------------------------------------

    data = pd.read_parquet(
        DATA_FILE
    )

    membership = pd.read_csv(
        MEMBERSHIP_FILE,
        parse_dates=["from", "to"]
    )

    components = (
        membership["ticker"]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Data rows           : "
        f"{len(data):,}"
    )

    print(
        f"Data columns        : "
        f"{len(data.columns):,}"
    )

    print(
        f"Components           : "
        f"{len(components)}"
    )

    print(
        f"Date range           : "
        f"{data.index.min().date()} "
        f"-> "
        f"{data.index.max().date()}"
    )

    print()

    # --------------------------------------------------------
    # Values of max_num_components
    # --------------------------------------------------------

    component_values = list(
        range(
            MIN_NUM_COMPONENTS,
            MAX_NUM_COMPONENTS + 1
        )
    )

    total_jobs = len(
        component_values
    )

    print(
        f"Total optimization jobs : "
        f"{total_jobs}"
    )

    print(
        f"Parallel workers        : "
        f"{MAX_WORKERS or 'automatic'}"
    )

    print()

    print("=" * 80)
    print(
        " STARTING PARALLEL OPTIMIZATION"
    )
    print("=" * 80)
    print()

    overall_start = time.time()

    all_results = []

    completed = 0

    # --------------------------------------------------------
    # Parallel execution
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                run_worker,
                n
            ): n

            for n in component_values
        }

        # ----------------------------------------------------
        # Process jobs as they finish
        # ----------------------------------------------------

        for future in as_completed(
            futures
        ):

            n = futures[
                future
            ]

            completed += 1

            try:

                result = future.result()

                if result is not None:

                    all_results.append(
                        result
                    )

                    try:

                        n_windows = len(
                            result
                        )

                    except TypeError:

                        n_windows = 0

                else:

                    n_windows = 0

                elapsed = (
                    time.time()
                    - overall_start
                )

                progress = (
                    100.0
                    * completed
                    / total_jobs
                )

                print()

                print(
                    f"[PROGRESS] "
                    f"{completed}/{total_jobs} "
                    f"({progress:.1f}%) "
                    f"| max_components={n} "
                    f"| windows={n_windows} "
                    f"| elapsed={elapsed / 60:.1f} min",
                    flush=True
                )

            except Exception as error:

                elapsed = (
                    time.time()
                    - overall_start
                )

                progress = (
                    100.0
                    * completed
                    / total_jobs
                )

                print()

                print(
                    f"[FAILED] "
                    f"{completed}/{total_jobs} "
                    f"({progress:.1f}%) "
                    f"| max_components={n} "
                    f"| {type(error).__name__}: "
                    f"{error} "
                    f"| elapsed={elapsed / 60:.1f} min",
                    flush=True
                )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No optimization results were produced."
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    print()
    print(
        "Combining optimization results..."
    )

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    sort_columns = []

    if "max_num_components" in results.columns:

        sort_columns.append(
            "max_num_components"
        )

    if "test_start" in results.columns:

        sort_columns.append(
            "test_start"
        )

    if sort_columns:

        results = (
            results
            .sort_values(
                sort_columns
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    total_time = (
        time.time()
        - overall_start
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print("=" * 80)
    print(
        " OPTIMIZATION FINISHED"
    )
    print("=" * 80)

    print(
        f"Successful jobs      : "
        f"{len(all_results)}/{total_jobs}"
    )

    print(
        f"Result rows          : "
        f"{len(results):,}"
    )

    print(
        f"Output               : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Total runtime        : "
        f"{total_time / 60:.2f} minutes"
    )

    print("=" * 80)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()