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

from src.walk_forward import (
    run_single_walk_forward
)


# ============================================================
# WORKER
# ============================================================

def run_worker(max_num_components):

    print(
        f"[START] "
        f"max_num_components={max_num_components}",
        flush=True
    )

    # --------------------------------------------------------
    # Load data inside worker
    # --------------------------------------------------------

    data = pd.read_parquet(DATA_FILE)

    membership = pd.read_csv(
        MEMBERSHIP_FILE,
        parse_dates=["from", "to"]
    )

    components = (
        membership["ticker"]
        .drop_duplicates()
        .tolist()
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Run walk-forward optimization
    # --------------------------------------------------------

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
        time.time() - start_time
    )

    print(
        f"[DONE ] "
        f"max_num_components={max_num_components} "
        f"windows={len(result)} "
        f"time={elapsed / 60:.2f} min",
        flush=True
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(" WALK-FORWARD MONTE-CARLO OPTIMIZATION")
    print("=" * 75)

    print(
        f"Strategy             : {STRATEGY}"
    )

    print(
        f"Components           : "
        f"{MIN_NUM_COMPONENTS} -> "
        f"{MAX_NUM_COMPONENTS}"
    )

    print(
        f"Start date           : {START_DATE}"
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

    print("=" * 75)
    print()

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Data file not found:\n{DATA_FILE}"
        )

    if not MEMBERSHIP_FILE.exists():

        raise FileNotFoundError(
            f"Membership file not found:\n"
            f"{MEMBERSHIP_FILE}"
        )

    # --------------------------------------------------------
    # Load once to show information
    # --------------------------------------------------------

    data = pd.read_parquet(
        DATA_FILE
    )

    membership = pd.read_csv(
        MEMBERSHIP_FILE
    )

    components = (
        membership["ticker"]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Data rows           : {len(data):,}"
    )

    print(
        f"Data columns        : {len(data.columns):,}"
    )

    print(
        f"Components           : {len(components)}"
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
        f"Total parallel jobs  : "
        f"{total_jobs}"
    )

    print()

    overall_start = time.time()

    all_results = []

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

        completed = 0

        for future in as_completed(
            futures
        ):

            n = futures[future]

            try:

                result = future.result()

                if result is not None:
                    all_results.append(
                        result
                    )

                completed += 1

                elapsed = (
                    time.time() -
                    overall_start
                )

                print(
                    f"[PROGRESS] "
                    f"{completed}/{total_jobs} "
                    f"({100 * completed / total_jobs:.1f}%) "
                    f"| max_components={n} "
                    f"| elapsed={elapsed / 60:.1f} min",
                    flush=True
                )

            except Exception as error:

                completed += 1

                print(
                    f"[ERROR] "
                    f"max_num_components={n} "
                    f"| {type(error).__name__}: "
                    f"{error}",
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
    # Combine
    # --------------------------------------------------------

    results = pd.concat(
        all_results,
        ignore_index=True
    )

    results = (
        results
        .sort_values(
            [
                "max_num_components",
                "test_start"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    total_time = (
        time.time() -
        overall_start
    )

    print()
    print("=" * 75)
    print(" OPTIMIZATION FINISHED")
    print("=" * 75)

    print(
        f"Result rows         : "
        f"{len(results):,}"
    )

    print(
        f"Output              : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Total runtime       : "
        f"{total_time / 60:.2f} minutes"
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()