import numpy as np
import pandas as pd
import time

from .optimization import monte_carlo_optimize
from .strategy import run_strategy


def run_single_walk_forward(
    data,
    components,
    membership,
    strategy,
    ma_column,
    ranking_column,
    max_num_components,
    start_date,
    window,
    moving_param,
    test_span,
    n_simulations=500,
    threshold_step=0.01,
    random_state=123,
    initial_capital=100_000,
    abs_cost_for_a_trade=5,
    percent_cost_for_a_trade=0.001,
    max_investment_size_in_percent=50,
    min_cash_in_percent=10
):
    """
    Run walk-forward optimization for ONE max_num_components.

    Progress is reported:
        - during Monte-Carlo simulations
        - after every completed walk-forward window
        - with ETA information
    """

    start_time = time.time()

    start_date = pd.Timestamp(start_date)

    data = data.sort_index()

    first_date = data.index.min()
    last_date = data.index.max()

    # ---------------------------------------------------------
    # First test period
    # ---------------------------------------------------------

    test_start = (
        start_date +
        pd.DateOffset(years=window)
    )

    # ---------------------------------------------------------
    # Determine total number of walk-forward windows
    # ---------------------------------------------------------

    temp_date = test_start
    total_windows = 0

    while temp_date <= last_date:

        total_windows += 1

        temp_date += pd.DateOffset(
            months=moving_param
        )

    print(
        f"[WF START] "
        f"components={max_num_components} | "
        f"windows={total_windows} | "
        f"MC/window={n_simulations}",
        flush=True
    )

    results = []

    iteration = 0

    # =========================================================
    # WALK-FORWARD LOOP
    # =========================================================

    while test_start <= last_date:

        iteration += 1

        window_start_time = time.time()

        # -----------------------------------------------------
        # Calibration period
        # -----------------------------------------------------

        calibration_start = first_date

        calibration_end = (
            test_start -
            pd.Timedelta(days=1)
        )

        # -----------------------------------------------------
        # Test period
        # -----------------------------------------------------

        test_end = (
            test_start +
            pd.DateOffset(months=test_span)
        )

        if test_end > last_date:

            test_end = last_date

        # =====================================================
        # 1. MONTE-CARLO CALIBRATION
        # =====================================================

        optimization = monte_carlo_optimize(

            data=data,

            components=components,

            membership=membership,

            calibration_start=calibration_start,

            calibration_end=calibration_end,

            max_num_components=max_num_components,

            strategy=strategy,

            ma_column=ma_column,

            ranking_column=ranking_column,

            n_simulations=n_simulations,

            threshold_step=threshold_step,

            random_state=(
                random_state +
                iteration
            ),

            initial_capital=initial_capital,

            abs_cost_for_a_trade=(
                abs_cost_for_a_trade
            ),

            percent_cost_for_a_trade=(
                percent_cost_for_a_trade
            ),

            max_investment_size_in_percent=(
                max_investment_size_in_percent
            ),

            min_cash_in_percent=(
                min_cash_in_percent
            ),

            # -------------------------------------------------
            # IMPORTANT:
            # Tell optimization.py which WF window
            # is currently running.
            # -------------------------------------------------

            progress_info={
                "window": iteration,
                "total_windows": total_windows
            }
        )

        # =====================================================
        # NO OPTIMIZATION RESULT
        # =====================================================

        if optimization is None:

            window_elapsed = (
                time.time() -
                window_start_time
            )

            print(
                f"[WF] "
                f"components={max_num_components} | "
                f"window={iteration}/{total_windows} | "
                f"{100 * iteration / total_windows:5.1f}% | "
                f"test={test_start.date()} -> "
                f"{test_end.date()} | "
                f"NO RESULT | "
                f"time={window_elapsed:.1f}s",
                flush=True
            )

            test_start += pd.DateOffset(
                months=moving_param
            )

            continue

        # -----------------------------------------------------
        # Selected thresholds
        # -----------------------------------------------------

        buy_thr = optimization[
            "best_buy_thr"
        ]

        sell_thr = optimization[
            "best_sell_thr"
        ]

        # =====================================================
        # 2. APPLY SELECTED PARAMETERS TO TEST DATA
        # =====================================================

        # IMPORTANT:
        #
        # Use data beyond test_end so that trades that are
        # opened during the test period can finish naturally.
        #

        test_data = data.loc[
            test_start:
        ].copy()

        result = run_strategy(

            data=test_data,

            components=components,

            ma_column=ma_column,

            buy_thr=buy_thr,

            sell_thr=sell_thr,

            strategy=strategy,

            membership=membership,

            max_num_components=max_num_components,

            start_date=test_start,

            ranking_column=ranking_column,

            initial_capital=initial_capital,

            abs_cost_for_a_trade=(
                abs_cost_for_a_trade
            ),

            percent_cost_for_a_trade=(
                percent_cost_for_a_trade
            ),

            max_investment_size_in_percent=(
                max_investment_size_in_percent
            ),

            min_cash_in_percent=(
                min_cash_in_percent
            )
        )

        portfolio = result[
            "portfolio"
        ]

        # =====================================================
        # 3. TEST-PERIOD PERFORMANCE
        # =====================================================

        evaluation = portfolio.loc[
            test_start:test_end
        ]

        if evaluation.empty:

            test_return = np.nan

        else:

            test_return = (
                evaluation[
                    "portfolio_value"
                ].iloc[-1]
                /
                evaluation[
                    "portfolio_value"
                ].iloc[0]
                -
                1
            )

        # =====================================================
        # 4. DOW / INDEX RETURN
        # =====================================================

        index_col = "INDEX"

        index_prices = data.loc[
            test_start:test_end,
            index_col
        ].dropna()

        if len(index_prices) >= 2:

            index_return = (
                index_prices.iloc[-1]
                /
                index_prices.iloc[0]
                -
                1
            )

        else:

            index_return = np.nan

        # =====================================================
        # 5. STORE RESULT
        # =====================================================

        results.append({

            "max_num_components":
                max_num_components,

            "calibration_start":
                calibration_start,

            "calibration_end":
                calibration_end,

            "test_start":
                test_start,

            "test_end":
                test_end,

            "best_buy_thr":
                buy_thr,

            "best_sell_thr":
                sell_thr,

            "mu":
                optimization["mu"],

            "sigma":
                optimization["sigma"],

            "mu_sigma":
                optimization["mu_sigma"],

            "portfolio_return":
                test_return,

            "index_return":
                index_return
        })

        # =====================================================
        # 6. PROGRESS / ETA
        # =====================================================

        window_elapsed = (
            time.time() -
            window_start_time
        )

        total_elapsed = (
            time.time() -
            start_time
        )

        completed_fraction = (
            iteration /
            total_windows
        )

        # Estimate total runtime based on completed windows

        estimated_total = (
            total_elapsed /
            iteration
        ) * total_windows

        eta_seconds = (
            estimated_total -
            total_elapsed
        )

        eta_minutes = max(
            0,
            eta_seconds / 60
        )

        print(
            f"[WF] "
            f"components={max_num_components} | "
            f"window={iteration}/{total_windows} | "
            f"{100 * completed_fraction:5.1f}% | "
            f"test={test_start.date()} -> "
            f"{test_end.date()} | "
            f"buy={buy_thr:+.2f} | "
            f"sell={sell_thr:+.2f} | "
            f"Sharpe={optimization['mu_sigma']:.3f} | "
            f"test_ret={test_return:+.2%} | "
            f"window_time={window_elapsed:.1f}s | "
            f"ETA={eta_minutes:.1f}min",
            flush=True
        )

        # =====================================================
        # 7. MOVE TO NEXT WALK-FORWARD WINDOW
        # =====================================================

        test_start += pd.DateOffset(
            months=moving_param
        )

    # =========================================================
    # FINISHED
    # =========================================================

    total_time = (
        time.time() -
        start_time
    )

    print(
        f"[WF DONE] "
        f"components={max_num_components} | "
        f"windows={len(results)}/{total_windows} | "
        f"time={total_time / 60:.2f} min",
        flush=True
    )

    return pd.DataFrame(
        results
    )