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

    For every walk-forward window:

        1. Calibrate thresholds using Monte-Carlo.
        2. Select the best threshold combination by Sharpe ratio.
        3. Apply the selected thresholds to the following
           out-of-sample test period.
        4. Store calibration statistics.
        5. Store out-of-sample portfolio and index returns.

    Calibration statistics stored:

        - total_return
        - mu
        - sigma
        - sharpe
        - max_drawdown
        - calmar

    Progress is reported after every completed
    walk-forward window.
    """

    start_time = time.time()

    start_date = pd.Timestamp(start_date)

    data = data.sort_index()

    first_date = data.index.min()
    last_date = data.index.max()

    # =========================================================
    # FIRST TEST DATE
    # =========================================================

    test_start = (
        start_date +
        pd.DateOffset(years=window)
    )

    # =========================================================
    # DETERMINE NUMBER OF WINDOWS
    # =========================================================

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

            progress_info={
                "window": iteration,
                "total_windows": total_windows
            }
        )

        # =====================================================
        # NO VALID OPTIMIZATION RESULT
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

        # =====================================================
        # BEST THRESHOLDS
        # =====================================================

        buy_thr = optimization[
            "best_buy_thr"
        ]

        sell_thr = optimization[
            "best_sell_thr"
        ]

        # =====================================================
        # 2. OUT-OF-SAMPLE TEST
        # =====================================================

        # IMPORTANT:
        #
        # We start at test_start and allow the strategy to
        # continue beyond test_end so that open positions
        # can naturally close.
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
        # 3. OUT-OF-SAMPLE PORTFOLIO RETURN
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
                - 1
            )

        # =====================================================
        # 4. DOW RETURN
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
                - 1
            )

        else:

            index_return = np.nan

        # =====================================================
        # 5. STORE WALK-FORWARD RESULT
        # =====================================================

        results.append({

            # -------------------------------------------------
            # Configuration
            # -------------------------------------------------

            "max_num_components":
                max_num_components,

            # -------------------------------------------------
            # Calibration period
            # -------------------------------------------------

            "calibration_start":
                calibration_start,

            "calibration_end":
                calibration_end,

            # -------------------------------------------------
            # Test period
            # -------------------------------------------------

            "test_start":
                test_start,

            "test_end":
                test_end,

            # -------------------------------------------------
            # Selected thresholds
            # -------------------------------------------------

            "best_buy_thr":
                buy_thr,

            "best_sell_thr":
                sell_thr,

            # -------------------------------------------------
            # Calibration statistics
            # -------------------------------------------------

            "calibration_total_return":
                optimization[
                    "total_return"
                ],

            "calibration_mu":
                optimization[
                    "mu"
                ],

            "calibration_sigma":
                optimization[
                    "sigma"
                ],

            "calibration_sharpe":
                optimization[
                    "sharpe"
                ],

            "calibration_max_drawdown":
                optimization[
                    "max_drawdown"
                ],

            "calibration_calmar":
                optimization[
                    "calmar"
                ],

            # -------------------------------------------------
            # Keep old mu_sigma name for compatibility
            # -------------------------------------------------

            "mu_sigma":
                optimization[
                    "sharpe"
                ],

            # -------------------------------------------------
            # Out-of-sample results
            # -------------------------------------------------

            "portfolio_return":
                test_return,

            "index_return":
                index_return
        })

        # =====================================================
        # 6. PROGRESS
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
            f"Sharpe={optimization['sharpe']:.3f} | "
            f"Calmar={optimization['calmar']:.3f} | "
            f"test_ret={test_return:+.2%} | "
            f"window_time={window_elapsed:.1f}s | "
            f"ETA={eta_minutes:.1f}min",
            flush=True
        )

        # =====================================================
        # 7. MOVE WALK-FORWARD WINDOW
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