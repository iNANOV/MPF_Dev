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
        2. Identify the single best Monte-Carlo combination.
        3. Identify the robust thresholds from the top 10%.
        4. Test BOTH approaches out-of-sample.
        5. Store calibration statistics.
        6. Store top-10% threshold stability statistics.
        7. Store out-of-sample performance.

    BEST:
        Single Monte-Carlo simulation with the highest Sharpe.

    ROBUST:
        Median buy/sell thresholds among the top 10%
        of Monte-Carlo simulations ranked by Sharpe.

    The following top-10% threshold stability statistics
    are stored:

        top10_buy_std
        top10_sell_std
        top10_buy_min
        top10_buy_max
        top10_sell_min
        top10_sell_max
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
        # 2. EXTRACT BEST PARAMETERS
        # =====================================================

        best_buy_thr = optimization[
            "best_buy_thr"
        ]

        best_sell_thr = optimization[
            "best_sell_thr"
        ]

        # =====================================================
        # 3. EXTRACT ROBUST TOP-10% PARAMETERS
        # =====================================================

        robust_buy_thr = optimization[
            "robust_buy_thr"
        ]

        robust_sell_thr = optimization[
            "robust_sell_thr"
        ]

        # =====================================================
        # 4. PREPARE TEST DATA
        # =====================================================

        # Start at test_start and allow the strategy to continue
        # beyond test_end so that open positions can naturally
        # close.

        test_data = data.loc[
            test_start:
        ].copy()

        # =====================================================
        # 5. TEST SINGLE BEST PARAMETERS
        # =====================================================

        best_result = run_strategy(

            data=test_data,

            components=components,

            ma_column=ma_column,

            buy_thr=best_buy_thr,

            sell_thr=best_sell_thr,

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

        best_portfolio = best_result[
            "portfolio"
        ]

        # =====================================================
        # 6. TEST ROBUST TOP-10% PARAMETERS
        # =====================================================

        robust_result = run_strategy(

            data=test_data,

            components=components,

            ma_column=ma_column,

            buy_thr=robust_buy_thr,

            sell_thr=robust_sell_thr,

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

        robust_portfolio = robust_result[
            "portfolio"
        ]

        # =====================================================
        # 7. EVALUATE BEST OUT-OF-SAMPLE PERFORMANCE
        # =====================================================

        best_evaluation = best_portfolio.loc[
            test_start:test_end
        ]

        if best_evaluation.empty:

            best_test_return = np.nan

        else:

            best_test_return = (
                best_evaluation[
                    "portfolio_value"
                ].iloc[-1]
                /
                best_evaluation[
                    "portfolio_value"
                ].iloc[0]
                - 1
            )

        # =====================================================
        # 8. EVALUATE ROBUST OUT-OF-SAMPLE PERFORMANCE
        # =====================================================

        robust_evaluation = robust_portfolio.loc[
            test_start:test_end
        ]

        if robust_evaluation.empty:

            robust_test_return = np.nan

        else:

            robust_test_return = (
                robust_evaluation[
                    "portfolio_value"
                ].iloc[-1]
                /
                robust_evaluation[
                    "portfolio_value"
                ].iloc[0]
                - 1
            )

        # =====================================================
        # 9. INDEX RETURN
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
        # 10. EXCESS RETURNS
        # =====================================================

        if (
            not np.isnan(best_test_return)
            and
            not np.isnan(index_return)
        ):

            best_excess_return = (
                best_test_return -
                index_return
            )

        else:

            best_excess_return = np.nan

        if (
            not np.isnan(robust_test_return)
            and
            not np.isnan(index_return)
        ):

            robust_excess_return = (
                robust_test_return -
                index_return
            )

        else:

            robust_excess_return = np.nan

        # =====================================================
        # 11. STORE WALK-FORWARD RESULT
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

            # =================================================
            # BEST PARAMETERS
            # =================================================

            "best_buy_thr":
                best_buy_thr,

            "best_sell_thr":
                best_sell_thr,

            # -------------------------------------------------
            # Best calibration statistics
            # -------------------------------------------------

            "best_calibration_total_return":
                optimization[
                    "best_total_return"
                ],

            "best_calibration_mu":
                optimization[
                    "best_mu"
                ],

            "best_calibration_sigma":
                optimization[
                    "best_sigma"
                ],

            "best_calibration_sharpe":
                optimization[
                    "best_sharpe"
                ],

            "best_calibration_max_drawdown":
                optimization[
                    "best_max_drawdown"
                ],

            "best_calibration_calmar":
                optimization[
                    "best_calmar"
                ],

            # -------------------------------------------------
            # Best out-of-sample return
            # -------------------------------------------------

            "best_portfolio_return":
                best_test_return,

            # =================================================
            # ROBUST TOP-10% PARAMETERS
            # =================================================

            "robust_buy_thr":
                robust_buy_thr,

            "robust_sell_thr":
                robust_sell_thr,

            # -------------------------------------------------
            # Top-10% size
            # -------------------------------------------------

            "top10_n":
                optimization[
                    "top10_n"
                ],

            # -------------------------------------------------
            # Top-10% performance
            # -------------------------------------------------

            "top10_mean_sharpe":
                optimization[
                    "top10_mean_sharpe"
                ],

            "top10_median_sharpe":
                optimization[
                    "top10_median_sharpe"
                ],

            "top10_mean_return":
                optimization[
                    "top10_mean_return"
                ],

            "top10_median_return":
                optimization[
                    "top10_median_return"
                ],

            "top10_mean_drawdown":
                optimization[
                    "top10_mean_drawdown"
                ],

            "top10_median_drawdown":
                optimization[
                    "top10_median_drawdown"
                ],

            "top10_mean_calmar":
                optimization[
                    "top10_mean_calmar"
                ],

            "top10_median_calmar":
                optimization[
                    "top10_median_calmar"
                ],

            # =================================================
            # NEW: TOP-10% THRESHOLD STABILITY
            # =================================================

            "top10_buy_std":
                optimization[
                    "top10_buy_std"
                ],

            "top10_sell_std":
                optimization[
                    "top10_sell_std"
                ],

            "top10_buy_min":
                optimization[
                    "top10_buy_min"
                ],

            "top10_buy_max":
                optimization[
                    "top10_buy_max"
                ],

            "top10_sell_min":
                optimization[
                    "top10_sell_min"
                ],

            "top10_sell_max":
                optimization[
                    "top10_sell_max"
                ],

            # -------------------------------------------------
            # Robust out-of-sample return
            # -------------------------------------------------

            "robust_portfolio_return":
                robust_test_return,

            # =================================================
            # INDEX
            # =================================================

            "index_return":
                index_return,

            # =================================================
            # DIFFERENCE VS INDEX
            # =================================================

            "best_excess_return":
                best_excess_return,

            "robust_excess_return":
                robust_excess_return,

            # -------------------------------------------------
            # Compatibility
            # -------------------------------------------------

            "mu_sigma":
                optimization[
                    "sharpe"
                ]
        })

        # =====================================================
        # 12. PROGRESS
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
            f"BEST="
            f"({best_buy_thr:+.2f},"
            f"{best_sell_thr:+.2f}) | "
            f"ROBUST="
            f"({robust_buy_thr:+.2f},"
            f"{robust_sell_thr:+.2f}) | "
            f"Sharpe={optimization['sharpe']:.3f} | "
            f"Calmar={optimization['calmar']:.3f} | "
            f"Best OOS={best_test_return:+.2%} | "
            f"Robust OOS={robust_test_return:+.2%} | "
            f"Index={index_return:+.2%} | "
            f"time={window_elapsed:.1f}s | "
            f"ETA={eta_minutes:.1f}min",
            flush=True
        )

        # =====================================================
        # 13. MOVE WALK-FORWARD WINDOW
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