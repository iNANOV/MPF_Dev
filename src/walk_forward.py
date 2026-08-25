import numpy as np
import pandas as pd
import time

from config import (
    N_SIMULATIONS_STAGE1,
    N_SIMULATIONS_STAGE2,
)

from .optimization import (
    monte_carlo_optimize,
    monte_carlo_stage2_optimize
)

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
    Two-stage walk-forward optimization.

    Stage 1
    -------
    Broad Monte-Carlo search using N_SIMULATIONS_STAGE1.

    Stage 2
    -------
    Focused Monte-Carlo search around the robust top-10%
    threshold region identified in Stage 1, using
    N_SIMULATIONS_STAGE2.

    Four optimization criteria are evaluated:

        1. Sharpe
        2. Calmar
        3. Return / Risk
        4. Robust top-10% Sharpe region

    Each Stage-2 criterion is evaluated on the
    following out-of-sample period.
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
    # NUMBER OF WINDOWS
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
        f"MC Stage1/window={N_SIMULATIONS_STAGE1} | "
        f"MC Stage2/window={N_SIMULATIONS_STAGE2}",
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

        print(
            f"\n[WF WINDOW] "
            f"components={max_num_components} | "
            f"window={iteration}/{total_windows} | "
            f"calibration={calibration_start.date()} -> "
            f"{calibration_end.date()} | "
            f"test={test_start.date()} -> "
            f"{test_end.date()}",
            flush=True
        )

        # =====================================================
        # STAGE 1
        # =====================================================

        print(
            f"[STAGE1] "
            f"components={max_num_components} | "
            f"window={iteration}/{total_windows} | "
            f"simulations={N_SIMULATIONS_STAGE1}",
            flush=True
        )

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

            n_simulations=N_SIMULATIONS_STAGE1,

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
        # NO STAGE-1 RESULT
        # =====================================================

        if optimization is None:

            print(
                f"[WF] "
                f"components={max_num_components} | "
                f"window={iteration}/{total_windows} | "
                f"NO STAGE1 RESULT",
                flush=True
            )

            test_start += pd.DateOffset(
                months=moving_param
            )

            continue

        # =====================================================
        # STAGE 1 ROBUST REGION
        # =====================================================

        robust_buy_thr_stage1 = optimization[
            "robust_buy_thr"
        ]

        robust_sell_thr_stage1 = optimization[
            "robust_sell_thr"
        ]

        buy_std_stage1 = optimization.get(
            "top10_buy_std",
            np.nan
        )

        sell_std_stage1 = optimization.get(
            "top10_sell_std",
            np.nan
        )

        # -----------------------------------------------------
        # Fallback if standard deviation is unavailable
        # -----------------------------------------------------

        if not np.isfinite(buy_std_stage1):

            buy_std_stage1 = threshold_step

        if not np.isfinite(sell_std_stage1):

            sell_std_stage1 = threshold_step

        # =====================================================
        # STAGE 2
        # =====================================================

        print(
            f"[STAGE2] "
            f"components={max_num_components} | "
            f"window={iteration}/{total_windows} | "
            f"simulations={N_SIMULATIONS_STAGE2} | "
            f"center_buy={robust_buy_thr_stage1:+.3f} | "
            f"center_sell={robust_sell_thr_stage1:+.3f} | "
            f"buy_std={buy_std_stage1:.3f} | "
            f"sell_std={sell_std_stage1:.3f}",
            flush=True
        )

        stage2 = monte_carlo_stage2_optimize(

            data=data,

            components=components,

            membership=membership,

            calibration_start=calibration_start,

            calibration_end=calibration_end,

            max_num_components=max_num_components,

            strategy=strategy,

            ma_column=ma_column,

            ranking_column=ranking_column,

            center_buy_thr=robust_buy_thr_stage1,

            center_sell_thr=robust_sell_thr_stage1,

            buy_std=buy_std_stage1,

            sell_std=sell_std_stage1,

            n_simulations=N_SIMULATIONS_STAGE2,

            threshold_step=threshold_step,

            random_state=(
                random_state +
                10000 +
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
            )
        )

        # =====================================================
        # NO STAGE-2 RESULT
        # =====================================================

        if stage2 is None:

            print(
                f"[WF] "
                f"components={max_num_components} | "
                f"window={iteration}/{total_windows} | "
                f"NO STAGE2 RESULT",
                flush=True
            )

            test_start += pd.DateOffset(
                months=moving_param
            )

            continue

        # =====================================================
        # STAGE-2 THRESHOLDS
        # =====================================================

        sharpe_buy_thr = stage2[
            "sharpe_buy_thr"
        ]

        sharpe_sell_thr = stage2[
            "sharpe_sell_thr"
        ]

        calmar_buy_thr = stage2[
            "calmar_buy_thr"
        ]

        calmar_sell_thr = stage2[
            "calmar_sell_thr"
        ]

        return_risk_buy_thr = stage2[
            "return_risk_buy_thr"
        ]

        return_risk_sell_thr = stage2[
            "return_risk_sell_thr"
        ]

        robust_buy_thr = stage2[
            "robust_buy_thr"
        ]

        robust_sell_thr = stage2[
            "robust_sell_thr"
        ]

        # =====================================================
        # OOS HELPER
        # =====================================================

        def run_oos(
            buy_thr,
            sell_thr
        ):

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

            portfolio = result["portfolio"]

            evaluation = portfolio.loc[
                test_start:test_end
            ]

            if evaluation.empty:

                return np.nan

            values = evaluation[
                "portfolio_value"
            ].dropna()

            if len(values) < 2:

                return np.nan

            return (
                values.iloc[-1] /
                values.iloc[0]
                - 1
            )

        # =====================================================
        # FOUR OOS TESTS
        # =====================================================

        sharpe_oos_return = run_oos(
            sharpe_buy_thr,
            sharpe_sell_thr
        )

        calmar_oos_return = run_oos(
            calmar_buy_thr,
            calmar_sell_thr
        )

        return_risk_oos_return = run_oos(
            return_risk_buy_thr,
            return_risk_sell_thr
        )

        robust_oos_return = run_oos(
            robust_buy_thr,
            robust_sell_thr
        )

        # =====================================================
        # DOW RETURN
        # =====================================================

        index_col = "INDEX"

        index_prices = data.loc[
            test_start:test_end,
            index_col
        ].dropna()

        if len(index_prices) >= 2:

            index_return = (
                index_prices.iloc[-1] /
                index_prices.iloc[0]
                - 1
            )

        else:

            index_return = np.nan

        # =====================================================
        # EXCESS RETURNS
        # =====================================================

        sharpe_excess = (
            sharpe_oos_return -
            index_return
        )

        calmar_excess = (
            calmar_oos_return -
            index_return
        )

        return_risk_excess = (
            return_risk_oos_return -
            index_return
        )

        robust_excess = (
            robust_oos_return -
            index_return
        )

        # =====================================================
        # STORE RESULT
        # =====================================================

        row = {

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

            # -------------------------------------------------
            # Stage 1 robust region
            # -------------------------------------------------

            "stage1_robust_buy_thr":
                robust_buy_thr_stage1,

            "stage1_robust_sell_thr":
                robust_sell_thr_stage1,

            "stage1_top10_buy_std":
                buy_std_stage1,

            "stage1_top10_sell_std":
                sell_std_stage1,

            # -------------------------------------------------
            # Stage 2 search region
            # -------------------------------------------------

            "stage2_buy_min":
                stage2.get(
                    "stage2_buy_min",
                    np.nan
                ),

            "stage2_buy_max":
                stage2.get(
                    "stage2_buy_max",
                    np.nan
                ),

            "stage2_sell_min":
                stage2.get(
                    "stage2_sell_min",
                    np.nan
                ),

            "stage2_sell_max":
                stage2.get(
                    "stage2_sell_max",
                    np.nan
                ),

            # -------------------------------------------------
            # Sharpe
            # -------------------------------------------------

            "sharpe_buy_thr":
                sharpe_buy_thr,

            "sharpe_sell_thr":
                sharpe_sell_thr,

            "sharpe_calibration":
                stage2.get(
                    "sharpe",
                    np.nan
                ),

            "sharpe_calibration_return":
                stage2.get(
                    "sharpe_return",
                    np.nan
                ),

            "sharpe_oos_return":
                sharpe_oos_return,

            "sharpe_excess_return":
                sharpe_excess,

            # -------------------------------------------------
            # Calmar
            # -------------------------------------------------

            "calmar_buy_thr":
                calmar_buy_thr,

            "calmar_sell_thr":
                calmar_sell_thr,

            "calmar_calibration":
                stage2.get(
                    "calmar",
                    np.nan
                ),

            "calmar_calibration_return":
                stage2.get(
                    "calmar_return",
                    np.nan
                ),

            "calmar_oos_return":
                calmar_oos_return,

            "calmar_excess_return":
                calmar_excess,

            # -------------------------------------------------
            # Return / Risk
            # -------------------------------------------------

            "return_risk_buy_thr":
                return_risk_buy_thr,

            "return_risk_sell_thr":
                return_risk_sell_thr,

            "return_risk_calibration":
                stage2.get(
                    "return_risk",
                    np.nan
                ),

            "return_risk_calibration_return":
                stage2.get(
                    "return_risk_return",
                    np.nan
                ),

            "return_risk_oos_return":
                return_risk_oos_return,

            "return_risk_excess_return":
                return_risk_excess,

            # -------------------------------------------------
            # Robust
            # -------------------------------------------------

            "robust_buy_thr":
                robust_buy_thr,

            "robust_sell_thr":
                robust_sell_thr,

            "robust_calibration_sharpe":
                stage2.get(
                    "robust_sharpe",
                    np.nan
                ),

            "robust_calibration_return":
                stage2.get(
                    "robust_return",
                    np.nan
                ),

            "robust_calibration_calmar":
                stage2.get(
                    "robust_calmar",
                    np.nan
                ),

            "robust_calibration_return_risk":
                stage2.get(
                    "robust_return_risk",
                    np.nan
                ),

            "robust_oos_return":
                robust_oos_return,

            "robust_excess_return":
                robust_excess,

            # -------------------------------------------------
            # Stage 2 distribution
            # -------------------------------------------------

            "stage2_n":
                stage2.get(
                    "stage2_n",
                    np.nan
                ),

            "stage2_buy_std":
                stage2.get(
                    "stage2_buy_std",
                    np.nan
                ),

            "stage2_sell_std":
                stage2.get(
                    "stage2_sell_std",
                    np.nan
                ),

            "stage2_actual_buy_min":
                stage2.get(
                    "stage2_buy_min",
                    np.nan
                ),

            "stage2_actual_buy_max":
                stage2.get(
                    "stage2_buy_max",
                    np.nan
                ),

            "stage2_actual_sell_min":
                stage2.get(
                    "stage2_sell_min",
                    np.nan
                ),

            "stage2_actual_sell_max":
                stage2.get(
                    "stage2_sell_max",
                    np.nan
                ),

            # -------------------------------------------------
            # Benchmark
            # -------------------------------------------------

            "index_return":
                index_return,

            # -------------------------------------------------
            # Backward compatibility
            # -------------------------------------------------

            "mu_sigma":
                stage2.get(
                    "sharpe",
                    np.nan
                ),
        }

        results.append(row)

        # =====================================================
        # PROGRESS
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
            f"S={sharpe_oos_return:+.2%} | "
            f"C={calmar_oos_return:+.2%} | "
            f"RR={return_risk_oos_return:+.2%} | "
            f"R={robust_oos_return:+.2%} | "
            f"DOW={index_return:+.2%} | "
            f"time={window_elapsed:.1f}s | "
            f"ETA={eta_minutes:.1f}min",
            flush=True
        )

        # =====================================================
        # NEXT WINDOW
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

    return pd.DataFrame(results)