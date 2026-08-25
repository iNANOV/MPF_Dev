import numpy as np
import pandas as pd
import time

from .optimization import (
    monte_carlo_optimize,
    calculate_portfolio_statistics
)

from .strategy import run_strategy


def _run_test_strategy(
    data,
    components,
    membership,
    strategy,
    ma_column,
    ranking_column,
    max_num_components,
    test_start,
    buy_thr,
    sell_thr,
    initial_capital,
    abs_cost_for_a_trade,
    percent_cost_for_a_trade,
    max_investment_size_in_percent,
    min_cash_in_percent
):
    """
    Run one out-of-sample strategy using a fixed
    threshold pair.

    The strategy starts at test_start and continues
    beyond test_end so that open positions can naturally
    close.
    """

    test_data = data.loc[
        test_start:
    ].copy()

    return run_strategy(

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


def _evaluate_test_portfolio(
    portfolio,
    test_start,
    test_end
):
    """
    Evaluate one portfolio only over the requested
    out-of-sample test period.

    Returns:

        total_return
        mu
        sigma
        sharpe
        max_drawdown
        calmar
    """

    if portfolio is None or portfolio.empty:
        return {
            "total_return": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan
        }

    evaluation = portfolio.loc[
        test_start:test_end
    ].copy()

    if evaluation.empty:
        return {
            "total_return": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan
        }

    return calculate_portfolio_statistics(
        evaluation
    )


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
    Run walk-forward optimization for ONE
    max_num_components.

    For every walk-forward window:

        1. Run Monte-Carlo calibration.
        2. Obtain four threshold-selection methods:

           A. Sharpe winner
           B. Calmar winner
           C. Return / Risk winner
           D. Robust top-10% Sharpe region

        3. Apply ALL FOUR threshold pairs to the
           following out-of-sample test period.

        4. Calculate OOS performance for every method.

        5. Store all calibration and OOS statistics.

    The four methods are therefore compared on exactly
    the same moving test windows.

    IMPORTANT:

    The robust top-10% method is not an independent
    fourth optimization. It is the median threshold
    pair of the top 10% of MC simulations ranked
    by Sharpe.
    """

    start_time = time.time()

    start_date = pd.Timestamp(
        start_date
    )

    data = data.sort_index()

    first_date = data.index.min()
    last_date = data.index.max()

    # =========================================================
    # FIRST TEST DATE
    # =========================================================

    test_start = (
        start_date +
        pd.DateOffset(
            years=window
        )
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

        # =====================================================
        # CALIBRATION PERIOD
        # =====================================================

        calibration_start = first_date

        calibration_end = (
            test_start -
            pd.Timedelta(
                days=1
            )
        )

        # =====================================================
        # TEST PERIOD
        # =====================================================

        test_end = (
            test_start +
            pd.DateOffset(
                months=test_span
            )
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
        # 2. EXTRACT FOUR THRESHOLD PAIRS
        # =====================================================

        # -----------------------------------------------------
        # A. SHARPE
        # -----------------------------------------------------

        sharpe_buy_thr = optimization[
            "best_buy_thr"
        ]

        sharpe_sell_thr = optimization[
            "best_sell_thr"
        ]

        # -----------------------------------------------------
        # B. CALMAR
        # -----------------------------------------------------

        calmar_buy_thr = optimization[
            "calmar_buy_thr"
        ]

        calmar_sell_thr = optimization[
            "calmar_sell_thr"
        ]

        # -----------------------------------------------------
        # C. RETURN / RISK
        # -----------------------------------------------------

        return_risk_buy_thr = optimization[
            "return_risk_buy_thr"
        ]

        return_risk_sell_thr = optimization[
            "return_risk_sell_thr"
        ]

        # -----------------------------------------------------
        # D. ROBUST TOP-10% SHARPE
        # -----------------------------------------------------

        robust_buy_thr = optimization[
            "robust_buy_thr"
        ]

        robust_sell_thr = optimization[
            "robust_sell_thr"
        ]

        # =====================================================
        # 3. RUN FOUR OUT-OF-SAMPLE STRATEGIES
        # =====================================================

        # -----------------------------------------------------
        # A. SHARPE
        # -----------------------------------------------------

        sharpe_result = _run_test_strategy(

            data=data,

            components=components,

            membership=membership,

            strategy=strategy,

            ma_column=ma_column,

            ranking_column=ranking_column,

            max_num_components=max_num_components,

            test_start=test_start,

            buy_thr=sharpe_buy_thr,

            sell_thr=sharpe_sell_thr,

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

        # -----------------------------------------------------
        # B. CALMAR
        # -----------------------------------------------------

        calmar_result = _run_test_strategy(

            data=data,

            components=components,

            membership=membership,

            strategy=strategy,

            ma_column=ma_column,

            ranking_column=ranking_column,

            max_num_components=max_num_components,

            test_start=test_start,

            buy_thr=calmar_buy_thr,

            sell_thr=calmar_sell_thr,

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

        # -----------------------------------------------------
        # C. RETURN / RISK
        # -----------------------------------------------------

        return_risk_result = _run_test_strategy(

            data=data,

            components=components,

            membership=membership,

            strategy=strategy,

            ma_column=ma_column,

            ranking_column=ranking_column,

            max_num_components=max_num_components,

            test_start=test_start,

            buy_thr=return_risk_buy_thr,

            sell_thr=return_risk_sell_thr,

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

        # -----------------------------------------------------
        # D. ROBUST TOP-10%
        # -----------------------------------------------------

        robust_result = _run_test_strategy(

            data=data,

            components=components,

            membership=membership,

            strategy=strategy,

            ma_column=ma_column,

            ranking_column=ranking_column,

            max_num_components=max_num_components,

            test_start=test_start,

            buy_thr=robust_buy_thr,

            sell_thr=robust_sell_thr,

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
        # 4. EXTRACT PORTFOLIOS
        # =====================================================

        sharpe_portfolio = sharpe_result[
            "portfolio"
        ]

        calmar_portfolio = calmar_result[
            "portfolio"
        ]

        return_risk_portfolio = (
            return_risk_result[
                "portfolio"
            ]
        )

        robust_portfolio = robust_result[
            "portfolio"
        ]

        # =====================================================
        # 5. OOS STATISTICS
        # =====================================================

        sharpe_oos = _evaluate_test_portfolio(

            sharpe_portfolio,

            test_start,

            test_end
        )

        calmar_oos = _evaluate_test_portfolio(

            calmar_portfolio,

            test_start,

            test_end
        )

        return_risk_oos = _evaluate_test_portfolio(

            return_risk_portfolio,

            test_start,

            test_end
        )

        robust_oos = _evaluate_test_portfolio(

            robust_portfolio,

            test_start,

            test_end
        )

        # =====================================================
        # 6. DOW / INDEX RETURN
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
        # 7. EXCESS RETURNS
        # =====================================================

        sharpe_excess = (
            sharpe_oos["total_return"]
            - index_return
            if (
                not np.isnan(
                    sharpe_oos[
                        "total_return"
                    ]
                )
                and
                not np.isnan(
                    index_return
                )
            )
            else np.nan
        )

        calmar_excess = (
            calmar_oos["total_return"]
            - index_return
            if (
                not np.isnan(
                    calmar_oos[
                        "total_return"
                    ]
                )
                and
                not np.isnan(
                    index_return
                )
            )
            else np.nan
        )

        return_risk_excess = (
            return_risk_oos[
                "total_return"
            ]
            - index_return
            if (
                not np.isnan(
                    return_risk_oos[
                        "total_return"
                    ]
                )
                and
                not np.isnan(
                    index_return
                )
            )
            else np.nan
        )

        robust_excess = (
            robust_oos[
                "total_return"
            ]
            - index_return
            if (
                not np.isnan(
                    robust_oos[
                        "total_return"
                    ]
                )
                and
                not np.isnan(
                    index_return
                )
            )
            else np.nan
        )

        # =====================================================
        # 8. STORE WALK-FORWARD RESULT
        # =====================================================

        results.append({

            # =================================================
            # CONFIGURATION
            # =================================================

            "max_num_components":
                max_num_components,

            # =================================================
            # CALIBRATION PERIOD
            # =================================================

            "calibration_start":
                calibration_start,

            "calibration_end":
                calibration_end,

            # =================================================
            # TEST PERIOD
            # =================================================

            "test_start":
                test_start,

            "test_end":
                test_end,

            # =================================================
            # SHARPE THRESHOLDS
            # =================================================

            "sharpe_buy_thr":
                sharpe_buy_thr,

            "sharpe_sell_thr":
                sharpe_sell_thr,

            # =================================================
            # SHARPE CALIBRATION
            # =================================================

            "sharpe_calibration_score":
                optimization[
                    "best_sharpe"
                ],

            "sharpe_calibration_return":
                optimization[
                    "best_total_return"
                ],

            "sharpe_calibration_mu":
                optimization[
                    "best_mu"
                ],

            "sharpe_calibration_sigma":
                optimization[
                    "best_sigma"
                ],

            "sharpe_calibration_calmar":
                optimization[
                    "best_calmar"
                ],

            # =================================================
            # SHARPE OOS
            # =================================================

            "sharpe_oos_return":
                sharpe_oos[
                    "total_return"
                ],

            "sharpe_oos_mu":
                sharpe_oos[
                    "mu"
                ],

            "sharpe_oos_sigma":
                sharpe_oos[
                    "sigma"
                ],

            "sharpe_oos_ratio":
                sharpe_oos[
                    "sharpe"
                ],

            "sharpe_oos_max_drawdown":
                sharpe_oos[
                    "max_drawdown"
                ],

            "sharpe_oos_calmar":
                sharpe_oos[
                    "calmar"
                ],

            "sharpe_excess_return":
                sharpe_excess,

            # =================================================
            # CALMAR THRESHOLDS
            # =================================================

            "calmar_buy_thr":
                calmar_buy_thr,

            "calmar_sell_thr":
                calmar_sell_thr,

            # =================================================
            # CALMAR CALIBRATION
            # =================================================

            "calmar_calibration_score":
                optimization[
                    "calmar_score"
                ],

            "calmar_calibration_return":
                optimization[
                    "calmar_total_return"
                ],

            "calmar_calibration_sharpe":
                optimization[
                    "calmar_sharpe"
                ],

            # =================================================
            # CALMAR OOS
            # =================================================

            "calmar_oos_return":
                calmar_oos[
                    "total_return"
                ],

            "calmar_oos_mu":
                calmar_oos[
                    "mu"
                ],

            "calmar_oos_sigma":
                calmar_oos[
                    "sigma"
                ],

            "calmar_oos_ratio":
                calmar_oos[
                    "sharpe"
                ],

            "calmar_oos_max_drawdown":
                calmar_oos[
                    "max_drawdown"
                ],

            "calmar_oos_calmar":
                calmar_oos[
                    "calmar"
                ],

            "calmar_excess_return":
                calmar_excess,

            # =================================================
            # RETURN / RISK THRESHOLDS
            # =================================================

            "return_risk_buy_thr":
                return_risk_buy_thr,

            "return_risk_sell_thr":
                return_risk_sell_thr,

            # =================================================
            # RETURN / RISK CALIBRATION
            # =================================================

            "return_risk_calibration_score":
                optimization[
                    "return_risk_score"
                ],

            "return_risk_calibration_return":
                optimization[
                    "return_risk_total_return"
                ],

            "return_risk_calibration_sharpe":
                optimization[
                    "return_risk_sharpe"
                ],

            # =================================================
            # RETURN / RISK OOS
            # =================================================

            "return_risk_oos_return":
                return_risk_oos[
                    "total_return"
                ],

            "return_risk_oos_mu":
                return_risk_oos[
                    "mu"
                ],

            "return_risk_oos_sigma":
                return_risk_oos[
                    "sigma"
                ],

            "return_risk_oos_ratio":
                return_risk_oos[
                    "sharpe"
                ],

            "return_risk_oos_max_drawdown":
                return_risk_oos[
                    "max_drawdown"
                ],

            "return_risk_oos_calmar":
                return_risk_oos[
                    "calmar"
                ],

            "return_risk_excess_return":
                return_risk_excess,

            # =================================================
            # ROBUST THRESHOLDS
            # =================================================

            "robust_buy_thr":
                robust_buy_thr,

            "robust_sell_thr":
                robust_sell_thr,

            # =================================================
            # ROBUST TOP-10% INFORMATION
            # =================================================

            "top10_n":
                optimization[
                    "top10_n"
                ],

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

            "robust_calibration_sharpe":
                optimization[
                    "robust_sharpe"
                ],

            "robust_calibration_calmar":
                optimization[
                    "robust_calmar"
                ],

            "robust_calibration_return_risk":
                optimization[
                    "robust_return_risk"
                ],

            # =================================================
            # ROBUST OOS
            # =================================================

            "robust_oos_return":
                robust_oos[
                    "total_return"
                ],

            "robust_oos_mu":
                robust_oos[
                    "mu"
                ],

            "robust_oos_sigma":
                robust_oos[
                    "sigma"
                ],

            "robust_oos_ratio":
                robust_oos[
                    "sharpe"
                ],

            "robust_oos_max_drawdown":
                robust_oos[
                    "max_drawdown"
                ],

            "robust_oos_calmar":
                robust_oos[
                    "calmar"
                ],

            "robust_excess_return":
                robust_excess,

            # =================================================
            # INDEX
            # =================================================

            "index_return":
                index_return,

            # =================================================
            # COMPATIBILITY
            # =================================================

            "best_buy_thr":
                sharpe_buy_thr,

            "best_sell_thr":
                sharpe_sell_thr,

            "best_portfolio_return":
                sharpe_oos[
                    "total_return"
                ],

            "robust_portfolio_return":
                robust_oos[
                    "total_return"
                ],

            "best_excess_return":
                sharpe_excess,

            "mu_sigma":
                sharpe_oos[
                    "sharpe"
                ]
        })

        # =====================================================
        # 9. PROGRESS
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
            f""
            f"Sharpe="
            f"({sharpe_buy_thr:+.2f},"
            f"{sharpe_sell_thr:+.2f}) "
            f"OOS={sharpe_oos['total_return']:+.2%} | "
            f""
            f"Calmar="
            f"({calmar_buy_thr:+.2f},"
            f"{calmar_sell_thr:+.2f}) "
            f"OOS={calmar_oos['total_return']:+.2%} | "
            f""
            f"ReturnRisk="
            f"({return_risk_buy_thr:+.2f},"
            f"{return_risk_sell_thr:+.2f}) "
            f"OOS={return_risk_oos['total_return']:+.2%} | "
            f""
            f"Robust="
            f"({robust_buy_thr:+.2f},"
            f"{robust_sell_thr:+.2f}) "
            f"OOS={robust_oos['total_return']:+.2%} | "
            f""
            f"Index={index_return:+.2%} | "
            f"time={window_elapsed:.1f}s | "
            f"ETA={eta_minutes:.1f}min",
            flush=True
        )

        # =====================================================
        # 10. MOVE WALK-FORWARD WINDOW
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