import numpy as np
import pandas as pd

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

    Workflow
    --------
    1. Everything before the first test period is calibration data.
    2. Monte-Carlo optimization is performed on all data available
       before the test period.
    3. The selected thresholds are applied to the following test period.
    4. Trades are allowed to continue beyond test_end so that open
       trades can finish naturally.
    5. Portfolio performance is evaluated only inside the test period.
    6. The index return is calculated using the first and last
       available trading day inside the test period.
    7. The walk-forward window is then moved by moving_param months.
    """

    # --------------------------------------------------
    # 0. Prepare data
    # --------------------------------------------------

    start_date = pd.Timestamp(start_date)

    data = data.sort_index()

    first_date = data.index.min()
    last_date = data.index.max()

    # First test period starts after the initial calibration window
    test_start = (
        start_date +
        pd.DateOffset(years=window)
    )

    results = []

    iteration = 0

    # ==================================================
    # WALK-FORWARD LOOP
    # ==================================================

    while test_start <= last_date:

        iteration += 1

        # --------------------------------------------------
        # 1. Calibration period
        #
        # Everything before test_start is available for
        # calibration.
        # --------------------------------------------------

        calibration_start = first_date

        calibration_end = (
            test_start -
            pd.Timedelta(days=1)
        )

        # --------------------------------------------------
        # 2. Define test period
        # --------------------------------------------------

        requested_test_end = (
            test_start +
            pd.DateOffset(months=test_span)
        )

        test_end = min(
            requested_test_end,
            last_date
        )

        # --------------------------------------------------
        # 3. Monte-Carlo calibration
        # --------------------------------------------------

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
            )
        )

        # --------------------------------------------------
        # If calibration failed, move to next window
        # --------------------------------------------------

        if optimization is None:

            test_start += pd.DateOffset(
                months=moving_param
            )

            continue

        # --------------------------------------------------
        # Selected optimal thresholds
        # --------------------------------------------------

        buy_thr = optimization[
            "best_buy_thr"
        ]

        sell_thr = optimization[
            "best_sell_thr"
        ]

        # ==================================================
        # 4. APPLY OPTIMIZED STRATEGY
        # ==================================================
        #
        # IMPORTANT:
        #
        # We deliberately use ALL data from test_start
        # onwards, not only until test_end.
        #
        # This allows trades opened during the test period
        # to finish after test_end.
        #
        # However, performance is evaluated only until
        # test_end below.
        # ==================================================

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

        # ==================================================
        # 5. PORTFOLIO RETURN
        # ==================================================

        evaluation = portfolio.loc[
            test_start:test_end
        ].copy()

        if evaluation.empty:

            test_return = np.nan

        else:

            first_value = (
                evaluation[
                    "portfolio_value"
                ].iloc[0]
            )

            last_value = (
                evaluation[
                    "portfolio_value"
                ].iloc[-1]
            )

            if (
                pd.notna(first_value)
                and
                first_value != 0
                and
                pd.notna(last_value)
            ):

                test_return = (
                    last_value /
                    first_value
                    - 1
                )

            else:

                test_return = np.nan

        # ==================================================
        # 6. INDEX RETURN
        # ==================================================
        #
        # IMPORTANT:
        #
        # Do NOT require:
        #
        #     test_start in data.index
        #     test_end   in data.index
        #
        # because these dates can be weekends or holidays.
        #
        # Instead, select all available observations between
        # the requested test dates and use the first and last
        # actual trading days.
        # ==================================================

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

        # ==================================================
        # 7. STORE RESULT
        # ==================================================

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

        # ==================================================
        # 8. MOVE WALK-FORWARD WINDOW
        # ==================================================

        test_start += pd.DateOffset(
            months=moving_param
        )

    # ==================================================
    # 9. RETURN RESULTS
    # ==================================================

    return pd.DataFrame(results)