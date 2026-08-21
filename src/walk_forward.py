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
    """

    start_date = pd.Timestamp(start_date)

    data = data.sort_index()

    first_date = data.index.min()
    last_date = data.index.max()

    test_start = start_date + pd.DateOffset(years=window)

    results = []

    iteration = 0

    while test_start <= last_date:

        iteration += 1

        calibration_start = first_date
        calibration_end = test_start - pd.Timedelta(days=1)

        test_end = (
            test_start +
            pd.DateOffset(months=test_span)
        )

        if test_end > last_date:
            test_end = last_date

        # --------------------------------------------------
        # 1. Monte-Carlo calibration
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
            random_state=random_state + iteration,
            initial_capital=initial_capital,
            abs_cost_for_a_trade=abs_cost_for_a_trade,
            percent_cost_for_a_trade=percent_cost_for_a_trade,
            max_investment_size_in_percent=max_investment_size_in_percent,
            min_cash_in_percent=min_cash_in_percent
        )

        if optimization is None:
            test_start += pd.DateOffset(months=moving_param)
            continue

        buy_thr = optimization["best_buy_thr"]
        sell_thr = optimization["best_sell_thr"]

        # --------------------------------------------------
        # 2. Apply selected parameters
        #
        # IMPORTANT:
        # use data beyond test_end so open trades can
        # finish naturally.
        # --------------------------------------------------

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
            abs_cost_for_a_trade=abs_cost_for_a_trade,
            percent_cost_for_a_trade=percent_cost_for_a_trade,
            max_investment_size_in_percent=max_investment_size_in_percent,
            min_cash_in_percent=min_cash_in_percent
        )

        portfolio = result["portfolio"]

        # --------------------------------------------------
        # 3. Evaluate only portfolio performance inside
        #    the TEST period.
        #
        #    Trades themselves are allowed to continue
        #    beyond test_end.
        # --------------------------------------------------

        evaluation = portfolio.loc[
            test_start:test_end
        ]

        if evaluation.empty:
            test_return = np.nan
        else:
            test_return = (
                evaluation["portfolio_value"].iloc[-1]
                / evaluation["portfolio_value"].iloc[0]
                - 1
            )

        # --------------------------------------------------
        # 4. Dow return over same test period
        # --------------------------------------------------

        index_col = "INDEX"

        if (
            test_start in data.index
            and test_end in data.index
        ):

            index_prices = data.loc[
                test_start:test_end,
                index_col
            ].dropna()

            if len(index_prices) >= 2:
                index_return = (
                    index_prices.iloc[-1]
                    / index_prices.iloc[0]
                    - 1
                )
            else:
                index_return = np.nan

        else:
            index_return = np.nan

        # --------------------------------------------------
        # 5. Store result
        # --------------------------------------------------

        results.append({
            "max_num_components": max_num_components,
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_buy_thr": buy_thr,
            "best_sell_thr": sell_thr,
            "mu": optimization["mu"],
            "sigma": optimization["sigma"],
            "mu_sigma": optimization["mu_sigma"],
            "portfolio_return": test_return,
            "index_return": index_return
        })

        # --------------------------------------------------
        # 6. Move window
        # --------------------------------------------------

        test_start += pd.DateOffset(
            months=moving_param
        )

    return pd.DataFrame(results)