import numpy as np
import pandas as pd

from .strategy import run_strategy


def calculate_mu_sigma(portfolio):
    """
    Calculate annualized mean and standard deviation
    from daily portfolio returns.
    """

    returns = portfolio["return"].diff().dropna()

    if len(returns) < 2:
        return np.nan, np.nan, np.nan

    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)

    if sigma == 0 or np.isnan(sigma):
        sharpe = np.nan
    else:
        sharpe = mu / sigma

    return mu, sigma, sharpe


def generate_thresholds(strategy, step=0.01):
    """
    Generate all possible buy/sell thresholds.

    Mean reversion:
        buy  = [0, -0.3]
        sell = [0,  0.3]

    Momentum:
        buy  = [0,  0.3]
        sell = [0, -0.3]
    """

    values = np.round(
        np.arange(-0.30, 0.30 + step / 2, step),
        2
    )

    if strategy == "mean_reversion":

        buy_values = values[
            (values >= -0.30) &
            (values <= 0)
        ]

        sell_values = values[
            (values >= 0) &
            (values <= 0.30)
        ]

    elif strategy == "momentum":

        buy_values = values[
            (values >= 0) &
            (values <= 0.30)
        ]

        sell_values = values[
            (values >= -0.30) &
            (values <= 0)
        ]

    else:
        raise ValueError(
            "strategy must be 'mean_reversion' or 'momentum'"
        )

    return buy_values, sell_values


def monte_carlo_optimize(
    data,
    components,
    membership,
    calibration_start,
    calibration_end,
    max_num_components,
    strategy,
    ma_column,
    ranking_column,
    n_simulations=500,
    threshold_step=0.01,
    random_state=None,
    initial_capital=100_000,
    abs_cost_for_a_trade=5,
    percent_cost_for_a_trade=0.001,
    max_investment_size_in_percent=50,
    min_cash_in_percent=10
):
    """
    Monte-Carlo optimization on historical calibration data.

    Only information available up to calibration_end is used.
    """

    rng = np.random.default_rng(random_state)

    calibration_data = data.loc[
        calibration_start:calibration_end
    ].copy()

    if calibration_data.empty:
        return None

    buy_values, sell_values = generate_thresholds(
        strategy,
        step=threshold_step
    )

    results = []

    for _ in range(n_simulations):

        buy_thr = float(rng.choice(buy_values))
        sell_thr = float(rng.choice(sell_values))

        try:

            result = run_strategy(
                data=calibration_data,
                components=components,
                ma_column=ma_column,
                buy_thr=buy_thr,
                sell_thr=sell_thr,
                strategy=strategy,
                membership=membership,
                max_num_components=max_num_components,
                start_date=calibration_start,
                ranking_column=ranking_column,
                initial_capital=initial_capital,
                abs_cost_for_a_trade=abs_cost_for_a_trade,
                percent_cost_for_a_trade=percent_cost_for_a_trade,
                max_investment_size_in_percent=max_investment_size_in_percent,
                min_cash_in_percent=min_cash_in_percent
            )

            portfolio = result["portfolio"]

            mu, sigma, sharpe = calculate_mu_sigma(
                portfolio
            )

            if np.isnan(sharpe):
                continue

            results.append({
                "buy_thr": buy_thr,
                "sell_thr": sell_thr,
                "mu": mu,
                "sigma": sigma,
                "mu_sigma": sharpe
            })

        except Exception:
            continue

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # Best parameter combination
    best = results_df.loc[
        results_df["mu_sigma"].idxmax()
    ].copy()

    return {
        "best_buy_thr": best["buy_thr"],
        "best_sell_thr": best["sell_thr"],
        "mu": best["mu"],
        "sigma": best["sigma"],
        "mu_sigma": best["mu_sigma"],
        "all_results": results_df
    }