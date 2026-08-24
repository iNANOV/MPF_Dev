import numpy as np
import pandas as pd

from .strategy import run_strategy


# ============================================================
# MU / SIGMA / SHARPE
# ============================================================

def calculate_mu_sigma(portfolio):
    """
    Calculate annualized mean, volatility and Sharpe ratio
    from daily portfolio returns.

    Kept for compatibility with existing code.
    """

    if portfolio is None or portfolio.empty:
        return np.nan, np.nan, np.nan

    if "return" not in portfolio.columns:
        return np.nan, np.nan, np.nan

    returns = (
        portfolio["return"]
        .diff()
        .dropna()
    )

    if len(returns) < 2:
        return np.nan, np.nan, np.nan

    mu = returns.mean() * 252

    sigma = (
        returns.std()
        * np.sqrt(252)
    )

    if sigma == 0 or np.isnan(sigma):
        sharpe = np.nan
    else:
        sharpe = mu / sigma

    return (
        mu,
        sigma,
        sharpe
    )


# ============================================================
# PORTFOLIO STATISTICS
# ============================================================

def calculate_portfolio_statistics(portfolio):
    """
    Calculate performance statistics from daily
    portfolio values.

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

    if "portfolio_value" not in portfolio.columns:

        return {
            "total_return": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan
        }

    values = (
        portfolio["portfolio_value"]
        .dropna()
    )

    if len(values) < 2:

        return {
            "total_return": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan
        }

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    total_return = (
        values.iloc[-1]
        /
        values.iloc[0]
        - 1
    )

    # --------------------------------------------------------
    # Daily returns
    # --------------------------------------------------------

    daily_returns = (
        values
        .pct_change()
        .dropna()
    )

    if len(daily_returns) < 2:

        return {
            "total_return": total_return,
            "mu": np.nan,
            "sigma": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan
        }

    # --------------------------------------------------------
    # Annualized return
    # --------------------------------------------------------

    mu = (
        daily_returns.mean()
        * 252
    )

    # --------------------------------------------------------
    # Annualized volatility
    # --------------------------------------------------------

    sigma = (
        daily_returns.std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # Sharpe ratio
    # --------------------------------------------------------

    if (
        sigma == 0
        or np.isnan(sigma)
    ):

        sharpe = np.nan

    else:

        sharpe = mu / sigma

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    running_max = values.cummax()

    drawdown = (
        values / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    # --------------------------------------------------------
    # Annualized return for Calmar
    # --------------------------------------------------------

    try:

        start_date = values.index[0]
        end_date = values.index[-1]

        days = (
            end_date - start_date
        ).days

        years = days / 365.25

    except Exception:

        years = np.nan

    if (
        not np.isnan(years)
        and years > 0
    ):

        annualized_return = (
            values.iloc[-1]
            /
            values.iloc[0]
        ) ** (1 / years) - 1

    else:

        annualized_return = np.nan

    # --------------------------------------------------------
    # Calmar ratio
    # --------------------------------------------------------

    if (
        max_drawdown == 0
        or np.isnan(max_drawdown)
        or np.isnan(annualized_return)
    ):

        calmar = np.nan

    else:

        calmar = (
            annualized_return
            /
            abs(max_drawdown)
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "total_return":
            total_return,

        "mu":
            mu,

        "sigma":
            sigma,

        "sharpe":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "calmar":
            calmar
    }


# ============================================================
# THRESHOLD GENERATION
# ============================================================

def generate_thresholds(
    strategy,
    step=0.01
):
    """
    Generate possible buy/sell thresholds.

    Mean reversion:
        buy  = [-0.30, 0]
        sell = [0, 0.30]

    Momentum:
        buy  = [0, 0.30]
        sell = [-0.30, 0]
    """

    values = np.round(
        np.arange(
            -0.30,
            0.30 + step / 2,
            step
        ),
        2
    )

    if strategy == "mean_reversion":

        buy_values = values[
            (values >= -0.30)
            &
            (values <= 0)
        ]

        sell_values = values[
            (values >= 0)
            &
            (values <= 0.30)
        ]

    elif strategy == "momentum":

        buy_values = values[
            (values >= 0)
            &
            (values <= 0.30)
        ]

        sell_values = values[
            (values >= -0.30)
            &
            (values <= 0)
        ]

    else:

        raise ValueError(
            "strategy must be "
            "'mean_reversion' or "
            "'momentum'"
        )

    return (
        buy_values,
        sell_values
    )


# ============================================================
# MONTE-CARLO OPTIMIZATION
# ============================================================

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
    min_cash_in_percent=10,
    progress_info=None
):
    """
    Monte-Carlo optimization on historical calibration data.

    Step 1:
        - Run broad Monte-Carlo search.
        - Calculate several performance statistics.
        - Identify the single best simulation.
        - Identify the top 10% region.
        - Measure the stability/width of the top 10% region.
        - Calculate several possible optimization objectives.

    No two-stage Monte-Carlo search is performed yet.

    Stored statistics:

        Single best:
            total_return
            mu
            sigma
            sharpe
            max_drawdown
            calmar

        Top 10%:
            top10_n
            mean/median Sharpe
            mean/median return
            mean/median drawdown
            mean/median Calmar

        Top 10% threshold distribution:
            buy std/min/max
            sell std/min/max

        Candidate optimization scores:
            optimize_sharpe
            optimize_calmar
            optimize_return_risk
            optimize_robust_top10_sharpe
    """

    # ========================================================
    # RANDOM NUMBER GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        random_state
    )

    # ========================================================
    # CALIBRATION DATA
    # ========================================================

    calibration_data = data.loc[
        calibration_start:calibration_end
    ].copy()

    if calibration_data.empty:
        return None

    # ========================================================
    # THRESHOLD RANGES
    # ========================================================

    buy_values, sell_values = generate_thresholds(
        strategy,
        step=threshold_step
    )

    results = []

    # ========================================================
    # PROGRESS INFORMATION
    # ========================================================

    if progress_info is not None:

        window = progress_info.get(
            "window",
            "?"
        )

        total_windows = progress_info.get(
            "total_windows",
            "?"
        )

    else:

        window = "?"
        total_windows = "?"

    # ========================================================
    # MONTE-CARLO SEARCH
    # ========================================================

    for simulation in range(
        1,
        n_simulations + 1
    ):

        # ----------------------------------------------------
        # Random thresholds
        # ----------------------------------------------------

        buy_thr = float(
            rng.choice(
                buy_values
            )
        )

        sell_thr = float(
            rng.choice(
                sell_values
            )
        )

        try:

            # ------------------------------------------------
            # Run strategy
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Calculate statistics
            # ------------------------------------------------

            stats = calculate_portfolio_statistics(
                portfolio
            )

            # ------------------------------------------------
            # Require valid Sharpe
            # ------------------------------------------------

            if np.isnan(
                stats["sharpe"]
            ):
                continue

            # ------------------------------------------------
            # Store simulation
            # ------------------------------------------------

            results.append(
                {
                    "buy_thr": buy_thr,
                    "sell_thr": sell_thr,

                    "total_return":
                        stats["total_return"],

                    "mu":
                        stats["mu"],

                    "sigma":
                        stats["sigma"],

                    "sharpe":
                        stats["sharpe"],

                    "max_drawdown":
                        stats["max_drawdown"],

                    "calmar":
                        stats["calmar"]
                }
            )

        except Exception:
            continue

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            simulation == 1
            or
            simulation % 10 == 0
            or
            simulation == n_simulations
        ):

            print(
                f"[MC] "
                f"components={max_num_components} | "
                f"window={window}/{total_windows} | "
                f"simulation={simulation}/{n_simulations} | "
                f"valid={len(results)}",
                flush=True
            )

    # ========================================================
    # NO VALID RESULTS
    # ========================================================

    if not results:
        return None

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SINGLE BEST SIMULATION
    # ========================================================

    best = results_df.loc[
        results_df["sharpe"].idxmax()
    ].copy()

    # ========================================================
    # TOP 10%
    # ========================================================

    top_n = max(
        1,
        int(
            np.ceil(
                len(results_df) * 0.10
            )
        )
    )

    top10_results = (
        results_df
        .sort_values(
            "sharpe",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    # ========================================================
    # TOP 10% THRESHOLD DISTRIBUTION
    # ========================================================

    top10_buy_std = (
        top10_results["buy_thr"]
        .std()
    )

    top10_sell_std = (
        top10_results["sell_thr"]
        .std()
    )

    top10_buy_min = (
        top10_results["buy_thr"]
        .min()
    )

    top10_buy_max = (
        top10_results["buy_thr"]
        .max()
    )

    top10_sell_min = (
        top10_results["sell_thr"]
        .min()
    )

    top10_sell_max = (
        top10_results["sell_thr"]
        .max()
    )

    # ========================================================
    # TOP 10% PERFORMANCE STATISTICS
    # ========================================================

    top10_mean_sharpe = (
        top10_results["sharpe"]
        .mean()
    )

    top10_median_sharpe = (
        top10_results["sharpe"]
        .median()
    )

    top10_mean_return = (
        top10_results["total_return"]
        .mean()
    )

    top10_median_return = (
        top10_results["total_return"]
        .median()
    )

    top10_mean_drawdown = (
        top10_results["max_drawdown"]
        .mean()
    )

    top10_median_drawdown = (
        top10_results["max_drawdown"]
        .median()
    )

    top10_mean_calmar = (
        top10_results["calmar"]
        .mean()
    )

    top10_median_calmar = (
        top10_results["calmar"]
        .median()
    )

    # ========================================================
    # ROBUST THRESHOLDS
    # ========================================================

    robust_buy_thr = (
        top10_results["buy_thr"]
        .median()
    )

    robust_sell_thr = (
        top10_results["sell_thr"]
        .median()
    )

    # ========================================================
    # OPTIMIZATION OBJECTIVES
    # ========================================================

    # --------------------------------------------------------
    # 1. Sharpe
    # --------------------------------------------------------

    optimize_sharpe = (
        best["sharpe"]
    )

    # --------------------------------------------------------
    # 2. Calmar
    #
    # Best Calmar simulation
    # --------------------------------------------------------

    valid_calmar = (
        results_df[
            np.isfinite(
                results_df["calmar"]
            )
        ]
    )

    if valid_calmar.empty:

        optimize_calmar = np.nan

    else:

        best_calmar = valid_calmar.loc[
            valid_calmar["calmar"].idxmax()
        ]

        optimize_calmar = (
            best_calmar["calmar"]
        )

    # --------------------------------------------------------
    # 3. Return / Risk
    #
    # Here risk = annualized volatility.
    #
    # This is effectively:
    #
    #     mu / sigma
    #
    # but is explicitly stored under its own
    # optimization objective name.
    # --------------------------------------------------------

    results_df["return_risk"] = (
        results_df["mu"]
        /
        results_df["sigma"]
    )

    valid_return_risk = (
        results_df[
            np.isfinite(
                results_df["return_risk"]
            )
        ]
    )

    if valid_return_risk.empty:

        optimize_return_risk = np.nan

    else:

        best_return_risk = (
            valid_return_risk.loc[
                valid_return_risk[
                    "return_risk"
                ].idxmax()
            ]
        )

        optimize_return_risk = (
            best_return_risk[
                "return_risk"
            ]
        )

    # --------------------------------------------------------
    # 4. Robust top-10% Sharpe
    #
    # This measures the quality of the region rather
    # than the single best point.
    # --------------------------------------------------------

    optimize_robust_top10_sharpe = (
        top10_median_sharpe
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        # ----------------------------------------------------
        # Single best simulation
        # ----------------------------------------------------

        "best_buy_thr":
            best["buy_thr"],

        "best_sell_thr":
            best["sell_thr"],

        "best_sharpe":
            best["sharpe"],

        "best_total_return":
            best["total_return"],

        "best_mu":
            best["mu"],

        "best_sigma":
            best["sigma"],

        "best_max_drawdown":
            best["max_drawdown"],

        "best_calmar":
            best["calmar"],

        # ----------------------------------------------------
        # Robust thresholds
        # ----------------------------------------------------

        "robust_buy_thr":
            robust_buy_thr,

        "robust_sell_thr":
            robust_sell_thr,

        # ----------------------------------------------------
        # Top 10% threshold distribution
        # ----------------------------------------------------

        "top10_n":
            top_n,

        "top10_buy_std":
            top10_buy_std,

        "top10_sell_std":
            top10_sell_std,

        "top10_buy_min":
            top10_buy_min,

        "top10_buy_max":
            top10_buy_max,

        "top10_sell_min":
            top10_sell_min,

        "top10_sell_max":
            top10_sell_max,

        # ----------------------------------------------------
        # Top 10% performance
        # ----------------------------------------------------

        "top10_mean_sharpe":
            top10_mean_sharpe,

        "top10_median_sharpe":
            top10_median_sharpe,

        "top10_mean_return":
            top10_mean_return,

        "top10_median_return":
            top10_median_return,

        "top10_mean_drawdown":
            top10_mean_drawdown,

        "top10_median_drawdown":
            top10_median_drawdown,

        "top10_mean_calmar":
            top10_mean_calmar,

        "top10_median_calmar":
            top10_median_calmar,

        # ----------------------------------------------------
        # Optimization objectives
        # ----------------------------------------------------

        "optimize_sharpe":
            optimize_sharpe,

        "optimize_calmar":
            optimize_calmar,

        "optimize_return_risk":
            optimize_return_risk,

        "optimize_robust_top10_sharpe":
            optimize_robust_top10_sharpe,

        # ----------------------------------------------------
        # Compatibility
        # ----------------------------------------------------

        "mu":
            best["mu"],

        "sigma":
            best["sigma"],

        "mu_sigma":
            best["sharpe"],

        "sharpe":
            best["sharpe"],

        "total_return":
            best["total_return"],

        "max_drawdown":
            best["max_drawdown"],

        "calmar":
            best["calmar"],

        # ----------------------------------------------------
        # Complete simulation results
        # ----------------------------------------------------

        "all_results":
            results_df,

        "top10_results":
            top10_results
    }