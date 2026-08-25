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

    One Monte-Carlo run evaluates the same threshold combinations
    against four optimization criteria:

        1. Sharpe
        2. Calmar
        3. Return / Risk
        4. Robust top-10% Sharpe region

    Every valid simulation stores:

        buy_thr
        sell_thr

        total_return
        mu
        sigma
        sharpe
        max_drawdown
        calmar

        return_risk

    The robust criterion is NOT a fourth independent simulation.

    Instead, the top 10% of simulations ranked by Sharpe are
    treated as a robust performance region.

    The median thresholds of that region are used as the
    robust threshold candidate.

    Additional top-10% statistics describe the stability
    and width of that region.
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
    # THRESHOLD RANGE
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
    # MONTE-CARLO SIMULATIONS
    # ========================================================

    for simulation in range(
        1,
        n_simulations + 1
    ):

        # ----------------------------------------------------
        # RANDOM THRESHOLD COMBINATION
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

            # =================================================
            # RUN STRATEGY
            # =================================================

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

            # =================================================
            # PORTFOLIO STATISTICS
            # =================================================

            stats = calculate_portfolio_statistics(
                portfolio
            )

            total_return = stats[
                "total_return"
            ]

            mu = stats[
                "mu"
            ]

            sigma = stats[
                "sigma"
            ]

            sharpe = stats[
                "sharpe"
            ]

            max_drawdown = stats[
                "max_drawdown"
            ]

            calmar = stats[
                "calmar"
            ]

            # =================================================
            # CRITERION 1
            # SHARPE
            # =================================================

            # Sharpe is already calculated as:

            #     annualized return / annualized volatility

            sharpe_score = sharpe

            # =================================================
            # CRITERION 2
            # CALMAR
            # =================================================

            calmar_score = calmar

            # =================================================
            # CRITERION 3
            # RETURN / RISK
            # =================================================

            # Here we deliberately use total return divided
            # by annualized volatility.
            #
            # This is different from Sharpe because:
            #
            #     Sharpe     = annualized return / volatility
            #
            #     ReturnRisk = total return / volatility
            #
            # This allows us to investigate whether a threshold
            # combination produces a strong absolute return
            # relative to the risk taken.

            if (
                sigma is None
                or np.isnan(sigma)
                or sigma == 0
                or total_return is None
                or np.isnan(total_return)
            ):

                return_risk_score = np.nan

            else:

                return_risk_score = (
                    total_return /
                    sigma
                )

            # =================================================
            # VALID SIMULATION
            # =================================================

            if (
                sharpe_score is None
                or np.isnan(sharpe_score)
            ):

                continue

            # =================================================
            # STORE SIMULATION
            # =================================================

            results.append(
                {
                    "buy_thr":
                        buy_thr,

                    "sell_thr":
                        sell_thr,

                    # -----------------------------------------
                    # Portfolio statistics
                    # -----------------------------------------

                    "total_return":
                        total_return,

                    "mu":
                        mu,

                    "sigma":
                        sigma,

                    "sharpe":
                        sharpe_score,

                    "max_drawdown":
                        max_drawdown,

                    "calmar":
                        calmar_score,

                    # -----------------------------------------
                    # Criterion 3
                    # -----------------------------------------

                    "return_risk":
                        return_risk_score
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
    # CRITERION 1
    # BEST SHARPE
    # ========================================================

    valid_sharpe = results_df[
        results_df["sharpe"].notna()
    ]

    if valid_sharpe.empty:
        return None

    best_sharpe = valid_sharpe.loc[
        valid_sharpe[
            "sharpe"
        ].idxmax()
    ].copy()

    # ========================================================
    # CRITERION 2
    # BEST CALMAR
    # ========================================================

    valid_calmar = results_df[
        results_df["calmar"].notna()
    ]

    if valid_calmar.empty:

        best_calmar = None

    else:

        best_calmar = valid_calmar.loc[
            valid_calmar[
                "calmar"
            ].idxmax()
        ].copy()

    # ========================================================
    # CRITERION 3
    # BEST RETURN / RISK
    # ========================================================

    valid_return_risk = results_df[
        results_df["return_risk"].notna()
    ]

    if valid_return_risk.empty:

        best_return_risk = None

    else:

        best_return_risk = (
            valid_return_risk.loc[
                valid_return_risk[
                    "return_risk"
                ].idxmax()
            ].copy()
        )

    # ========================================================
    # CRITERION 4
    # ROBUST TOP-10% SHARPE REGION
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
    # ROBUST THRESHOLDS
    # ========================================================

    robust_buy_thr = (
        top10_results[
            "buy_thr"
        ].median()
    )

    robust_sell_thr = (
        top10_results[
            "sell_thr"
        ].median()
    )

    # ========================================================
    # TOP-10% THRESHOLD STABILITY
    # ========================================================

    top10_buy_std = (
        top10_results[
            "buy_thr"
        ].std()
    )

    top10_sell_std = (
        top10_results[
            "sell_thr"
        ].std()
    )

    top10_buy_min = (
        top10_results[
            "buy_thr"
        ].min()
    )

    top10_buy_max = (
        top10_results[
            "buy_thr"
        ].max()
    )

    top10_sell_min = (
        top10_results[
            "sell_thr"
        ].min()
    )

    top10_sell_max = (
        top10_results[
            "sell_thr"
        ].max()
    )

    # ========================================================
    # TOP-10% PERFORMANCE
    # ========================================================

    top10_mean_sharpe = (
        top10_results[
            "sharpe"
        ].mean()
    )

    top10_median_sharpe = (
        top10_results[
            "sharpe"
        ].median()
    )

    top10_mean_return = (
        top10_results[
            "total_return"
        ].mean()
    )

    top10_median_return = (
        top10_results[
            "total_return"
        ].median()
    )

    top10_mean_drawdown = (
        top10_results[
            "max_drawdown"
        ].mean()
    )

    top10_median_drawdown = (
        top10_results[
            "max_drawdown"
        ].median()
    )

    top10_mean_calmar = (
        top10_results[
            "calmar"
        ].mean()
    )

    top10_median_calmar = (
        top10_results[
            "calmar"
        ].median()
    )

    top10_mean_return_risk = (
        top10_results[
            "return_risk"
        ].mean()
    )

    top10_median_return_risk = (
        top10_results[
            "return_risk"
        ].median()
    )

    # ========================================================
    # ROBUST CRITERION SCORE
    # ========================================================

    # The robust criterion is represented by the performance
    # of the top-10% region, NOT by pretending that the median
    # threshold pair itself was simulated.
    #
    # The median Sharpe of the top-10% region is therefore
    # used as the robust-region score.

    robust_sharpe = (
        top10_median_sharpe
    )

    robust_calmar = (
        top10_median_calmar
    )

    robust_return_risk = (
        top10_median_return_risk
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        # ====================================================
        # CRITERION 1: SHARPE
        # ====================================================

        "best_buy_thr":
            best_sharpe[
                "buy_thr"
            ],

        "best_sell_thr":
            best_sharpe[
                "sell_thr"
            ],

        "best_sharpe":
            best_sharpe[
                "sharpe"
            ],

        "best_total_return":
            best_sharpe[
                "total_return"
            ],

        "best_mu":
            best_sharpe[
                "mu"
            ],

        "best_sigma":
            best_sharpe[
                "sigma"
            ],

        "best_max_drawdown":
            best_sharpe[
                "max_drawdown"
            ],

        "best_calmar":
            best_sharpe[
                "calmar"
            ],

        "best_return_risk":
            best_sharpe[
                "return_risk"
            ],

        # ====================================================
        # CRITERION 2: CALMAR
        # ====================================================

        "calmar_buy_thr":
            (
                best_calmar[
                    "buy_thr"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_sell_thr":
            (
                best_calmar[
                    "sell_thr"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_score":
            (
                best_calmar[
                    "calmar"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_total_return":
            (
                best_calmar[
                    "total_return"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_sharpe":
            (
                best_calmar[
                    "sharpe"
                ]
                if best_calmar is not None
                else np.nan
            ),

        # ====================================================
        # CRITERION 3: RETURN / RISK
        # ====================================================

        "return_risk_buy_thr":
            (
                best_return_risk[
                    "buy_thr"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_sell_thr":
            (
                best_return_risk[
                    "sell_thr"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_score":
            (
                best_return_risk[
                    "return_risk"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_total_return":
            (
                best_return_risk[
                    "total_return"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_sharpe":
            (
                best_return_risk[
                    "sharpe"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        # ====================================================
        # CRITERION 4: ROBUST TOP-10%
        # ====================================================

        "robust_buy_thr":
            robust_buy_thr,

        "robust_sell_thr":
            robust_sell_thr,

        "robust_sharpe":
            robust_sharpe,

        "robust_calmar":
            robust_calmar,

        "robust_return_risk":
            robust_return_risk,

        # ====================================================
        # TOP-10% SIZE
        # ====================================================

        "top10_n":
            top_n,

        # ====================================================
        # TOP-10% THRESHOLD DISTRIBUTION
        # ====================================================

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

        # ====================================================
        # TOP-10% PERFORMANCE
        # ====================================================

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

        "top10_mean_return_risk":
            top10_mean_return_risk,

        "top10_median_return_risk":
            top10_median_return_risk,

        # ====================================================
        # COMPATIBILITY
        # ====================================================

        "mu":
            best_sharpe[
                "mu"
            ],

        "sigma":
            best_sharpe[
                "sigma"
            ],

        "mu_sigma":
            best_sharpe[
                "sharpe"
            ],

        "sharpe":
            best_sharpe[
                "sharpe"
            ],

        "total_return":
            best_sharpe[
                "total_return"
            ],

        "max_drawdown":
            best_sharpe[
                "max_drawdown"
            ],

        "calmar":
            best_sharpe[
                "calmar"
            ],

        # ====================================================
        # ALL VALID SIMULATIONS
        # ====================================================

        "all_results":
            results_df,

        # ====================================================
        # TOP 10% SIMULATIONS
        # ====================================================

        "top10_results":
            top10_results
    }

def monte_carlo_stage2_optimize(
    data,
    components,
    membership,
    calibration_start,
    calibration_end,
    max_num_components,
    strategy,
    ma_column,
    ranking_column,
    center_buy_thr,
    center_sell_thr,
    buy_std,
    sell_std,
    n_simulations=500,
    threshold_step=0.01,
    random_state=123,
    initial_capital=100_000,
    abs_cost_for_a_trade=5,
    percent_cost_for_a_trade=0.001,
    max_investment_size_in_percent=50,
    min_cash_in_percent=10,
):
    """
    Stage-2 Monte-Carlo optimization.

    Stage 1 identifies a promising robust threshold region.

    Stage 2 performs a denser Monte-Carlo search around that
    region.

    Four criteria are evaluated for every valid simulation:

        1. Sharpe
        2. Calmar
        3. Return / Risk
        4. Robust top-10% Sharpe region

    The robust criterion is based on the top 10% of Stage-2
    simulations ranked by Sharpe.

    Parameters
    ----------
    center_buy_thr : float
        Stage-1 robust buy threshold.

    center_sell_thr : float
        Stage-1 robust sell threshold.

    buy_std : float
        Stage-1 top-10% standard deviation of buy thresholds.

    sell_std : float
        Stage-1 top-10% standard deviation of sell thresholds.

    The Stage-2 search region is:

        center +/- 2 * std

    and is clipped to the valid threshold range for the
    selected strategy.
    """

    # =========================================================
    # RANDOM NUMBER GENERATOR
    # =========================================================

    rng = np.random.default_rng(
        random_state
    )

    # =========================================================
    # CALIBRATION DATA
    # =========================================================

    calibration_data = data.loc[
        calibration_start:calibration_end
    ].copy()

    if calibration_data.empty:
        return None

    # =========================================================
    # VALIDATE STAGE-1 CENTERS
    # =========================================================

    if (
        center_buy_thr is None
        or center_sell_thr is None
        or not np.isfinite(center_buy_thr)
        or not np.isfinite(center_sell_thr)
    ):
        return None

    # =========================================================
    # VALIDATE STANDARD DEVIATIONS
    # =========================================================

    if (
        buy_std is None
        or not np.isfinite(buy_std)
        or buy_std <= 0
    ):
        buy_std = threshold_step

    if (
        sell_std is None
        or not np.isfinite(sell_std)
        or sell_std <= 0
    ):
        sell_std = threshold_step

    # =========================================================
    # STAGE-2 REGION
    # =========================================================
    #
    # We search around the Stage-1 robust center.
    #
    # The width is based on the Stage-1 top-10% threshold
    # dispersion.
    #
    # Using +/- 2 standard deviations gives Stage 2 enough
    # freedom to explore the promising region without going
    # back to the entire [-0.30, +0.30] space.
    # =========================================================

    region_buy_min = (
        center_buy_thr -
        2.0 * buy_std
    )

    region_buy_max = (
        center_buy_thr +
        2.0 * buy_std
    )

    region_sell_min = (
        center_sell_thr -
        2.0 * sell_std
    )

    region_sell_max = (
        center_sell_thr +
        2.0 * sell_std
    )

    # =========================================================
    # VALID STRATEGY RANGE
    # =========================================================

    if strategy == "mean_reversion":

        valid_buy_min = -0.30
        valid_buy_max = 0.00

        valid_sell_min = 0.00
        valid_sell_max = 0.30

    elif strategy == "momentum":

        valid_buy_min = 0.00
        valid_buy_max = 0.30

        valid_sell_min = -0.30
        valid_sell_max = 0.00

    else:

        raise ValueError(
            "strategy must be "
            "'mean_reversion' or "
            "'momentum'"
        )

    # =========================================================
    # CLIP STAGE-2 REGION TO VALID RANGE
    # =========================================================

    region_buy_min = max(
        valid_buy_min,
        region_buy_min
    )

    region_buy_max = min(
        valid_buy_max,
        region_buy_max
    )

    region_sell_min = max(
        valid_sell_min,
        region_sell_min
    )

    region_sell_max = min(
        valid_sell_max,
        region_sell_max
    )

    # =========================================================
    # SAFETY CHECK
    # =========================================================

    if region_buy_min > region_buy_max:
        return None

    if region_sell_min > region_sell_max:
        return None

    # =========================================================
    # ROUND SEARCH REGION
    # =========================================================

    region_buy_min = round(
        region_buy_min,
        2
    )

    region_buy_max = round(
        region_buy_max,
        2
    )

    region_sell_min = round(
        region_sell_min,
        2
    )

    region_sell_max = round(
        region_sell_max,
        2
    )

    # =========================================================
    # POSSIBLE STAGE-2 THRESHOLDS
    # =========================================================

    buy_values = np.round(
        np.arange(
            region_buy_min,
            region_buy_max +
            threshold_step / 2,
            threshold_step
        ),
        2
    )

    sell_values = np.round(
        np.arange(
            region_sell_min,
            region_sell_max +
            threshold_step / 2,
            threshold_step
        ),
        2
    )

    if len(buy_values) == 0:
        return None

    if len(sell_values) == 0:
        return None

    # =========================================================
    # RESULTS
    # =========================================================

    results = []

    # =========================================================
    # MONTE-CARLO LOOP
    # =========================================================

    for simulation in range(
        1,
        n_simulations + 1
    ):

        # -----------------------------------------------------
        # RANDOM THRESHOLD
        # -----------------------------------------------------

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

            # =================================================
            # RUN STRATEGY
            # =================================================

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

            # =================================================
            # PORTFOLIO STATISTICS
            # =================================================

            stats = calculate_portfolio_statistics(
                portfolio
            )

            total_return = stats[
                "total_return"
            ]

            mu = stats[
                "mu"
            ]

            sigma = stats[
                "sigma"
            ]

            sharpe = stats[
                "sharpe"
            ]

            max_drawdown = stats[
                "max_drawdown"
            ]

            calmar = stats[
                "calmar"
            ]

            # =================================================
            # CRITERION 1
            # SHARPE
            # =================================================
            #
            # Already annualized by
            # calculate_portfolio_statistics().
            # =================================================

            sharpe_score = sharpe

            # =================================================
            # CRITERION 2
            # CALMAR
            # =================================================
            #
            # Already based on annualized return / abs(MDD).
            # =================================================

            calmar_score = calmar

            # =================================================
            # CRITERION 3
            # RETURN / RISK
            # =================================================
            #
            # Keep exactly the same definition as Stage 1:
            #
            #     total return / annualized volatility
            #
            # This deliberately differs from Sharpe.
            # =================================================

            if (
                sigma is None
                or not np.isfinite(sigma)
                or sigma == 0
                or total_return is None
                or not np.isfinite(total_return)
            ):

                return_risk_score = np.nan

            else:

                return_risk_score = (
                    total_return /
                    sigma
                )

            # =================================================
            # VALID SIMULATION
            # =================================================

            if (
                sharpe_score is None
                or not np.isfinite(sharpe_score)
            ):
                continue

            # =================================================
            # STORE SIMULATION
            # =================================================

            results.append(
                {
                    "buy_thr":
                        buy_thr,

                    "sell_thr":
                        sell_thr,

                    "total_return":
                        total_return,

                    "mu":
                        mu,

                    "sigma":
                        sigma,

                    "sharpe":
                        sharpe_score,

                    "max_drawdown":
                        max_drawdown,

                    "calmar":
                        calmar_score,

                    "return_risk":
                        return_risk_score,
                }
            )

        except Exception:

            continue

        # =====================================================
        # PROGRESS
        # =====================================================

        if (
            simulation == 1
            or simulation % 10 == 0
            or simulation == n_simulations
        ):

            print(
                f"[MC2] "
                f"components={max_num_components} | "
                f"simulation={simulation}/{n_simulations} | "
                f"valid={len(results)}",
                flush=True
            )

    # =========================================================
    # NO VALID RESULTS
    # =========================================================

    if not results:
        return None

    # =========================================================
    # RESULTS DATAFRAME
    # =========================================================

    results_df = pd.DataFrame(
        results
    )

    # =========================================================
    # CRITERION 1
    # BEST SHARPE
    # =========================================================

    valid_sharpe = results_df[
        results_df["sharpe"].notna()
    ]

    if valid_sharpe.empty:
        return None

    best_sharpe = valid_sharpe.loc[
        valid_sharpe[
            "sharpe"
        ].idxmax()
    ].copy()

    # =========================================================
    # CRITERION 2
    # BEST CALMAR
    # =========================================================

    valid_calmar = results_df[
        results_df["calmar"].notna()
    ]

    if valid_calmar.empty:

        best_calmar = None

    else:

        best_calmar = valid_calmar.loc[
            valid_calmar[
                "calmar"
            ].idxmax()
        ].copy()

    # =========================================================
    # CRITERION 3
    # BEST RETURN / RISK
    # =========================================================

    valid_return_risk = results_df[
        results_df["return_risk"].notna()
    ]

    if valid_return_risk.empty:

        best_return_risk = None

    else:

        best_return_risk = (
            valid_return_risk.loc[
                valid_return_risk[
                    "return_risk"
                ].idxmax()
            ].copy()
        )

    # =========================================================
    # CRITERION 4
    # ROBUST TOP-10% SHARPE REGION
    # =========================================================

    top10_n = max(
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
        .head(top10_n)
        .copy()
    )

    # =========================================================
    # ROBUST THRESHOLDS
    # =========================================================

    robust_buy_thr = (
        top10_results[
            "buy_thr"
        ].median()
    )

    robust_sell_thr = (
        top10_results[
            "sell_thr"
        ].median()
    )

    # =========================================================
    # TOP-10% PERFORMANCE
    # =========================================================

    robust_sharpe = (
        top10_results[
            "sharpe"
        ].median()
    )

    robust_calmar = (
        top10_results[
            "calmar"
        ].median()
    )

    robust_return_risk = (
        top10_results[
            "return_risk"
        ].median()
    )

    robust_return = (
        top10_results[
            "total_return"
        ].median()
    )

    # =========================================================
    # ACTUAL STAGE-2 DISTRIBUTION
    # =========================================================

    stage2_actual_buy_min = (
        results_df[
            "buy_thr"
        ].min()
    )

    stage2_actual_buy_max = (
        results_df[
            "buy_thr"
        ].max()
    )

    stage2_actual_sell_min = (
        results_df[
            "sell_thr"
        ].min()
    )

    stage2_actual_sell_max = (
        results_df[
            "sell_thr"
        ].max()
    )

    stage2_actual_buy_std = (
        results_df[
            "buy_thr"
        ].std()
    )

    stage2_actual_sell_std = (
        results_df[
            "sell_thr"
        ].std()
    )

    # =========================================================
    # TOP-10% DISTRIBUTION
    # =========================================================

    top10_buy_std = (
        top10_results[
            "buy_thr"
        ].std()
    )

    top10_sell_std = (
        top10_results[
            "sell_thr"
        ].std()
    )

    top10_buy_min = (
        top10_results[
            "buy_thr"
        ].min()
    )

    top10_buy_max = (
        top10_results[
            "buy_thr"
        ].max()
    )

    top10_sell_min = (
        top10_results[
            "sell_thr"
        ].min()
    )

    top10_sell_max = (
        top10_results[
            "sell_thr"
        ].max()
    )

    # =========================================================
    # RETURN
    # =========================================================

    return {

        # =====================================================
        # STAGE-2 SEARCH REGION
        # =====================================================

        "stage2_region_buy_min":
            region_buy_min,

        "stage2_region_buy_max":
            region_buy_max,

        "stage2_region_sell_min":
            region_sell_min,

        "stage2_region_sell_max":
            region_sell_max,

        # -----------------------------------------------------
        # Backward-compatible names
        # -----------------------------------------------------

        "stage2_buy_min":
            region_buy_min,

        "stage2_buy_max":
            region_buy_max,

        "stage2_sell_min":
            region_sell_min,

        "stage2_sell_max":
            region_sell_max,

        # =====================================================
        # STAGE-2 SIMULATION DISTRIBUTION
        # =====================================================

        "stage2_n":
            len(results_df),

        "stage2_actual_buy_min":
            stage2_actual_buy_min,

        "stage2_actual_buy_max":
            stage2_actual_buy_max,

        "stage2_actual_sell_min":
            stage2_actual_sell_min,

        "stage2_actual_sell_max":
            stage2_actual_sell_max,

        "stage2_actual_buy_std":
            stage2_actual_buy_std,

        "stage2_actual_sell_std":
            stage2_actual_sell_std,

        # =====================================================
        # CRITERION 1: SHARPE
        # =====================================================

        "sharpe_buy_thr":
            best_sharpe[
                "buy_thr"
            ],

        "sharpe_sell_thr":
            best_sharpe[
                "sell_thr"
            ],

        "sharpe":
            best_sharpe[
                "sharpe"
            ],

        "sharpe_return":
            best_sharpe[
                "total_return"
            ],

        "sharpe_mu":
            best_sharpe[
                "mu"
            ],

        "sharpe_sigma":
            best_sharpe[
                "sigma"
            ],

        "sharpe_calmar":
            best_sharpe[
                "calmar"
            ],

        "sharpe_return_risk":
            best_sharpe[
                "return_risk"
            ],

        # =====================================================
        # CRITERION 2: CALMAR
        # =====================================================

        "calmar_buy_thr":
            (
                best_calmar[
                    "buy_thr"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_sell_thr":
            (
                best_calmar[
                    "sell_thr"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar":
            (
                best_calmar[
                    "calmar"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_return":
            (
                best_calmar[
                    "total_return"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_sharpe":
            (
                best_calmar[
                    "sharpe"
                ]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_return_risk":
            (
                best_calmar[
                    "return_risk"
                ]
                if best_calmar is not None
                else np.nan
            ),

        # =====================================================
        # CRITERION 3: RETURN / RISK
        # =====================================================

        "return_risk_buy_thr":
            (
                best_return_risk[
                    "buy_thr"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_sell_thr":
            (
                best_return_risk[
                    "sell_thr"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk":
            (
                best_return_risk[
                    "return_risk"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_return":
            (
                best_return_risk[
                    "total_return"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_sharpe":
            (
                best_return_risk[
                    "sharpe"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        "return_risk_calmar":
            (
                best_return_risk[
                    "calmar"
                ]
                if best_return_risk is not None
                else np.nan
            ),

        # =====================================================
        # CRITERION 4: ROBUST TOP-10%
        # =====================================================

        "robust_buy_thr":
            robust_buy_thr,

        "robust_sell_thr":
            robust_sell_thr,

        "robust_sharpe":
            robust_sharpe,

        "robust_return":
            robust_return,

        "robust_calmar":
            robust_calmar,

        "robust_return_risk":
            robust_return_risk,

        # =====================================================
        # TOP-10% INFORMATION
        # =====================================================

        "top10_n":
            top10_n,

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

        "top10_mean_sharpe":
            top10_results[
                "sharpe"
            ].mean(),

        "top10_median_sharpe":
            top10_results[
                "sharpe"
            ].median(),

        "top10_mean_return":
            top10_results[
                "total_return"
            ].mean(),

        "top10_median_return":
            top10_results[
                "total_return"
            ].median(),

        "top10_mean_calmar":
            top10_results[
                "calmar"
            ].mean(),

        "top10_median_calmar":
            top10_results[
                "calmar"
            ].median(),

        "top10_mean_return_risk":
            top10_results[
                "return_risk"
            ].mean(),

        "top10_median_return_risk":
            top10_results[
                "return_risk"
            ].median(),
    }