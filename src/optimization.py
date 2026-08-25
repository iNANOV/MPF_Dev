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

    Stage 1 identifies a promising threshold region.

    Stage 2 performs a denser Monte-Carlo search around that
    region instead of searching the complete threshold space.

    Four criteria are evaluated:

        1. Sharpe
        2. Calmar
        3. Return / Risk
        4. Robust top-10% Sharpe

    The function returns the best threshold pair for each criterion
    plus statistics describing the Stage-2 search region.
    """

    from .strategy import run_strategy

    rng = np.random.default_rng(random_state)

    candidates = []

    # ---------------------------------------------------------
    # Define Stage-2 search region
    # ---------------------------------------------------------

    buy_low = center_buy_thr - 2 * buy_std
    buy_high = center_buy_thr + 2 * buy_std

    sell_low = center_sell_thr - 2 * sell_std
    sell_high = center_sell_thr + 2 * sell_std

    # Keep thresholds inside the normal search range
    buy_low = max(-0.30, buy_low)
    buy_high = min(0.10, buy_high)

    sell_low = max(-0.05, sell_low)
    sell_high = min(0.35, sell_high)

    # ---------------------------------------------------------
    # Monte-Carlo Stage 2
    # ---------------------------------------------------------

    for i in range(n_simulations):

        buy_thr = rng.uniform(
            buy_low,
            buy_high
        )

        sell_thr = rng.uniform(
            sell_low,
            sell_high
        )

        # Optional grid rounding
        if threshold_step is not None and threshold_step > 0:
            buy_thr = round(
                buy_thr / threshold_step
            ) * threshold_step

            sell_thr = round(
                sell_thr / threshold_step
            ) * threshold_step

        try:

            result = run_strategy(
                data=data,
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
                min_cash_in_percent=min_cash_in_percent,
            )

            portfolio = result["portfolio"]

            evaluation = portfolio.loc[
                calibration_start:calibration_end
            ]

            if evaluation.empty:
                continue

            values = evaluation[
                "portfolio_value"
            ].dropna()

            if len(values) < 2:
                continue

            total_return = (
                values.iloc[-1] /
                values.iloc[0]
                - 1
            )

            returns = values.pct_change().dropna()

            if returns.empty:
                continue

            mu = returns.mean()
            sigma = returns.std()

            if sigma == 0 or not np.isfinite(sigma):
                continue

            sharpe = mu / sigma

            running_max = values.cummax()

            drawdown = (
                values / running_max - 1
            )

            max_drawdown = drawdown.min()

            if max_drawdown < 0:
                calmar = (
                    total_return /
                    abs(max_drawdown)
                )
            else:
                calmar = np.inf

            return_risk = (
                total_return /
                sigma
            )

            candidates.append({
                "buy_thr": buy_thr,
                "sell_thr": sell_thr,
                "total_return": total_return,
                "mu": mu,
                "sigma": sigma,
                "sharpe": sharpe,
                "calmar": calmar,
                "return_risk": return_risk,
                "max_drawdown": max_drawdown,
            })

        except Exception:
            continue

    # ---------------------------------------------------------
    # No valid simulations
    # ---------------------------------------------------------

    if not candidates:
        return None

    results = pd.DataFrame(candidates)

    # ---------------------------------------------------------
    # Remove invalid values
    # ---------------------------------------------------------

    results = results.replace(
        [np.inf, -np.inf],
        np.nan
    )

    results = results.dropna(
        subset=[
            "sharpe",
            "calmar",
            "return_risk",
            "total_return",
        ]
    )

    if results.empty:
        return None

    # ---------------------------------------------------------
    # Four optimization criteria
    # ---------------------------------------------------------

    best_sharpe = results.loc[
        results["sharpe"].idxmax()
    ]

    best_calmar = results.loc[
        results["calmar"].idxmax()
    ]

    best_return_risk = results.loc[
        results["return_risk"].idxmax()
    ]

    # ---------------------------------------------------------
    # Robust top 10%
    #
    # Select top 10% according to Sharpe and use the
    # median threshold pair.
    # ---------------------------------------------------------

    top_n = max(
        1,
        int(np.ceil(len(results) * 0.10))
    )

    top10 = results.nlargest(
        top_n,
        "sharpe"
    )

    robust_buy_thr = top10[
        "buy_thr"
    ].median()

    robust_sell_thr = top10[
        "sell_thr"
    ].median()

    robust_sharpe = top10[
        "sharpe"
    ].median()

    robust_return = top10[
        "total_return"
    ].median()

    robust_calmar = top10[
        "calmar"
    ].median()

    robust_return_risk = top10[
        "return_risk"
    ].median()

    # ---------------------------------------------------------
    # Return everything needed by walk-forward
    # ---------------------------------------------------------

    return {

        # Stage-2 search region
        "stage2_buy_min": buy_low,
        "stage2_buy_max": buy_high,
        "stage2_sell_min": sell_low,
        "stage2_sell_max": sell_high,

        "stage2_n": len(results),

        # -----------------------------------------------------
        # Sharpe
        # -----------------------------------------------------

        "sharpe_buy_thr":
            best_sharpe["buy_thr"],

        "sharpe_sell_thr":
            best_sharpe["sell_thr"],

        "sharpe":
            best_sharpe["sharpe"],

        "sharpe_return":
            best_sharpe["total_return"],

        # -----------------------------------------------------
        # Calmar
        # -----------------------------------------------------

        "calmar_buy_thr":
            best_calmar["buy_thr"],

        "calmar_sell_thr":
            best_calmar["sell_thr"],

        "calmar":
            best_calmar["calmar"],

        "calmar_return":
            best_calmar["total_return"],

        # -----------------------------------------------------
        # Return / Risk
        # -----------------------------------------------------

        "return_risk_buy_thr":
            best_return_risk["buy_thr"],

        "return_risk_sell_thr":
            best_return_risk["sell_thr"],

        "return_risk":
            best_return_risk["return_risk"],

        "return_risk_return":
            best_return_risk["total_return"],

        # -----------------------------------------------------
        # Robust top 10%
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Distribution statistics
        # -----------------------------------------------------

        "stage2_buy_std":
            results["buy_thr"].std(),

        "stage2_sell_std":
            results["sell_thr"].std(),

        "stage2_buy_min":
            results["buy_thr"].min(),

        "stage2_buy_max":
            results["buy_thr"].max(),

        "stage2_sell_min":
            results["sell_thr"].min(),

        "stage2_sell_max":
            results["sell_thr"].max(),
    }

