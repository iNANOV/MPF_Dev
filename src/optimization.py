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

    Four optimization candidates are determined from the
    SAME Monte-Carlo simulations:

        1. Sharpe
        2. Calmar
        3. Return / Risk
        4. Robust top-10% region

    The robust candidate is based on the median thresholds
    of the top 10% simulations ranked by Sharpe.

    Additionally, statistics describing the top-10% threshold
    region are returned:

        top10_buy_std
        top10_sell_std
        top10_buy_min
        top10_buy_max
        top10_sell_min
        top10_sell_max

    No additional Monte-Carlo run is required for the
    different optimization criteria.
    """

    # ========================================================
    # Random number generator
    # ========================================================

    rng = np.random.default_rng(
        random_state
    )

    # ========================================================
    # Calibration data
    # ========================================================

    calibration_data = data.loc[
        calibration_start:calibration_end
    ].copy()

    if calibration_data.empty:
        return None

    # ========================================================
    # Generate threshold ranges
    # ========================================================

    buy_values, sell_values = generate_thresholds(
        strategy,
        step=threshold_step
    )

    results = []

    # ========================================================
    # Progress information
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
    # Monte-Carlo simulations
    # ========================================================

    for simulation in range(
        1,
        n_simulations + 1
    ):

        # ----------------------------------------------------
        # Random threshold combination
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
            # Extract statistics
            # ------------------------------------------------

            sharpe = stats[
                "sharpe"
            ]

            calmar = stats[
                "calmar"
            ]

            sigma = stats[
                "sigma"
            ]

            total_return = stats[
                "total_return"
            ]

            # ------------------------------------------------
            # Return / Risk
            #
            # Total return divided by annualized volatility.
            # ------------------------------------------------

            if (
                sigma is None
                or
                np.isnan(sigma)
                or
                sigma == 0
            ):

                return_risk = np.nan

            else:

                return_risk = (
                    total_return /
                    sigma
                )

            # ------------------------------------------------
            # Skip invalid simulations
            # ------------------------------------------------

            if (
                sharpe is None
                or
                np.isnan(sharpe)
            ):

                continue

            # ------------------------------------------------
            # Store simulation
            # ------------------------------------------------

            results.append(
                {
                    "buy_thr":
                        buy_thr,

                    "sell_thr":
                        sell_thr,

                    "total_return":
                        total_return,

                    "mu":
                        stats["mu"],

                    "sigma":
                        sigma,

                    "sharpe":
                        sharpe,

                    "max_drawdown":
                        stats[
                            "max_drawdown"
                        ],

                    "calmar":
                        calmar,

                    "return_risk":
                        return_risk
                }
            )

        except Exception:

            continue

        # ====================================================
        # Progress
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
    # No valid results
    # ========================================================

    if not results:
        return None

    # ========================================================
    # Results DataFrame
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 1. BEST SHARPE
    # ========================================================

    best_sharpe = results_df.loc[
        results_df["sharpe"].idxmax()
    ].copy()

    # ========================================================
    # 2. BEST CALMAR
    # ========================================================

    valid_calmar = results_df[
        results_df["calmar"].notna()
    ]

    if valid_calmar.empty:

        best_calmar = None

    else:

        best_calmar = valid_calmar.loc[
            valid_calmar["calmar"].idxmax()
        ].copy()

    # ========================================================
    # 3. BEST RETURN / RISK
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
    # ROBUST TOP-10% THRESHOLDS
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
    # TOP-10% THRESHOLD DISTRIBUTION
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
    # TOP-10% SHARPE STATISTICS
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

    # ========================================================
    # 4. ROBUST TOP-10% SHARPE CANDIDATE
    #
    # The candidate thresholds are the median thresholds
    # of the top-10% Sharpe region.
    #
    # We do NOT claim this is itself the Sharpe of the
    # median threshold pair. That will be evaluated later
    # out-of-sample by walk_forward.py.
    # ========================================================

    robust_sharpe = (
        top10_median_sharpe
    )

    # ========================================================
    # Return
    # ========================================================

    return {

        # ----------------------------------------------------
        # BEST SHARPE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BEST CALMAR
        # ----------------------------------------------------

        "calmar_buy_thr":
            (
                best_calmar["buy_thr"]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_sell_thr":
            (
                best_calmar["sell_thr"]
                if best_calmar is not None
                else np.nan
            ),

        "calmar_score":
            (
                best_calmar["calmar"]
                if best_calmar is not None
                else np.nan
            ),

        # ----------------------------------------------------
        # BEST RETURN / RISK
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ROBUST TOP-10%
        # ----------------------------------------------------

        "robust_buy_thr":
            robust_buy_thr,

        "robust_sell_thr":
            robust_sell_thr,

        "robust_sharpe":
            robust_sharpe,

        # ----------------------------------------------------
        # TOP-10% SIZE
        # ----------------------------------------------------

        "top10_n":
            top_n,

        # ----------------------------------------------------
        # TOP-10% THRESHOLD DISTRIBUTION
        # ----------------------------------------------------

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
        # TOP-10% PERFORMANCE
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
        # COMPATIBILITY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ALL VALID SIMULATIONS
        # ----------------------------------------------------

        "all_results":
            results_df,

        # ----------------------------------------------------
        # TOP 10% SIMULATIONS
        # ----------------------------------------------------

        "top10_results":
            top10_results
    }