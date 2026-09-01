# ============================================================
# MPF — EVALUATION FUNCTIONS
# ============================================================

import numpy as np
import pandas as pd


METHODS = {
    "Calmar": "calmar_excess_return",
    "Return/Risk": "return_risk_excess_return",
    "Robust": "robust_excess_return",
    "Sharpe": "sharpe_excess_return",
}


def rank_mpf_configurations(
    df: pd.DataFrame,
    methods: dict = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate and rank MPF configurations.

    Each configuration is defined by:

        (max_num_components, method)

    Four OOS performance criteria are calculated:

        1. Percentage of OOS windows beating benchmark
        2. Mean OOS excess return
        3. Compounded OOS excess return
        4. Sharpe-type ratio:
               mean(excess return) / std(excess return)

    Each criterion is ranked across max_num_components
    separately for each selection method.

    Rank 1 = best.

    The four ranks are then equally weighted to produce
    an overall average rank.

    Parameters
    ----------
    df : pd.DataFrame
        MPF simulation results.

    methods : dict, optional
        Mapping from method name to excess-return column.

    Returns
    -------
    metrics_df : pd.DataFrame
        Calculated performance metrics for every
        (max_num_components, method) combination.

    ranking_df : pd.DataFrame
        Overall ranking of all configurations.
    """

    if methods is None:
        methods = METHODS

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        ["max_num_components"]
        + list(methods.values())
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Calculate the four performance metrics
    # --------------------------------------------------------

    results = []

    for max_components, group in df.groupby(
        "max_num_components"
    ):

        for method, column in methods.items():

            values = group[column].dropna()

            if len(values) == 0:
                continue

            # ------------------------------------------------
            # 1. Percentage of OOS windows beating benchmark
            # ------------------------------------------------

            positive_pct = (
                100 * (values > 0).mean()
            )

            # ------------------------------------------------
            # 2. Mean excess return
            # ------------------------------------------------

            mean_excess = values.mean()

            # ------------------------------------------------
            # 3. Compounded excess return
            # ------------------------------------------------

            compounded_excess = (
                np.prod(1 + values) - 1
            ) * 100

            # ------------------------------------------------
            # 4. Sharpe-type ratio
            # ------------------------------------------------

            sd_excess = values.std(ddof=1)

            if sd_excess != 0:
                sharpe_ratio = (
                    mean_excess / sd_excess
                )
            else:
                sharpe_ratio = np.nan

            results.append({
                "max_num_components": max_components,
                "method": method,
                "positive_pct": positive_pct,
                "mean_excess": mean_excess,
                "compounded_excess": compounded_excess,
                "sharpe_ratio": sharpe_ratio,
            })

    metrics_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Create metric tables
    # --------------------------------------------------------

    positive = metrics_df.pivot(
        index="max_num_components",
        columns="method",
        values="positive_pct",
    )

    mean_excess = metrics_df.pivot(
        index="max_num_components",
        columns="method",
        values="mean_excess",
    )

    compounded = metrics_df.pivot(
        index="max_num_components",
        columns="method",
        values="compounded_excess",
    )

    sharpe = metrics_df.pivot(
        index="max_num_components",
        columns="method",
        values="sharpe_ratio",
    )

    # --------------------------------------------------------
    # Rank each criterion
    #
    # Rank 1 = BEST
    # --------------------------------------------------------

    rank_positive = positive.rank(
        axis=0,
        ascending=False,
        method="average",
    )

    rank_mean = mean_excess.rank(
        axis=0,
        ascending=False,
        method="average",
    )

    rank_compound = compounded.rank(
        axis=0,
        ascending=False,
        method="average",
    )

    rank_sharpe = sharpe.rank(
        axis=0,
        ascending=False,
        method="average",
    )

    # --------------------------------------------------------
    # Equal-weight average rank
    # --------------------------------------------------------

    average_rank = (
        rank_positive
        + rank_mean
        + rank_compound
        + rank_sharpe
    ) / 4

    # --------------------------------------------------------
    # Convert to long format
    # --------------------------------------------------------

    ranking_df = (
        average_rank
        .stack()
        .rename("Average Rank")
        .reset_index()
    )

    # --------------------------------------------------------
    # Global rank
    #
    # Lower average rank = better
    # --------------------------------------------------------

    ranking_df["Overall Rank"] = (
        ranking_df["Average Rank"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranking_df = ranking_df.sort_values(
        [
            "Overall Rank",
            "Average Rank",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Add individual ranks
    #
    # This is useful for research later.
    # --------------------------------------------------------

    individual_ranks = (
        pd.concat(
            {
                "Positive Windows": rank_positive.stack(),
                "Mean Excess": rank_mean.stack(),
                "Compound Excess": rank_compound.stack(),
                "Sharpe": rank_sharpe.stack(),
            },
            axis=1,
        )
        .reset_index()
    )

    individual_ranks.columns = [
        "max_num_components",
        "method",
        "Rank Positive Windows",
        "Rank Mean Excess",
        "Rank Compound Excess",
        "Rank Sharpe",
    ]

    ranking_df = ranking_df.merge(
        individual_ranks,
        on=[
            "max_num_components",
            "method",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    ranking_df = ranking_df[
        [
            "Overall Rank",
            "max_num_components",
            "method",
            "Average Rank",
            "Rank Positive Windows",
            "Rank Mean Excess",
            "Rank Compound Excess",
            "Rank Sharpe",
        ]
    ]

    return metrics_df, ranking_df