# ============================================================
# MPF — EVALUATION FUNCTIONS
# ============================================================

import pandas as pd


def rank_mpf_results(
    df: pd.DataFrame,
    idr_weights: dict = None,
):
    """
    Calculate:

    1. Classical Ranking
       - P+
       - Mean OOS excess return
       - Compounded OOS excess return
       - Sharpe-type OOS excess return
       - Equal-weight average rank

    2. Instability-Degradation Ranking (IDR)
       - Threshold instability
       - OOS instability
       - Calibration -> OOS degradation
       - OOS excess return
       - Weighted composite rank

    IMPORTANT
    ----------
    All metrics are ranked ACROSS ALL candidate
    (method, max_num_components) configurations.

    Lower rank = better.

    Returns
    -------
    classical_ranking : DataFrame
        Classical ranking of all candidates.

    idr_ranking : DataFrame
        Instability-Degradation ranking of all candidates.

    final_selection : DataFrame
        Best candidate for each method under both
        ranking frameworks.
    """

    # =========================================================
    # INPUT VALIDATION
    # =========================================================

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "df is empty. At least one MPF result is required."
        )

    required_base_columns = [
        "max_num_components",
        "calibration_start",
    ]

    missing_base = [
        column
        for column in required_base_columns
        if column not in df.columns
    ]

    if missing_base:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_base)
        )

    # =========================================================
    # DEFAULT IDR WEIGHTS
    # =========================================================

    if idr_weights is None:
        idr_weights = {
            "threshold": 0.30,
            "oos_instability": 0.30,
            "degradation": 0.25,
            "performance": 0.15,
        }
    else:
        required_weights = {
            "threshold",
            "oos_instability",
            "degradation",
            "performance",
        }

        missing_weights = (
            required_weights
            - set(idr_weights.keys())
        )

        if missing_weights:
            raise ValueError(
                "Missing IDR weights: "
                + ", ".join(sorted(missing_weights))
            )

        try:
            idr_weights = {
                key: float(idr_weights[key])
                for key in required_weights
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All IDR weights must be numeric."
            ) from exc

    if any(
        weight < 0
        for weight in idr_weights.values()
    ):
        raise ValueError(
            "IDR weights must be non-negative."
        )

    weight_sum = sum(idr_weights.values())

    if weight_sum <= 0:
        raise ValueError(
            "The sum of IDR weights must be greater than zero."
        )

    if abs(weight_sum - 1.0) > 1e-10:
        raise ValueError(
            "IDR weights must sum to 1. "
            f"Current sum: {weight_sum:.12g}"
        )

    # =========================================================
    # METHOD DEFINITIONS
    # =========================================================

    methods = {
        "Sharpe": "sharpe",
        "Calmar": "calmar",
        "Return/Risk": "return_risk",
        "Robust": "robust",
    }

    # =========================================================
    # BUILD CANDIDATE-LEVEL DATA
    # =========================================================

    results = []

    for method, prefix in methods.items():

        buy_col = f"{prefix}_buy_thr"
        sell_col = f"{prefix}_sell_thr"

        cal_col = f"{prefix}_calibration_return"
        oos_col = f"{prefix}_oos_return"
        excess_col = f"{prefix}_excess_return"

        required = [
            buy_col,
            sell_col,
            cal_col,
            oos_col,
            excess_col,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns for {method}: "
                + ", ".join(missing)
            )

        # -----------------------------------------------------
        # Validate numeric columns
        # -----------------------------------------------------

        numeric_columns = [
            buy_col,
            sell_col,
            cal_col,
            oos_col,
            excess_col,
            "max_num_components",
        ]

        for column in numeric_columns:
            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                raise TypeError(
                    f"Column '{column}' must be numeric."
                )

        # -----------------------------------------------------
        # Validate calibration_start
        # -----------------------------------------------------

        if df["calibration_start"].isna().all():
            raise ValueError(
                "'calibration_start' contains no valid values."
            )

        for n_components, group in df.groupby(
            "max_num_components",
            dropna=True,
        ):

            group = group.sort_values(
                "calibration_start"
            ).copy()

            buy = group[buy_col].dropna()
            sell = group[sell_col].dropna()

            cal = group[cal_col].dropna()
            oos = group[oos_col].dropna()
            excess = group[excess_col].dropna()

            # -------------------------------------------------
            # Require enough observations
            # -------------------------------------------------

            if len(excess) == 0:
                continue

            # -------------------------------------------------
            # Threshold instability
            # -------------------------------------------------

            threshold_change = (
                (
                    group[buy_col].diff() ** 2
                    +
                    group[sell_col].diff() ** 2
                ) ** 0.5
            ).mean()

            buy_change = buy.diff().abs().mean()
            sell_change = sell.diff().abs().mean()

            # -------------------------------------------------
            # OOS instability
            # -------------------------------------------------

            oos_excess_sd = excess.std(
                ddof=1
            )

            # -------------------------------------------------
            # Calibration -> OOS degradation
            # -------------------------------------------------

            degradation = cal - oos

            degradation_mean = degradation.mean()

            # -------------------------------------------------
            # CLASSICAL OOS METRICS
            # -------------------------------------------------

            # P+
            positive_pct = (
                (excess > 0).mean()
            )

            # Mean excess return
            mean_excess = excess.mean()

            # Compound excess return
            compound_excess = (
                (1 + excess).prod() - 1
            )

            # Sharpe-type excess return
            excess_sd = excess.std(
                ddof=1
            )

            if (
                pd.isna(excess_sd)
                or excess_sd == 0
            ):
                sharpe_excess = float("nan")
            else:
                sharpe_excess = (
                    mean_excess / excess_sd
                )

            results.append({
                "method": method,
                "max_num_components": n_components,

                # ---------------------------------------------
                # Classical metrics
                # ---------------------------------------------

                "P_positive": positive_pct,
                "mean_excess_return": mean_excess,
                "compound_excess_return": compound_excess,
                "sharpe_excess": sharpe_excess,

                # ---------------------------------------------
                # Instability / degradation
                # ---------------------------------------------

                "threshold_instability": threshold_change,
                "buy_thr_mean_abs_change": buy_change,
                "sell_thr_mean_abs_change": sell_change,

                "oos_instability": oos_excess_sd,

                "calibration_return_mean": cal.mean(),
                "oos_return_mean": oos.mean(),

                "degradation": degradation_mean,

                "n_windows": len(group),
            })

    # =========================================================
    # BUILD CANDIDATE DATAFRAME
    # =========================================================

    candidates = pd.DataFrame(results)

    if candidates.empty:
        raise ValueError(
            "No valid MPF candidates could be constructed. "
            "Check the required columns and available data."
        )

    # =========================================================
    # CLASSICAL RANKING
    # =========================================================
    #
    # ALL candidates are ranked together.
    #
    # Higher metric = better
    # Rank 1 = best
    # =========================================================

    candidates["rank_P_positive"] = (
        candidates["P_positive"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_mean_excess"] = (
        candidates["mean_excess_return"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_compound_excess"] = (
        candidates["compound_excess_return"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    candidates["rank_sharpe_excess"] = (
        candidates["sharpe_excess"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # Validate that classical ranking metrics are defined
    # ---------------------------------------------------------

    classical_rank_columns = [
        "rank_P_positive",
        "rank_mean_excess",
        "rank_compound_excess",
        "rank_sharpe_excess",
    ]

    if candidates[classical_rank_columns].isna().all(
        axis=1
    ).any():
        raise ValueError(
            "At least one candidate has no valid classical "
            "ranking metrics."
        )

    # Equal-weight average rank
    candidates["classical_average_rank"] = (
        candidates[
            [
                "rank_P_positive",
                "rank_mean_excess",
                "rank_compound_excess",
                "rank_sharpe_excess",
            ]
        ]
        .mean(axis=1)
    )

    # Overall classical rank
    candidates["classical_rank"] = (
        candidates["classical_average_rank"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # Ensure ranks are integer-valued where defined
    # ---------------------------------------------------------

    candidates["classical_rank"] = (
        candidates["classical_rank"]
        .astype(int)
    )

    classical_ranking = (
        candidates
        .sort_values(
            [
                "classical_average_rank",
                "method",
                "max_num_components",
            ]
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # IDR RANKING
    # =========================================================
    #
    # AGAIN: ALL candidates are ranked together.
    # =========================================================

    # ---------------------------------------------------------
    # Threshold instability
    # Lower = better
    # ---------------------------------------------------------

    candidates["rank_threshold_instability"] = (
        candidates["threshold_instability"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # OOS instability
    # Lower = better
    # ---------------------------------------------------------

    candidates["rank_oos_instability"] = (
        candidates["oos_instability"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # Calibration -> OOS degradation
    # Lower = better
    # ---------------------------------------------------------

    candidates["rank_degradation"] = (
        candidates["degradation"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # OOS performance
    # Higher = better
    # ---------------------------------------------------------

    candidates["rank_oos_performance"] = (
        candidates["mean_excess_return"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # Validate IDR ranking inputs
    # ---------------------------------------------------------

    idr_rank_columns = [
        "rank_threshold_instability",
        "rank_oos_instability",
        "rank_degradation",
        "rank_oos_performance",
    ]

    if candidates[idr_rank_columns].isna().all(
        axis=1
    ).any():
        raise ValueError(
            "At least one candidate has no valid IDR "
            "ranking metrics."
        )

    # ---------------------------------------------------------
    # Weighted IDR score
    # ---------------------------------------------------------

    candidates["idr_score"] = (
        idr_weights["threshold"]
        * candidates["rank_threshold_instability"]

        + idr_weights["oos_instability"]
        * candidates["rank_oos_instability"]

        + idr_weights["degradation"]
        * candidates["rank_degradation"]

        + idr_weights["performance"]
        * candidates["rank_oos_performance"]
    )

    # Overall IDR rank
    candidates["idr_rank"] = (
        candidates["idr_score"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    candidates["idr_rank"] = (
        candidates["idr_rank"]
        .astype(int)
    )

    idr_ranking = (
        candidates
        .sort_values(
            [
                "idr_score",
                "method",
                "max_num_components",
            ]
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # BEST CANDIDATE PER METHOD
    # =========================================================

    classical_best = (
        candidates
        .sort_values(
            [
                "method",
                "classical_average_rank",
                "max_num_components",
            ]
        )
        .groupby(
            "method",
            as_index=False,
        )
        .first()
    )

    classical_best["selection_method"] = (
        "Classical"
    )

    idr_best = (
        candidates
        .sort_values(
            [
                "method",
                "idr_score",
                "max_num_components",
            ]
        )
        .groupby(
            "method",
            as_index=False,
        )
        .first()
    )

    idr_best["selection_method"] = (
        "Instability-Degradation"
    )

    # =========================================================
    # 8 FINAL CONFIGURATIONS
    # =========================================================

    final_selection = pd.concat(
        [
            classical_best,
            idr_best,
        ],
        ignore_index=True,
    )

    final_selection = final_selection[
        [
            "selection_method",
            "method",
            "max_num_components",

            "classical_average_rank",
            "classical_rank",

            "idr_score",
            "idr_rank",

            "P_positive",
            "mean_excess_return",
            "compound_excess_return",
            "sharpe_excess",

            "threshold_instability",
            "oos_instability",
            "degradation",
        ]
    ]

    # ---------------------------------------------------------
    # Final sanity check
    # ---------------------------------------------------------

    expected_methods = set(methods.keys())

    selected_methods = set(
        final_selection["method"]
    )

    if selected_methods != expected_methods:
        raise ValueError(
            "Final selection does not contain all methods. "
            f"Expected {sorted(expected_methods)}, "
            f"got {sorted(selected_methods)}."
        )

    expected_rows = len(methods) * 2

    if len(final_selection) != expected_rows:
        raise ValueError(
            "Final selection should contain exactly "
            f"{expected_rows} configurations "
            f"({len(methods)} methods × 2 ranking frameworks), "
            f"but contains {len(final_selection)}."
        )

    return (
        classical_ranking,
        idr_ranking,
        final_selection,
    )