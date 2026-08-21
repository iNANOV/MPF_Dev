import pandas as pd
import numpy as np

def generate_signals(
    data,
    components,
    column,
    buy_thr,
    sell_thr,
    strategy="mean_reversion",
    membership=None
):
    """
    Generate buy/sell signals with point-in-time membership.

    Components that are not available in `data` are skipped.

    BUY:
        Only allowed if the component is a DJIA member on the
        execution date.

    SELL:
        Always allowed, including after the component has left
        the DJIA.
    """

    signals_list = []
    skipped = []

    for component in components:

        signal_column = column.format(component)

        # --------------------------------------------------
        # Skip components for which data was not downloaded
        # --------------------------------------------------

        if signal_column not in data.columns:
            skipped.append(component)
            continue

        values = data[signal_column]

        # --------------------------------------------------
        # Generate theoretical signals at t
        # --------------------------------------------------

        if strategy == "mean_reversion":

            buy = (
                (values.shift(1) >= buy_thr)
                & (values < buy_thr)
            )

            sell = (
                (values.shift(1) < sell_thr)
                & (values >= sell_thr)
            )

        elif strategy == "momentum":

            buy = (
                (values.shift(1) < buy_thr)
                & (values >= buy_thr)
            )

            sell = (
                (values.shift(1) >= sell_thr)
                & (values < sell_thr)
            )

        else:
            raise ValueError(
                "strategy must be 'mean_reversion' or 'momentum'"
            )

        # --------------------------------------------------
        # Execute signals at t+1
        # --------------------------------------------------

        buy_dates = data.index[
            buy.shift(1, fill_value=False)
        ]

        sell_dates = data.index[
            sell.shift(1, fill_value=False)
        ]

        # --------------------------------------------------
        # Point-in-time membership: BUY only
        # --------------------------------------------------

        if membership is not None:

            member_info = membership[
                membership["ticker"] == component
            ]

            valid_buy_dates = []

            for date in buy_dates:

                is_member = (
                    (member_info["from"] <= date)
                    & (member_info["to"] >= date)
                ).any()

                if is_member:
                    valid_buy_dates.append(date)

            buy_dates = pd.DatetimeIndex(
                valid_buy_dates
            )

        # --------------------------------------------------
        # SELL is NOT membership filtered
        # --------------------------------------------------

        signal_dates = (
            buy_dates
            .union(sell_dates)
            .sort_values()
        )

        signals = pd.DataFrame(index=signal_dates)

        signals["buy"] = signal_dates.isin(buy_dates)
        signals["sell"] = signal_dates.isin(sell_dates)

        signals_list.append(
            (component, signals)
        )

    # ------------------------------------------------------
    # Inform user about unavailable components
    # ------------------------------------------------------

    if skipped:
        print(
            f"Skipped {len(skipped)} components without "
            f"required data: {', '.join(skipped)}"
        )

    return signals_list

def create_trades(data, signals_list):

    trades_list = []

    for component, signals in signals_list:

        trades = []
        buy_date = None

        for date, row in signals.iterrows():

            if buy_date is None and row["buy"]:
                buy_date = date

            elif buy_date is not None and row["sell"]:

                sell_date = date

                buy_price = data.loc[
                    buy_date,
                    component
                ]

                sell_price = data.loc[
                    sell_date,
                    component
                ]

                trades.append({
                    "buy": buy_date,
                    "buy_price": buy_price,
                    "sell": sell_date,
                    "sell_price": sell_price,
                    "duration": (
                        sell_date - buy_date
                    ).days,
                    "return": (
                        sell_price / buy_price - 1
                    )
                })

                buy_date = None

        # -----------------------------------------------------
        # Still-open position
        # -----------------------------------------------------

        if buy_date is not None:

            trades.append({
                "buy": buy_date,
                "buy_price": data.loc[
                    buy_date,
                    component
                ],
                "sell": pd.NaT,
                "sell_price": None,
                "duration": None,
                "return": None
            })

        # -----------------------------------------------------
        # Always create the expected columns
        # -----------------------------------------------------

        trades_df = pd.DataFrame(
            trades,
            columns=[
                "buy",
                "buy_price",
                "sell",
                "sell_price",
                "duration",
                "return"
            ]
        )

        trades_list.append(
            (component, trades_df)
        )

    return trades_list

def create_invest_trades(
    data,
    trade_list,
    max_num_components,
    start_date,
    ranking_column="{}_mean_sd"
):
    """
    Select theoretical trades subject to portfolio constraints.

    Handles:
    - components with no trades
    - empty trade DataFrames
    - missing 'buy' columns
    - missing ranking columns
    - missing ranking values
    - open trades
    - no pd.concat() FutureWarning
    """

    start_date = pd.Timestamp(start_date)

    # ---------------------------------------------------------
    # Expected output columns
    # ---------------------------------------------------------

    empty_columns = [
        "buy",
        "buy_price",
        "sell",
        "sell_price",
        "duration",
        "return",
        "component"
    ]

    # ---------------------------------------------------------
    # 1. Collect all valid theoretical trades
    # ---------------------------------------------------------

    all_trade_records = []

    for component, trades in trade_list:

        # Skip None / empty DataFrames
        if trades is None or trades.empty:
            continue

        # Skip malformed DataFrames
        if "buy" not in trades.columns:
            continue

        trades = trades.copy()

        # Make sure buy is datetime
        trades["buy"] = pd.to_datetime(
            trades["buy"],
            errors="coerce"
        )

        # Remove invalid buy dates
        trades = trades.dropna(
            subset=["buy"]
        )

        # Only trades from start_date onward
        trades = trades[
            trades["buy"] >= start_date
        ].copy()

        if trades.empty:
            continue

        # Add component
        trades["component"] = component

        # -----------------------------------------------------
        # Convert directly to records
        # This avoids pandas concat completely.
        # -----------------------------------------------------

        for record in trades.to_dict("records"):

            # Keep only expected columns
            clean_record = {
                col: record.get(col, pd.NA)
                for col in empty_columns
            }

            all_trade_records.append(
                clean_record
            )

    # ---------------------------------------------------------
    # 2. No trades available
    # ---------------------------------------------------------

    if not all_trade_records:

        return pd.DataFrame(
            columns=empty_columns
        )

    # ---------------------------------------------------------
    # 3. Create one DataFrame directly from records
    # ---------------------------------------------------------

    all_trades = pd.DataFrame(
        all_trade_records,
        columns=empty_columns
    )

    # ---------------------------------------------------------
    # 4. Chronological order
    # ---------------------------------------------------------

    all_trades = (
        all_trades
        .sort_values("buy")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 5. Portfolio selection
    # ---------------------------------------------------------

    invested = set()
    selected = []

    for buy_date, day_trades in all_trades.groupby(
        "buy",
        sort=True
    ):

        # -----------------------------------------------------
        # Remove positions that have already been sold
        # -----------------------------------------------------

        for component in list(invested):

            component_selected = [
                x
                for x in selected
                if x["component"] == component
            ]

            if not component_selected:
                continue

            last_trade = component_selected[-1]

            sell_date = last_trade.get("sell")

            if (
                pd.notna(sell_date)
                and sell_date <= buy_date
            ):
                invested.remove(component)

        # -----------------------------------------------------
        # Available portfolio slots
        # -----------------------------------------------------

        available_slots = (
            max_num_components
            - len(invested)
        )

        if available_slots <= 0:
            continue

        # -----------------------------------------------------
        # Candidates not already invested
        # -----------------------------------------------------

        candidates = day_trades[
            ~day_trades["component"].isin(
                invested
            )
        ].copy()

        if candidates.empty:
            continue

        # -----------------------------------------------------
        # Calculate ranking
        # -----------------------------------------------------

        ranking_values = []

        for _, row in candidates.iterrows():

            component = row["component"]
            date = row["buy"]

            ranking_col = ranking_column.format(
                component
            )

            # Ranking column does not exist
            if ranking_col not in data.columns:

                ranking_values.append(
                    np.nan
                )

                continue

            # Date does not exist
            if date not in data.index:

                ranking_values.append(
                    np.nan
                )

                continue

            value = data.loc[
                date,
                ranking_col
            ]

            # Protect against duplicate index
            if isinstance(value, pd.Series):
                value = value.iloc[0]

            ranking_values.append(
                value
            )

        candidates["_ranking"] = ranking_values

        # -----------------------------------------------------
        # Remove candidates without ranking
        # -----------------------------------------------------

        candidates = candidates[
            candidates["_ranking"].notna()
        ].copy()

        if candidates.empty:
            continue

        # -----------------------------------------------------
        # Highest mu/sigma first
        # -----------------------------------------------------

        candidates = candidates.sort_values(
            "_ranking",
            ascending=False
        )

        # -----------------------------------------------------
        # Select best available components
        # -----------------------------------------------------

        selected_candidates = (
            candidates
            .head(available_slots)
            .drop(columns="_ranking")
        )

        selected.extend(
            selected_candidates.to_dict(
                "records"
            )
        )

        invested.update(
            selected_candidates[
                "component"
            ]
        )

    # ---------------------------------------------------------
    # 6. No selected trades
    # ---------------------------------------------------------

    if not selected:

        return pd.DataFrame(
            columns=empty_columns
        )

    # ---------------------------------------------------------
    # 7. Final DataFrame
    # ---------------------------------------------------------

    invest_trades = pd.DataFrame(
        selected,
        columns=empty_columns
    )

    return (
        invest_trades
        .sort_values("buy")
        .reset_index(drop=True)
    )

def simulate_portfolio(
    data,
    invest_trades,
    initial_capital=100_000,
    abs_cost_for_a_trade=0,
    percent_cost_for_a_trade=0,
    max_investment_size_in_percent=50,
    min_cash_in_percent=10
):
    """
    Simulate a fully invested portfolio based on selected trades.

    Parameters
    ----------
    data : pd.DataFrame
        Daily price data. Columns must contain the component names.

    invest_trades : pd.DataFrame
        Filtered trades with columns:
        buy, buy_price, sell, sell_price, duration, return, component

    initial_capital : float
        Starting cash.

    abs_cost_for_a_trade : float
        Fixed transaction cost per BUY or SELL.

    percent_cost_for_a_trade : float
        Transaction cost as a fraction of trade value.
        Example: 0.001 = 0.1%.

    max_investment_size_in_percent : float
        Maximum percentage of portfolio value invested in one component.

    min_cash_in_percent : float
        Minimum percentage of portfolio value that must remain as cash.

    Returns
    -------
    pd.DataFrame
        Daily portfolio state.
    """

    # ------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------

    if invest_trades.empty:
        raise ValueError("invest_trades is empty.")

    required_columns = {
        "buy",
        "sell",
        "component"
    }

    missing = required_columns - set(invest_trades.columns)

    if missing:
        raise ValueError(
            f"Missing columns in invest_trades: {missing}"
        )

    # ------------------------------------------------------
    # Simulation period
    # ------------------------------------------------------

    simulation_start = invest_trades["buy"].min()

    data = data.loc[
        data.index >= simulation_start
    ].copy()

    # ------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------

    cash = initial_capital
    total_costs = 0.0

    # Current number of shares for each component
    positions = {}

    results = []

    # ------------------------------------------------------
    # Daily simulation
    # ------------------------------------------------------

    for date in data.index:

        # Trades occurring today
        day_trades = invest_trades[
            (invest_trades["buy"] == date) |
            (invest_trades["sell"] == date)
        ]

        daily_cost = 0.0
        daily_actions = []

        # ==================================================
        # 1. SELL FIRST
        # ==================================================

        for _, trade in day_trades.iterrows():

            component = trade["component"]

            if (
                pd.notna(trade["sell"])
                and trade["sell"] == date
            ):

                if component not in positions:
                    continue

                shares = positions[component]

                if shares <= 0:
                    continue

                price = data.loc[date, component]

                trade_value = shares * price

                cost = (
                    abs_cost_for_a_trade
                    + trade_value * percent_cost_for_a_trade
                )

                cash += trade_value - cost

                total_costs += cost
                daily_cost += cost

                positions[component] = 0

                daily_actions.append({
                    "component": component,
                    "action": "SELL",
                    "shares": shares,
                    "price": price,
                    "trade_value": trade_value,
                    "cost": cost
                })

        # ==================================================
        # 2. CURRENT PORTFOLIO VALUE AFTER SELLS
        # ==================================================

        investment_value = sum(
            shares * data.loc[date, component]
            for component, shares in positions.items()
            if shares > 0
        )

        portfolio_value = cash + investment_value

        # ==================================================
        # 3. BUY
        # ==================================================

        for _, trade in day_trades.iterrows():

            component = trade["component"]

            if (
                pd.notna(trade["buy"])
                and trade["buy"] == date
            ):

                price = data.loc[date, component]

                # ------------------------------------------
                # Maximum allowed size of this position
                # ------------------------------------------

                max_position_value = (
                    portfolio_value
                    * max_investment_size_in_percent
                    / 100
                )

                # ------------------------------------------
                # Minimum cash reserve
                # ------------------------------------------

                min_cash = (
                    portfolio_value
                    * min_cash_in_percent
                    / 100
                )

                available_cash = max(
                    0,
                    cash - min_cash
                )

                # ------------------------------------------
                # Amount we can invest
                # ------------------------------------------

                investment_amount = min(
                    max_position_value,
                    available_cash
                )

                # Whole shares only
                shares = int(
                    investment_amount / price
                )

                if shares <= 0:
                    continue

                trade_value = shares * price

                cost = (
                    abs_cost_for_a_trade
                    + trade_value * percent_cost_for_a_trade
                )

                # Make sure transaction fits into cash
                if trade_value + cost > available_cash:
                    continue

                cash -= trade_value + cost

                total_costs += cost
                daily_cost += cost

                positions[component] = shares

                daily_actions.append({
                    "component": component,
                    "action": "BUY",
                    "shares": shares,
                    "price": price,
                    "trade_value": trade_value,
                    "cost": cost
                })

        # ==================================================
        # 4. END-OF-DAY PORTFOLIO VALUE
        # ==================================================

        investment_value = sum(
            shares * data.loc[date, component]
            for component, shares in positions.items()
            if shares > 0
        )

        portfolio_value = cash + investment_value

        net_profit = (
            portfolio_value - initial_capital
        )

        portfolio_return = (
            net_profit / initial_capital
        )

        # ==================================================
        # 5. DAILY STATE
        # ==================================================

        result = {
            "date": date,
            "cash": cash,
            "investment_value": investment_value,
            "portfolio_value": portfolio_value,
            "initial_capital": initial_capital,
            "cost": daily_cost,
            "total_costs": total_costs,
            "net_profit": net_profit,
            "return": portfolio_return,
            "n_positions": sum(
                shares > 0
                for shares in positions.values()
            )
        }

        # --------------------------------------------------
        # Store today's transactions
        # --------------------------------------------------

        result["action"] = ",".join(
            x["action"]
            for x in daily_actions
        )

        result["component"] = ",".join(
            x["component"]
            for x in daily_actions
        )

        result["shares"] = ",".join(
            str(x["shares"])
            for x in daily_actions
        )

        result["price"] = ",".join(
            f"{x['price']:.6f}"
            for x in daily_actions
        )

        result["trade_value"] = sum(
            x["trade_value"]
            for x in daily_actions
        )

        results.append(result)

    # ------------------------------------------------------
    # Final DataFrame
    # ------------------------------------------------------

    portfolio = pd.DataFrame(results)

    portfolio = portfolio.set_index("date")

    return portfolio

def run_strategy(
    data,
    components,
    ma_column,
    buy_thr,
    sell_thr,
    strategy="mean_reversion",
    membership=None,
    max_num_components=2,
    start_date="2010-01-01",
    ranking_column="{}_mean_sd",
    initial_capital=100_000,
    abs_cost_for_a_trade=5,
    percent_cost_for_a_trade=0.001,
    max_investment_size_in_percent=50,
    min_cash_in_percent=10
):

    # ---------------------------------------------
    # 1. Generate membership-aware signals
    # ---------------------------------------------

    signals_list = generate_signals(
        data,
        components,
        ma_column,
        buy_thr=buy_thr,
        sell_thr=sell_thr,
        strategy=strategy,
        membership=membership
    )

    # ---------------------------------------------
    # 2. Convert signals to theoretical trades
    # ---------------------------------------------

    trades_list = create_trades(
        data,
        signals_list
    )

    # ---------------------------------------------
    # 3. Apply portfolio selection rules
    # ---------------------------------------------

    invest_trades = create_invest_trades(
        data,
        trades_list,
        max_num_components=max_num_components,
        start_date=start_date,
        ranking_column=ranking_column
    )

    # ---------------------------------------------
    # 4. Simulate portfolio
    # ---------------------------------------------

    portfolio = simulate_portfolio(
        data,
        invest_trades,
        initial_capital=initial_capital,
        abs_cost_for_a_trade=abs_cost_for_a_trade,
        percent_cost_for_a_trade=percent_cost_for_a_trade,
        max_investment_size_in_percent=max_investment_size_in_percent,
        min_cash_in_percent=min_cash_in_percent
    )

    return {
        "signals": signals_list,
        "trades": trades_list,
        "invest_trades": invest_trades,
        "portfolio": portfolio
    }
