import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from model_trainer import train_xgboost_meta_labeler



TICKERS = ["SPY", "AAPL", "DIA", "IWM"]  
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.7
TRADE_DIRECTION = "long_only" 


#INDICATORS
def compute_rsi2_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x['SMA_200'] = x['close'].rolling(window=200).mean()
    x['SMA_5'] = x['close'].rolling(window=5).mean()
    
    delta = x['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()
    rs = avg_gain / avg_loss
    x['RSI_2'] = 100 - (100 / (1 + rs))
    
    x['ret_1'] = x['close'].pct_change()
    x['ret_3'] = x['close'].pct_change(3)
    x['ret_5'] = x['close'].pct_change(5)
    x['oc_ret'] = (x['close'] - x['open']) / x['open']
    x['hl_range'] = (x['high'] - x['low']) / x['open']
    x['recent_vol_20'] = x['ret_1'].rolling(20).std().shift(1)
    x['overnight_ret'] = (x['open'] /x['close'].shift(1)) - 1

    x['SMA200_Dist'] = (x['close'] - x['SMA_200']) / x['SMA_200']
    x['SMA5_Dist'] = (x['close'] - x['SMA_5']) / x['SMA_5']
    return x

def drop_invalid_indicator_rows(df: pd.DataFrame) -> pd.DataFrame:
    req = ['SMA_200', 'SMA_5', 'RSI_2']
    x = df.copy()
    existing = [c for c in req if c in df.columns]
    x = x.dropna(subset=existing)
    return x


#SIGNAL GENERATION

def generate_rsi2_signals(
    df, 
    trade_direction='long_only', 
    rsi_os=10, 
    rsi_ob=90  
):
    data = df.copy()
    above_200 = data['close'] > data['SMA_200']
    below_200 = data['close'] < data['SMA_200']
    oversold = data['RSI_2'] < rsi_os
    overbought = data['RSI_2'] > rsi_ob
    
    long_entry = above_200 & oversold
    short_entry = below_200 & overbought 
    long_exit = data['close'] > data['SMA_5']
    short_exit = data['close'] < data['SMA_5']
    
    data['Trade_Direction'] = np.where(long_entry, 1, np.where(short_entry, -1, 0))

    long_entry_np = long_entry.fillna(False).to_numpy()
    short_entry_np = short_entry.fillna(False).to_numpy()
    long_exit_np = long_exit.fillna(False).to_numpy()
    short_exit_np = short_exit.fillna(False).to_numpy()

    n = len(data)
    pos = np.zeros(n, dtype=np.int8)          
    action = np.zeros(n, dtype=np.int8)       
    entry_sig = np.zeros(n, dtype=np.int8)    
    exit_sig = np.zeros(n, dtype=np.int8)     

    in_pos = 0

    for i in range(n):
        if in_pos == 1:
            if long_exit_np[i]:
                action[i] = -1
                exit_sig[i] = 1
                in_pos = 0
        elif in_pos == -1:
            if short_exit_np[i]:
                action[i] = 1
                exit_sig[i] = 1
                in_pos = 0
        else:
            if trade_direction in ['long_only', 'both'] and long_entry_np[i]:
                action[i] = 1
                entry_sig[i] = 1
                in_pos = 1
            elif trade_direction in ['short_only', 'both'] and short_entry_np[i]:
                action[i] = -1
                entry_sig[i] = 1
                in_pos = -1

        pos[i] = in_pos

    data['Entry_Signal'] = entry_sig
    data['Exit_Signal'] = exit_sig
    data['Signal_Action'] = action
    data['Position_State'] = pos

    return data





def calculate_kelly_position_size(win_prob):
    if pd.isna(win_prob):
        return POSITION_SIZE_PCT

    
    if win_prob < 0.60:
        return 0.0

    return 1.0


def build_trade_database_from_signals(
    df_with_signals: pd.DataFrame,
    feature_cols: list[str],
    cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    df = df_with_signals.copy()

    df['Exec_Entry'] = df['Entry_Signal'].shift(1).fillna(0)
    df['Exec_Exit'] = df['Exit_Signal'].shift(1).fillna(0)
    df['Exec_Direction'] = df['Signal_Action'].shift(1).fillna(0)
    
    entries = df[df['Exec_Entry'] == 1].index
    signal_exits = df[df['Exec_Exit'] == 1].index

    X_rows, y_rows = [], []

    for entry_dt in entries:

        entry_idx = df.index.get_loc(entry_dt)
        if entry_idx - 1 < 0:
            continue

        signal_dt = df.index[entry_idx - 1]  
        direction = df.loc[entry_dt, 'Exec_Direction'] 
        trade = simulate_trade_from_entry(df, entry_dt, signal_exits, cost_bps, direction)

        if trade is None:
            continue

        feats = df.loc[signal_dt, feature_cols]
        if feats.isna().any():
            continue

        X_rows.append({"Signal_Time": signal_dt, "Entry_Time": entry_dt, **feats.to_dict()})
        y_rows.append({"Signal_Time": signal_dt, **trade})

    X_db = pd.DataFrame(X_rows)
    y_db = pd.DataFrame(y_rows)

    if not y_db.empty:
        y_db.insert(0, "trade_id", np.arange(len(y_db)))
        X_db.insert(0, "trade_id", y_db["trade_id"].values)

    return X_db, y_db

def simulate_trade_from_entry(
    df: pd.DataFrame, entry_dt: pd.Timestamp, signal_exits: pd.DatetimeIndex, cost_bps: float, direction: int
) -> dict | None:
    
    entry_idx = df.index.get_loc(entry_dt)
    entry_price = float(df.loc[entry_dt, 'open'])

    next_signal_exits = signal_exits[signal_exits > entry_dt]

    if len(next_signal_exits) == 0:
        return None

    exit_dt_signal = next_signal_exits[0]
    exit_idx_signal = df.index.get_loc(exit_dt_signal)
    exit_price = float(df.iloc[exit_idx_signal]['open'])
    
    if direction == 1:
        gross_ret = (exit_price - entry_price) / entry_price
    else:
        gross_ret = (entry_price - exit_price) / entry_price
        
    net_ret = gross_ret - (2 * cost_bps * 1e-4)

    return {
        'Entry_Time': entry_dt,
        'Exit_Time': exit_dt_signal,
        'Direction': direction,
        'Entry_Price': entry_price,
        'Exit_Price': exit_price,
        'Holding_Bars': int(exit_idx_signal - entry_idx),
        'Exit_Reason': "SMA_5 Signal Exit",
        'Gross_Return': float(gross_ret),
        'Net_Return': float(net_ret),
        'y_good': int(net_ret > 0.003),
    }

#TWO-PASS EXECUTION
def calc_capital_returns_and_stats(df, initial_capital, position_size_pct=POSITION_SIZE_PCT, 
                                   cost_bps=1.0, starting_costs=0.0, use_ml=True):
    df = df.copy()
    df['Cash'] = initial_capital
    df['Shares_Held'] = 0.0
    df['Position_Value'] = 0.0
    df['Cumulative_Costs'] = np.nan
    df.iloc[0, df.columns.get_loc('Cumulative_Costs')] = starting_costs
    
    df['Exec_Entry'] = df['Entry_Signal'].shift(1).fillna(0)
    df['Exec_Exit'] = df['Exit_Signal'].shift(1).fillna(0)
    df['Exec_Direction'] = df['Signal_Action'].shift(1).fillna(0)
    
    entries = df[df['Exec_Entry'] == 1].index
    signal_exits = df[df['Exec_Exit'] == 1].index
    
    cash = initial_capital
    total_costs = starting_costs
    equity_curve = pd.Series(np.nan, index=df.index)
    equity_curve.iloc[0] = initial_capital
    df.iloc[0, df.columns.get_loc('Cash')] = initial_capital
    
    trades = [] 
    if len(entries) == 0:
        df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
        df['Total_Equity'] = initial_capital
        df['Equity_Return'] = 0.0
        return df, pd.DataFrame(trades)

    for entry_dt in entries:
        entry_price = df.loc[entry_dt, 'open']
        entry_idx = df.index.get_loc(entry_dt)
        direction = df.loc[entry_dt, 'Exec_Direction'] 
        
        if cash < (initial_capital * 0.1) or entry_idx == 0:
            break
        
        signal_dt = df.index[entry_idx - 1] 
        


        if use_ml:
            win_prob = df.loc[signal_dt, 'ML_Win_Prob'] if 'ML_Win_Prob' in df.columns else np.nan
            dynamic_size_pct = calculate_kelly_position_size(win_prob)
        else:
            win_prob = np.nan
            dynamic_size_pct = position_size_pct

        trade_sim = simulate_trade_from_entry(df, entry_dt, signal_exits, cost_bps, direction)
        if trade_sim is None:
            continue
            
        is_executed = dynamic_size_pct > 0
        
        if not is_executed:
            trades.append({
                'Entry_Time': entry_dt, 'Exit_Time': trade_sim['Exit_Time'],
                'ML_Win_Prob': win_prob, 'Kelly_Size': 0.0, 'Net_PnL': 0.0,
                'Gross_Return_Pct': trade_sim['Gross_Return'] * 100,
                'Holding_Bars': trade_sim['Holding_Bars'],
                'Is_Executed': False, 'Hypo_Win': trade_sim['y_good']
            })
            continue 

        exit_price = trade_sim['Exit_Price']
        exit_dt_final = trade_sim['Exit_Time']
        exit_idx_final = df.index.get_loc(exit_dt_final)

        margin_allocated = cash * dynamic_size_pct
        shares = margin_allocated / entry_price
        entry_cost = shares * entry_price * (cost_bps * 0.0001)
        
        cash -= (shares * entry_price + entry_cost)
        total_costs += entry_cost

        holding_period_idx = range(entry_idx, exit_idx_final)
        if len(holding_period_idx) > 0:
            holding_closes = df.iloc[holding_period_idx]['close'].values
            equity_curve.iloc[holding_period_idx] = cash + (shares * holding_closes)

        proceeds = shares * exit_price
        exit_cost = proceeds * (cost_bps * 0.0001)
        cash += (proceeds - exit_cost)
        gross_pnl = shares * (exit_price - entry_price)

        total_costs += exit_cost
        equity_curve.iloc[exit_idx_final] = cash
        net_pnl = gross_pnl - entry_cost - exit_cost
        
        trades.append({
            'Entry_Time': entry_dt, 'Exit_Time': exit_dt_final,
            'ML_Win_Prob': win_prob, 'Kelly_Size': dynamic_size_pct, 
            'Entry_Price': entry_price, 'Exit_Price': exit_price,
            'Holding_Bars': trade_sim['Holding_Bars'],
            'Gross_Return_Pct': trade_sim['Gross_Return'] * 100,
            'Portfolio_Impact_Pct': (trade_sim['Net_Return'] * 100) * dynamic_size_pct,
            'Gross_PnL': gross_pnl, 'Net_PnL': net_pnl,
            'Is_Executed': True, 'Hypo_Win': trade_sim['y_good']
        })

    equity_curve = equity_curve.ffill()
    df['Total_Equity'] = equity_curve
    df['Equity_Return'] = df['Total_Equity'].pct_change().fillna(0)
    return df, pd.DataFrame(trades)

#WALK-FORWARD ENGINE

def master_walk_forward_backtest(all_ticker_data, min_train_days=1095, test_window_days=30, cost_bps=1.0,
                                 silo_capital=INITIAL_CAPITAL, feature_cols=None, use_ml=True):
    
    start_date = min(df.index.min() for df in all_ticker_data.values())
    end_date = max(df.index.max() for df in all_ticker_data.values())
    first_exec_date = start_date + pd.Timedelta(days=min_train_days)
    
    # Inject Empty Probability Column
    for t in all_ticker_data.keys():
        all_ticker_data[t] = all_ticker_data[t].copy()
        all_ticker_data[t]['ML_Win_Prob'] = np.nan

    # Pass 1: time travel predictions
    if use_ml:
        current_date = first_exec_date
        while current_date < end_date:
            test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
            
            master_X_list, master_y_list = [], []
            for ticker, df in all_ticker_data.items():
                train_data = df.loc[start_date:current_date].iloc[:-1].copy()
                if len(train_data) > 200:
                    train_data = compute_rsi2_indicators(train_data)
                    train_data = drop_invalid_indicator_rows(train_data)
                    train_data = generate_rsi2_signals(train_data, trade_direction=TRADE_DIRECTION, rsi_os=60, rsi_ob=40)
                    X_db, y_db = build_trade_database_from_signals(train_data, feature_cols, cost_bps)
                    
                    if not X_db.empty:
                        master_X_list.append(X_db[X_db['Trade_Direction'] == 1])
                        master_y_list.append(y_db[y_db['Direction'] == 1])
            
            ml_model = None
            if master_X_list:
                X_long = pd.concat(master_X_list, ignore_index=True)
                y_long = pd.concat(master_y_list, ignore_index=True)
                if len(y_long) >= 30 and len(y_long['y_good'].unique()) > 1:
                    ml_model_long = train_xgboost_meta_labeler(X_db=X_long, y_db=y_long)
                    ml_model = {'long': ml_model_long, 'short': None}

            # Predict and Stamp the OOS Slice
            for ticker, df in all_ticker_data.items():
                buffer_start = current_date - pd.Timedelta(days=300)
                buffered_data = df.loc[buffer_start:test_end].copy()
                if len(buffered_data) < 200: continue
                
                buffered_data = compute_rsi2_indicators(buffered_data)
                buffered_data = drop_invalid_indicator_rows(buffered_data)
                buffered_data = generate_rsi2_signals(buffered_data, trade_direction=TRADE_DIRECTION, rsi_os=30, rsi_ob=70)
                
                oos_data = buffered_data.loc[current_date:test_end]
                signal_dates = oos_data[oos_data['Entry_Signal'] == 1].index
                
                for s_dt in signal_dates:
                    if ml_model is not None:
                        feats = oos_data.loc[s_dt, feature_cols].to_frame().T.astype(float)
                        if not feats.isna().any().any():
                            prob = ml_model['long'].predict_proba(feats)[0, 1]
                            all_ticker_data[ticker].loc[s_dt, 'ML_Win_Prob'] = prob
            
            current_date = test_end

    #Pass 2: Exectuion
    final_equity_curves = {}
    all_trades = []
    
    for ticker, df in all_ticker_data.items():
        # Prepare the entire chronological timeline
        full_df = df.copy()
        full_df = compute_rsi2_indicators(full_df)
        full_df = drop_invalid_indicator_rows(full_df)
        full_df = generate_rsi2_signals(full_df, trade_direction=TRADE_DIRECTION, rsi_os=30, rsi_ob=70)
        

        exec_df = full_df[full_df.index >= first_exec_date].copy()
        if exec_df.empty: continue
        
        exec_df, fold_trades = calc_capital_returns_and_stats(
            exec_df, silo_capital, POSITION_SIZE_PCT, cost_bps, starting_costs=0.0, use_ml=use_ml
        )
        
        final_equity_curves[ticker] = exec_df['Total_Equity']
        if not fold_trades.empty:
            fold_trades['Ticker'] = ticker
            all_trades.append(fold_trades)
            
    final_trade_df = pd.concat(all_trades, axis=0, ignore_index=True) if all_trades else pd.DataFrame()
    return final_equity_curves, final_trade_df


# Monte Carlo SIM.
import numpy as np
import pandas as pd


def _summarise_distribution(x: np.ndarray, as_pct: bool = False) -> dict:
    x = np.asarray(x, dtype=float)
    vals = x * 100.0 if as_pct else x

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "p5": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
        "ci_low_95": float(np.percentile(vals, 2.5)),
        "ci_high_95": float(np.percentile(vals, 97.5)),
    }


def prepare_monte_carlo_daily_return_input(master_equity_series: pd.Series) -> pd.Series:
    if master_equity_series is None or len(master_equity_series) == 0:
        return pd.Series(dtype=float)

    equity = pd.Series(master_equity_series, copy=True).sort_index()
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    if len(equity) < 2:
        return pd.Series(dtype=float)

    daily_returns = equity.pct_change()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    return daily_returns


def generate_monte_carlo_block_starts(
    n_periods: int,
    n_sims: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")

    n_blocks = int(np.ceil(n_periods / block_size))
    max_start = max(n_periods - block_size, 0)
    rng = np.random.default_rng(seed)

    if max_start == 0:
        return np.zeros((n_blocks, n_sims), dtype=int)
    return rng.integers(0, max_start + 1, size=(n_blocks, n_sims), dtype=int)


def monte_carlo_block_bootstrap_from_daily_returns(
    daily_returns: pd.Series,
    initial_capital: float,
    n_sims: int = 2000,
    block_size: int = 5,
    seed: int = 42,
    periods_per_year: int = 252,
    block_starts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if daily_returns is None or len(daily_returns) == 0:
        raise ValueError("daily_returns is empty; cannot run Monte Carlo.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")

    mc_input = pd.Series(daily_returns, copy=True).replace([np.inf, -np.inf], np.nan).dropna()
    returns = mc_input.to_numpy(dtype=float)
    n_periods = len(returns)
    if n_periods < max(20, block_size):
        raise ValueError("Not enough daily returns for daily-return bootstrap.")

    n_blocks = int(np.ceil(n_periods / block_size))
    if block_starts is None:
        block_starts = generate_monte_carlo_block_starts(
            n_periods=n_periods,
            n_sims=n_sims,
            block_size=block_size,
            seed=seed,
        )
    else:
        block_starts = np.asarray(block_starts, dtype=int)
        if block_starts.shape != (n_blocks, n_sims):
            raise ValueError("block_starts has incompatible shape for the return series length.")

    boot_returns = np.empty((n_periods, n_sims), dtype=float)
    paths = np.empty((n_periods + 1, n_sims), dtype=float)
    paths[0, :] = initial_capital

    for s in range(n_sims):
        return_pieces = []
        for start in block_starts[:, s]:
            stop = start + block_size
            return_pieces.append(returns[start:stop])

        sim_returns = np.concatenate(return_pieces)[:n_periods]
        boot_returns[:, s] = sim_returns

    for t in range(n_periods):
        paths[t + 1, :] = paths[t, :] * (1.0 + boot_returns[t, :])

    paths_df = pd.DataFrame(paths)

    final_equity = paths[-1, :]
    total_return = final_equity / initial_capital - 1.0

    running_max = np.maximum.accumulate(paths, axis=0)
    dd = paths / running_max - 1.0
    max_dd = dd.min(axis=0)

    annualized_sharpes = np.empty(n_sims, dtype=float)
    for s in range(n_sims):
        run_rets = boot_returns[:, s]
        finite_rets = run_rets[np.isfinite(run_rets)]
        vol = finite_rets.std(ddof=1) if len(finite_rets) > 1 else 0.0
        annualized_sharpes[s] = (finite_rets.mean() / vol) * np.sqrt(periods_per_year) if vol > 0 else 0.0

    sim_stats_df = pd.DataFrame({
        "final_equity": final_equity,
        "total_return_pct": total_return * 100.0,
        "annualized_sharpe": annualized_sharpes,
        "max_drawdown_pct": max_dd * 100.0,
    })

    summary = {
        "final_equity": _summarise_distribution(final_equity, as_pct=False),
        "total_return_pct": _summarise_distribution(total_return, as_pct=True),
        "annualized_sharpe": _summarise_distribution(annualized_sharpes, as_pct=False),
        "max_drawdown_pct": _summarise_distribution(max_dd, as_pct=True),
        "probability_of_loss": float(np.mean(final_equity < initial_capital)),
        "probability_of_positive_return": float(np.mean(total_return > 0.0)),
        "periods_per_year": float(periods_per_year),
    }

    return paths_df, sim_stats_df, summary


def compare_monte_carlo_strategies(ai_stats_df: pd.DataFrame, baseline_stats_df: pd.DataFrame) -> dict:
    metric_configs = {
        "total_return_pct": {"higher_is_better": True},
        "annualized_sharpe": {"higher_is_better": True},
        "max_drawdown_pct": {"higher_is_better": True},
    }

    comparison = {}
    for metric, config in metric_configs.items():
        ai_vals = ai_stats_df[metric].to_numpy(dtype=float)
        base_vals = baseline_stats_df[metric].to_numpy(dtype=float)
        n = min(len(ai_vals), len(base_vals))
        if n == 0:
            continue

        ai_vals = ai_vals[:n]
        base_vals = base_vals[:n]
        diff = ai_vals - base_vals
        is_better = diff > 0 if config["higher_is_better"] else diff < 0

        comparison[metric] = {
            "improvement": _summarise_distribution(diff, as_pct=False),
            "probability_improvement_gt_0": float(np.mean(is_better)),
            "probability_ai_beats_baseline": float(np.mean(is_better)),
        }

    comparison["probability_ai_beats_baseline"] = comparison.get(
        "total_return_pct", {}
    ).get("probability_ai_beats_baseline", np.nan)

    return comparison


def _format_metric_value(value: float, fmt: str) -> str:
    if fmt == "currency":
        return f"${value:,.2f}"
    if fmt == "percent":
        return f"{value:.2f}%"
    return f"{value:.3f}"


def print_metric_distribution(title: str, stats: dict, fmt: str) -> None:
    print(title + ":")
    print(
        f"  Mean: {_format_metric_value(stats['mean'], fmt)} | "
        f"Median: {_format_metric_value(stats['median'], fmt)} | "
        f"Std: {_format_metric_value(stats['std'], fmt)}"
    )
    print(
        f"  Min: {_format_metric_value(stats['min'], fmt)} | "
        f"Max: {_format_metric_value(stats['max'], fmt)}"
    )
    print(
        f"  P5: {_format_metric_value(stats['p5'], fmt)} | "
        f"P25: {_format_metric_value(stats['p25'], fmt)} | "
        f"P50: {_format_metric_value(stats['p50'], fmt)} | "
        f"P75: {_format_metric_value(stats['p75'], fmt)} | "
        f"P95: {_format_metric_value(stats['p95'], fmt)}"
    )
    print(
        f"  95% CI: [{_format_metric_value(stats['ci_low_95'], fmt)}, "
        f"{_format_metric_value(stats['ci_high_95'], fmt)}]"
    )

def plot_monte_carlo(paths_df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    arr = paths_df.to_numpy()
    plt.figure(figsize=(12, 5))
    k = min(50, arr.shape[1])
    for i in range(k):
        plt.plot(arr[:, i], linewidth=0.8, alpha=0.5)
    plt.axhline(initial_capital, linestyle='--', color='black', linewidth=1)
    plt.title("Monte Carlo Equity Paths", fontsize = 14)
    plt.xlabel("Day #")
    plt.ylabel("Master Portfolio Equity ($)", fontsize = 13)
    plt.tight_layout()
    plt.savefig("monte_carlo.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_dual_distributions(executed_trades: pd.DataFrame, mc_paths_df: pd.DataFrame, initial_capital: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    #Per trade distribution
    trade_profits = executed_trades['Net_PnL'].dropna()
    median_profit = trade_profits.median()
    
    ax1.hist(trade_profits, bins=40, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='-', linewidth=1.5, label='Breakeven ($0)')
    ax1.axvline(x=median_profit, color='green', linestyle='--', linewidth=2, label=f'Median (${median_profit:.0f})')
    ax1.set_title("Distribution of Net Profit/Loss Per Trade", fontsize=12)
    ax1.set_xlabel("Net PnL ($)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Per run distribution
    final_equities = mc_paths_df.iloc[-1].to_numpy()
    p5_equity = np.percentile(final_equities, 5)
    median_equity = np.median(final_equities)
    
    ax2.hist(final_equities, bins=50, color='lightgreen', edgecolor='black')
    ax2.axvline(x=initial_capital, color='red', linestyle='-', linewidth=1.5, label=f'Initial Capital (${initial_capital:,.0f})')
    ax2.axvline(x=p5_equity, color='orange', linestyle='--', linewidth=2, label=f'5th Percentile (${p5_equity:,.0f})')
    ax2.axvline(x=median_equity, color='green', linestyle='-', linewidth=2, label=f'Median (${median_equity:,.0f})')
    ax2.set_title(f"Distribution of Final Portfolio Equity ({len(final_equities):,} MC Runs)", fontsize=12)
    ax2.set_xlabel("Final Portfolio Equity ($)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trade_distribution.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# UNIFIED PORTFOLIO METRICS

def calculate_unified_portfolio_metrics(master_equity_series, initial_capital):
    ann_factor = 252 
    final_equity = master_equity_series.iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    running_max = master_equity_series.cummax()
    drawdown = (master_equity_series / running_max) - 1
    max_dd = drawdown.min() * 100
    
    returns = master_equity_series.pct_change().dropna()
    
    days_in_backtest = (master_equity_series.index[-1] - master_equity_series.index[0]).days
    years = days_in_backtest / 365.25

    cagr_pct = (((final_equity / initial_capital) ** (1 / years)) - 1) * 100 if final_equity > 0 else -100.0

    sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor) if returns.std() > 0 else 0
    downside_std = np.sqrt(np.mean(np.minimum(0, returns)**2))
    sortino = (returns.mean() / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0.0
    calmar = cagr_pct / abs(max_dd) if max_dd < 0 else 0.0

    return {
        'CAGR': cagr_pct,
        'Total_Return': total_return,
        'Final_Equity': final_equity,
        'Max_DD': max_dd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar
    }



def main():
    TRADING_COST_BPS = 1
    SILO_CAPITAL = INITIAL_CAPITAL / len(TICKERS)
    
    print("="*60)
    print("MULTI-TICKER PORTFOLIO: UNIVERSAL ML vs BASELINE")
    print("="*60)
    print(f"Total Master Capital:  ${INITIAL_CAPITAL:,.0f}")
    print(f"Allocated Per Ticker:  ${SILO_CAPITAL:,.0f} ({len(TICKERS)} Tickers)")
    print("="*60)

    FEATURE_COLS = [
        "SMA_200", "SMA_5", "RSI_2",
        "ret_1", "ret_3", "ret_5", "recent_vol_20", "oc_ret", "hl_range",
        "SMA200_Dist", "SMA5_Dist", "Trade_Direction", "overnight_ret"
    ]

    # Load data
    all_ticker_data = {}



    for ticker in TICKERS:
        FILEPATH = f"Data_1D/{ticker}_1d.parquet"
        if not os.path.exists(FILEPATH):
            print(f"Skipping {ticker} - File not found: {FILEPATH}")
            continue
            
        print(f"Loading {ticker}...")
        df = pd.read_parquet(FILEPATH)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        df = df.sort_index()
        df = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        all_ticker_data[ticker] = df

    if not all_ticker_data:
        print("ERROR: No valid data processed.")
        return

    #Run baseline
    print("\nRunning Master Baseline Portfolio...")
    base_wfv_dict, base_trade_df = master_walk_forward_backtest(
        all_ticker_data, 1095, 30, TRADING_COST_BPS, SILO_CAPITAL, feature_cols=FEATURE_COLS, use_ml=False
    )
    base_equity_curves = {t: df for t, df in base_wfv_dict.items() if not df.empty}
    base_master_trade_list = [base_trade_df] if not base_trade_df.empty else []

    # Run ML augmented model
    print("Running Master Universal ML Portfolio...")
    ml_wfv_dict, ml_trade_df = master_walk_forward_backtest(
        all_ticker_data, 1095, 30, TRADING_COST_BPS, SILO_CAPITAL, feature_cols=FEATURE_COLS, use_ml=True
    )
    ml_equity_curves = {t: df for t, df in ml_wfv_dict.items() if not df.empty}
    ml_master_trade_list = [ml_trade_df] if not ml_trade_df.empty else []

    all_base_trades = pd.concat(base_master_trade_list, axis=0, ignore_index=True) if base_master_trade_list else pd.DataFrame()
    all_ml_trades = pd.concat(ml_master_trade_list, axis=0, ignore_index=True) if ml_master_trade_list else pd.DataFrame()



    # Add the profits on different tickers together

    base_port_df = pd.DataFrame(base_equity_curves).ffill().fillna(SILO_CAPITAL)
    base_master_equity = base_port_df.sum(axis=1)
    base_metrics = calculate_unified_portfolio_metrics(base_master_equity, INITIAL_CAPITAL)

    ml_port_df = pd.DataFrame(ml_equity_curves).ffill().fillna(SILO_CAPITAL)
    ml_master_equity = ml_port_df.sum(axis=1)
    ml_metrics = calculate_unified_portfolio_metrics(ml_master_equity, INITIAL_CAPITAL)

    plt.figure(figsize=(12, 6))
    plt.plot(base_master_equity.index, base_master_equity, color='gray', alpha=0.7, label=f"Baseline Portfolio (Return: {base_metrics['Total_Return']:.2f}%)")
    plt.plot(ml_master_equity.index, ml_master_equity, color='blue', linewidth=2.5, label=f"Universal ML Portfolio (Return: {ml_metrics['Total_Return']:.2f}%)")
    plt.axhline(INITIAL_CAPITAL, linestyle='--', color='black', alpha=0.5)
    #plt.title("Master Portfolio Comparison: Baseline vs. Universal ML Dynamic Sizing", fontsize=17)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Master Portfolio Equity ($)", fontsize=17)
    plt.legend(loc="upper left", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Stat summary 
    print("\n" + "="*60)
    print("UNIFIED PORTFOLIO RESULTS: BASELINE vs UNIVERSAL MACHINE LEARNING")
    print("="*60)
    print(f"{'Metric':<25} | {'Baseline RSI(2)':<15} | {'Universal ML RSI(2)':<15}")
    print("-" * 60)
    print(f"{'Final Equity':<25} | ${base_metrics['Final_Equity']:<14,.0f} | ${ml_metrics['Final_Equity']:<14,.0f}")
    print(f"{'Annualized Return':<25} | {base_metrics['CAGR']:<14.2f}% | {ml_metrics['CAGR']:<14.2f}%")
    print(f"{'Total Return':<25} | {base_metrics['Total_Return']:<14.2f}% | {ml_metrics['Total_Return']:<14.2f}%")
    print(f"{'Max Drawdown':<25} | {base_metrics['Max_DD']:<14.2f}% | {ml_metrics['Max_DD']:<14.2f}%")
    print(f"{'Sharpe Ratio':<25} | {base_metrics['Sharpe']:<14.2f}  | {ml_metrics['Sharpe']:<14.2f}")
    print(f"{'Sortino Ratio':<25} | {base_metrics['Sortino']:<14.2f}  | {ml_metrics['Sortino']:<14.2f}")
    print(f"{'Calmar Ratio':<25} | {base_metrics['Calmar']:<14.2f}  | {ml_metrics['Calmar']:<14.2f}")
    print("="*60)

    executed_trades = pd.DataFrame()

    # ML stats 
    if ml_master_trade_list:
        print("\n" + "="*60)
        print("MACHINE LEARNING CLASSIFICATION ACCURACY (PORTFOLIO)")
        print("="*60)
        
        total_signals = len(all_ml_trades)
        executed_mask = all_ml_trades['Is_Executed'] == True
        rejected_mask = all_ml_trades['Is_Executed'] == False
        
        executed_trades = all_ml_trades[executed_mask].copy()
        rejected_trades = all_ml_trades[rejected_mask].copy()
        
        true_positives = len(executed_trades[executed_trades['Hypo_Win'] == 1])
        false_positives = len(executed_trades[executed_trades['Hypo_Win'] == 0])
        true_negatives = len(rejected_trades[rejected_trades['Hypo_Win'] == 0])
        false_negatives = len(rejected_trades[rejected_trades['Hypo_Win'] == 1])
        
        accuracy = (true_positives + true_negatives) / total_signals * 100 if total_signals > 0 else 0
        precision = true_positives / (true_positives + false_positives) * 100 if len(executed_trades) > 0 else 0
        rejected_win_rate = (false_negatives / len(rejected_trades) * 100) if len(rejected_trades) > 0 else 0
        
        print(f"Total OOS Signals Evaluated: {total_signals}")
        print(f"Signals Approved (Executed): {len(executed_trades)}")
        print(f"Signals Rejected (Skipped):  {len(rejected_trades)}")
        print("-" * 60)
        print(f"Overall Portfolio Accuracy:  {accuracy:.2f}%")
        print(f"Model Precision (Win Rate):  {precision:.2f}%")
        print(f"Win Rate of REJECTED Trades: {rejected_win_rate:.2f}%")
        print("-" * 60)
        
        print("Probability Calibration (Deciles):")
        bins = [0, 0.4, 0.5, 0.6, 0.7, 1.0]
        labels = ['<40%', '40-50%', '50-60%', '60-70%', '>70%']
        all_ml_trades['Prob_Bucket'] = pd.cut(all_ml_trades['ML_Win_Prob'], bins=bins, labels=labels)
        
        calib = all_ml_trades.groupby('Prob_Bucket', observed=False).agg(
            Count=('Hypo_Win', 'size'),
            Actual_Win_Rate=('Hypo_Win', lambda x: x.mean() * 100)
        )
        for index, row in calib.iterrows():
            if row['Count'] > 0:
                print(f"  Predicted {index:<6} -> Actual Win Rate: {row['Actual_Win_Rate']:>5.1f}% (Count: {row['Count']})")

        if not executed_trades.empty:
            print("\n" + "="*60)
            print("FINANCIAL TRADE STATISTICS (ML EXECUTED TRADES)")
            print("="*60)
            wins = executed_trades[executed_trades['Net_PnL'] > 0]
            losses = executed_trades[executed_trades['Net_PnL'] <= 0]
            win_rate = (len(wins) / len(executed_trades) * 100)
            
            print(f"Total Trades Taken:     {len(executed_trades):>8} (Across {len(TICKERS)} Tickers)")
            print(f"Overall Win Rate:       {win_rate:>8.2f}%")
            print(f"Winners / Losers:       {len(wins):>4} / {len(losses):>4}")
            print(f"-" * 60)
            print(f"Average Trade (Net):    ${executed_trades['Net_PnL'].mean():>12,.2f}")
            print(f"Average Kelly Size:     {executed_trades['Kelly_Size'].mean():>12.2f}")
            print(f"Max Kelly Size:         {executed_trades['Kelly_Size'].max():>12.2f}")
            print(f"Median Trade (Net):     ${executed_trades['Net_PnL'].median():>12,.2f}")
            print(f"Best Trade (Net):       ${executed_trades['Net_PnL'].max():>12,.2f}")
            print(f"Worst Trade (Net):      ${executed_trades['Net_PnL'].min():>12,.2f}")
            print(f"-" * 60)
            
            if len(wins) > 0:
                print(f"Avg Winner:             ${wins['Net_PnL'].mean():>12,.2f}")
                print(f"Avg Kelly Size (Wins):  {wins['Kelly_Size'].mean():>12.2f}")
            if len(losses) > 0:
                print(f"Avg Loser:              ${losses['Net_PnL'].mean():>12,.2f}")
                print(f"Avg Kelly Size (Losses): {losses['Kelly_Size'].mean():>12.2f}")

            avg_holding = executed_trades['Holding_Bars'].mean()
            print(f"Avg Holding Period:     {avg_holding:>12.1f} Days")
            print("="*60)

    base_mc_returns = prepare_monte_carlo_daily_return_input(base_master_equity)
    ml_mc_returns = prepare_monte_carlo_daily_return_input(ml_master_equity)

    BLOCK_SIZE = 5
    N_SIMS = 2000
    base_mc_stats = None
    ml_mc_stats = None
    base_mc_summary = None
    ml_mc_summary = None
    ml_mc_paths = None
    paired_mc_block_starts = None

    if len(base_mc_returns) >= 20 and len(base_mc_returns) == len(ml_mc_returns):
        paired_mc_block_starts = generate_monte_carlo_block_starts(
            n_periods=len(base_mc_returns),
            n_sims=N_SIMS,
            block_size=BLOCK_SIZE,
            seed=42,
        )

    try:
        if len(base_mc_returns) >= 20:
            _, base_mc_stats, base_mc_summary = monte_carlo_block_bootstrap_from_daily_returns(
                base_mc_returns,
                initial_capital=INITIAL_CAPITAL,
                n_sims=N_SIMS,
                block_size=BLOCK_SIZE,
                seed=42,
                periods_per_year=252,
                block_starts=paired_mc_block_starts
            )

        if len(ml_mc_returns) >= 20:
            ml_mc_paths, ml_mc_stats, ml_mc_summary = monte_carlo_block_bootstrap_from_daily_returns(
                ml_mc_returns,
                initial_capital=INITIAL_CAPITAL,
                n_sims=N_SIMS,
                block_size=BLOCK_SIZE,
                seed=42,
                periods_per_year=252,
                block_starts=paired_mc_block_starts
            )

        if ml_mc_summary is not None:
            print("\n" + "="*60)
            print(f"MONTE CARLO FOR ML PORTFOLIO (DAILY RETURN BLOCK BOOTSTRAP, size={BLOCK_SIZE})")
            print("="*60)
            print(f"Annualized Sharpe uses {ml_mc_summary['periods_per_year']:.0f} trading days/year.")
            print_metric_distribution("Final Equity", ml_mc_summary['final_equity'], "currency")
            print_metric_distribution("Total Return", ml_mc_summary['total_return_pct'], "percent")
            print_metric_distribution("Annualized Sharpe", ml_mc_summary['annualized_sharpe'], "ratio")
            print_metric_distribution("Max Drawdown", ml_mc_summary['max_drawdown_pct'], "percent")
            print(f"Probability of Loss: {ml_mc_summary['probability_of_loss'] * 100:.2f}%")

            if ml_mc_paths is not None:
                plot_monte_carlo(ml_mc_paths, INITIAL_CAPITAL)
            if not executed_trades.empty and ml_mc_paths is not None:
                plot_dual_distributions(executed_trades, ml_mc_paths, INITIAL_CAPITAL)

        if base_mc_summary is not None:
            print("\n" + "="*60)
            print(f"MONTE CARLO FOR BASELINE PORTFOLIO (DAILY RETURN BLOCK BOOTSTRAP, size={BLOCK_SIZE})")
            print("="*60)
            print(f"Annualized Sharpe uses {base_mc_summary['periods_per_year']:.0f} trading days/year.")
            print_metric_distribution("Final Equity", base_mc_summary['final_equity'], "currency")
            print_metric_distribution("Total Return", base_mc_summary['total_return_pct'], "percent")
            print_metric_distribution("Annualized Sharpe", base_mc_summary['annualized_sharpe'], "ratio")
            print_metric_distribution("Max Drawdown", base_mc_summary['max_drawdown_pct'], "percent")
            print(f"Probability of Loss: {base_mc_summary['probability_of_loss'] * 100:.2f}%")

        if ml_mc_stats is not None and base_mc_stats is not None:
            mc_comparison = compare_monte_carlo_strategies(ml_mc_stats, base_mc_stats)

            print("\n" + "="*60)
            print("MONTE CARLO IMPROVEMENT: AI vs BASELINE")
            print("="*60)
            print(
                f"Probability AI beats baseline (Total Return): "
                f"{mc_comparison['probability_ai_beats_baseline'] * 100:.2f}%"
            )

            for metric_name, display_name, fmt in [
                ("total_return_pct", "Total Return Improvement (AI - Baseline)", "percent"),
                ("annualized_sharpe", "Annualized Sharpe Improvement (AI - Baseline)", "ratio"),
                ("max_drawdown_pct", "Max Drawdown Improvement (AI - Baseline)", "percent"),
            ]:
                metric_comp = mc_comparison[metric_name]
                print_metric_distribution(display_name, metric_comp['improvement'], fmt)
                print(f"  Probability Improvement > 0: {metric_comp['probability_improvement_gt_0'] * 100:.2f}%")

    except Exception as e:
        print(f"\nMonte Carlo skipped due to error: {e}")

if __name__ == "__main__":
    main()
