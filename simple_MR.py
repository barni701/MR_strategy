import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Strategy Logic (Fully Dynamic)
# ==========================================
def signal_generator(df, ema_span=20, trend_span=2000, entry_z=2.0, min_vol_bps=5.0, stop_buffer=2.0):
    '''
    Generates Long-Only signals with Dynamic Volatility, Entry, and Stop Loss.
    '''
    # 1. Calculate Mean (EMA) & Volatility (Std)
    col_mean = f'EMA_{ema_span}'
    col_std = f'Std_{ema_span}'
    
    df[col_mean] = df['close'].ewm(span=ema_span, adjust=False).mean()
    df[col_std] = df['close'].rolling(window=ema_span).std()
    
    # 2. Calculate Z-Score
    df['Z_Score'] = (df['close'] - df[col_mean]) / df[col_std]
    
    # 3. Calculate Relative Volatility in Basis Points
    # (StdDev / Price) * 10,000
    df['Vol_BPS'] = (df[col_std] / df['close']) * 10000
    
    # 4. Calculate Regime Filter (Trend)
    df['Trend_EMA'] = df['close'].ewm(span=trend_span, adjust=False).mean()
    df['Regime'] = np.where(df['close'] > df['Trend_EMA'], 1, -1) 
    
    # 5. Time Filters (US/Eastern)
    if df.index.tz is None:
        times = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        times = df.index.tz_convert('US/Eastern')

    mask_lunch = (times.hour == 12)
    mask_eod = (times.hour == 15) & (times.minute >= 50)
    
    # 6. Generate Signals
    df['Signal'] = np.nan
    
    # --- DYNAMIC ENTRY ---
    long_condition = (
        (df['Z_Score'] < -entry_z) & 
        (df['Vol_BPS'] > min_vol_bps) &
        (df['Regime'] == 1) & 
        (~mask_lunch) & 
        (~mask_eod)
    )
    df.loc[long_condition, 'Signal'] = 1
    
    # --- EXIT (Mean Reversion) ---
    df.loc[(df['Z_Score'] > 0), 'Signal'] = 0
    
    # --- STOP LOSS (Volatility Based) ---
    stop_level = -(entry_z + stop_buffer)
    df.loc[(df['Z_Score'] < stop_level), 'Signal'] = 0 
    
    # 7. Forward Fill (Hold positions)
    df['Signal'] = df['Signal'].ffill().fillna(0)
    
    # 8. Re-Apply Stop Loss & Mean Reversion on Filled Data
    mask_stop = (df['Signal'] == 1) & (df['Z_Score'] < stop_level)
    df.loc[mask_stop, 'Signal'] = 0
    
    mask_profit = (df['Signal'] == 1) & (df['Z_Score'] > 0)
    df.loc[mask_profit, 'Signal'] = 0

    # 9. Apply HARD EOD Exit
    df.loc[mask_eod, 'Signal'] = 0
    
    # 10. Position Management (Shift 1 bar)
    df['Position'] = df['Signal'].shift(1)
    
    return df

# ==========================================
# 2. Performance Metrics & Trade Analysis
# ==========================================
def calculate_metrics(df, strategy_col='Strategy_Return', bh_col='Buy_Hold_Return', cost_bps=4.0):
    metrics = {}
    ann_factor = 252 * 390
    
    trades_count = df['Position'].diff().abs().fillna(0)
    # Cost is subtracted from the SIMPLE return directly
    # 4 bps = 0.0004
    cost_drag = trades_count * (cost_bps * 0.0001)
    
    df['Net_Strategy_Return'] = df[strategy_col] - cost_drag
    
    for name, col in [('Strategy', 'Net_Strategy_Return'), ('Buy & Hold', bh_col)]:
        # CORRECT: Using Simple Returns for cumprod
        cumulative = (1 + df[col]).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100
        
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min() * 100
        
        std_dev = df[col].std()
        if std_dev == 0:
            sharpe = 0
        else:
            sharpe = (df[col].mean() / std_dev) * np.sqrt(ann_factor)
            
        metrics[f'{name}_Total_Return'] = total_return
        metrics[f'{name}_Max_DD'] = max_dd
        metrics[f'{name}_Sharpe'] = sharpe
        
    return metrics

def get_trade_stats(df):
    df['Trade_Action'] = df['Position'].diff()
    entries = df[df['Trade_Action'] == 1].index
    exits = df[df['Trade_Action'] == -1].index
    trades = []
    
    if len(exits) > 0 and len(entries) > 0:
        if exits[0] < entries[0]:
            exits = exits[1:]
            
    for entry_dt, exit_dt in zip(entries, exits):
        entry_price = df.loc[entry_dt, 'close']
        exit_price = df.loc[exit_dt, 'close']
        
        if isinstance(entry_price, pd.Series): entry_price = entry_price.iloc[0]
        if isinstance(exit_price, pd.Series): exit_price = exit_price.iloc[0]
        
        ret = (exit_price - entry_price) / entry_price
        
        trades.append({
            'Entry_Time': entry_dt, 'Entry_Price': entry_price,
            'Exit_Time': exit_dt, 'Exit_Price': exit_price,
            'Return_Pct': ret * 100
        })
        
    return pd.DataFrame(trades)

# ==========================================
# 3. 5-Dimensional Grid Search Optimizer
# ==========================================
def optimize_strategy(train_df, cost_bps):
    best_params = {'ema': 20, 'trend': 2000, 'entry_z': 2.0, 'vol': 5, 'stop': 2.0}
    best_perf = -np.inf
    
    # Optimization Grid
    ema_opts = [10, 30, 50]
    trend_opts = [2000, 4000]
    entry_z_opts = [2.0, 3.0]
    vol_opts = [5, 10, 15]
    stop_opts = [1.0, 3.0]
    
    # Pre-calculate simple returns for speed
    # CORRECT: Using pct_change for simple returns
    train_df['Simple_Ret'] = train_df['close'].pct_change().fillna(0)
    
    for ema in ema_opts:
        for trend in trend_opts:
            for entry in entry_z_opts:
                for vol in vol_opts:
                    for stop in stop_opts:
                        
                        temp_df = train_df.copy()
                        temp_df = signal_generator(
                            temp_df, 
                            ema_span=ema, 
                            trend_span=trend,
                            entry_z=entry,
                            min_vol_bps=vol,
                            stop_buffer=stop
                        )
                        
                        # Calculate Net Profit
                        strat_ret = temp_df['Position'] * temp_df['Simple_Ret']
                        trades = temp_df['Position'].diff().abs().fillna(0)
                        costs = trades * (cost_bps * 0.0001)
                        
                        net_ret = (strat_ret - costs).sum()
                        
                        if net_ret > best_perf:
                            best_perf = net_ret
                            best_params = {
                                'ema': ema, 'trend': trend, 
                                'entry_z': entry, 'vol': vol, 'stop': stop
                            }
            
    return best_params

def walk_forward_backtest(df, train_window_days=90, test_window_days=30, cost_bps=4.0):
    print(f"Starting WFV (Fixed Returns & Overlap Logic)...")
    
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    test_results = []
    
    warmup_period = pd.Timedelta(days=7)
    current_date = start_date + pd.Timedelta(days=train_window_days)
    
    while current_date < end_date:
        train_start = current_date - pd.Timedelta(days=train_window_days)
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        
        train_data = df.loc[train_start:current_date].copy()
        
        if len(train_data) > 2000: 
            params = optimize_strategy(train_data, cost_bps)
        else:
            params = {'ema': 20, 'trend': 2000, 'entry_z': 2.0, 'vol': 5, 'stop': 2.0}
        
        buffer_start = current_date - warmup_period
        test_data_buffered = df.loc[buffer_start:test_end].copy()
        
        test_data_buffered = signal_generator(
            test_data_buffered, 
            ema_span=params['ema'], 
            trend_span=params['trend'],
            entry_z=params['entry_z'],
            min_vol_bps=params['vol'],
            stop_buffer=params['stop']
        )
        
        # Get the slice for this period
        final_test_data = test_data_buffered.loc[current_date:test_end].copy()
        
        # CORRECT: Fix overlap double-counting.
        # If we have previous results, ensure we don't double count the boundary minute.
        if len(test_results) > 0:
            last_timestamp = test_results[-1].index[-1]
            # Keep only rows strictly AFTER the last timestamp we already processed
            final_test_data = final_test_data[final_test_data.index > last_timestamp]
            
        if final_test_data.empty:
            break
            
        final_test_data['Opt_Entry_Z'] = params['entry_z']
        final_test_data['Opt_Min_Vol'] = params['vol']
        test_results.append(final_test_data)
        
        print(f"Period: {current_date.date()} | EntryZ: {params['entry_z']} | MinVol: {params['vol']}bps")
        current_date = test_end

    full_results = pd.concat(test_results)
    return full_results

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    FILEPATH = "Data_adjusted/VZ_1min_2019_2024.parquet"
    TRADING_COST_BPS = 4.0 
    
    if not os.path.exists(FILEPATH):
        print("Data file not found.")
        return

    print("Loading data...")
    df = pd.read_parquet(FILEPATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # CORRECT: Use Simple Returns (Percentage Change) instead of Log Returns
    df['Return'] = df['close'].pct_change().fillna(0)
    
    # Remove the first row which is NaN from pct_change
    df = df.iloc[1:]

    wfv_df = walk_forward_backtest(df, train_window_days=90, test_window_days=30, cost_bps=TRADING_COST_BPS)
    
    # Calculate Strategy Return using Simple Returns
    wfv_df['Strategy_Return'] = wfv_df['Position'] * wfv_df['Return']
    wfv_df['Buy_Hold_Return'] = wfv_df['Return']

    results = calculate_metrics(wfv_df, cost_bps=TRADING_COST_BPS)
    
    print("\n" + "="*40)
    print(f"WALK-FORWARD RESULTS (Cost: {TRADING_COST_BPS} bps)")
    print("="*40)
    print(f"Strategy Total Return: {results['Strategy_Total_Return']:.2f}%")
    print(f"Buy & Hold Return:     {results['Buy & Hold_Total_Return']:.2f}%")
    print(f"Strategy Sharpe Ratio: {results['Strategy_Sharpe']:.2f}")
    
    trade_df = get_trade_stats(wfv_df)
    
    if not trade_df.empty:
        total_trades = len(trade_df)
        win_rate = (len(trade_df[trade_df['Return_Pct'] > 0]) / total_trades) * 100
        avg_pnl = trade_df['Return_Pct'].mean()
        
        print("-" * 40)
        print("TRADE STATISTICS (Gross PnL)")
        print("-" * 40)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate:     {win_rate:.2f}%")
        print(f"Avg PnL/Trade:{avg_pnl:.3f}%")
    
    # Plotting
    wfv_df['Cumulative_Strategy'] = (1 + wfv_df['Net_Strategy_Return']).cumprod()
    wfv_df['Cumulative_BuyHold'] = (1 + wfv_df['Buy_Hold_Return']).cumprod()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=False)

    ax1.plot(wfv_df.index, wfv_df['Cumulative_Strategy'], label='WFV Strategy (Net)', color='green')
    ax1.plot(wfv_df.index, wfv_df['Cumulative_BuyHold'], label='Buy & Hold', color='gray', alpha=0.5)
    ax1.set_title(f'Equity Curve: Net Return')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(wfv_df.index, wfv_df['Opt_Entry_Z'], label='Entry Z', color='purple', marker='.', linestyle='None')
    ax2r = ax2.twinx()
    ax2r.plot(wfv_df.index, wfv_df['Opt_Min_Vol'], label='Min Volatility (bps)', color='orange', marker='.', linestyle='None')
    ax2.set_title('Dynamic Parameters')
    ax2.set_ylabel('Entry Z-Score')
    ax2r.set_ylabel('Min Vol (bps)')
    ax2.grid(True)

    if not trade_df.empty:
        last_trade_idx = trade_df.index[-1]
        last_exit_time = trade_df.loc[last_trade_idx, 'Exit_Time']
        if last_exit_time in wfv_df.index:
            loc_idx = wfv_df.index.get_loc(last_exit_time)
            start_loc = max(0, loc_idx - 2000)
            end_loc = min(len(wfv_df), loc_idx + 500)
            subset = wfv_df.iloc[start_loc:end_loc]
        else:
            subset = wfv_df.iloc[-2000:]
    else:
        subset = wfv_df.iloc[-2000:]
    
    ax3.plot(subset.index, subset['close'], label='Price', color='black', alpha=0.3)
    if 'Trend_EMA' in subset.columns:
        ax3.plot(subset.index, subset['Trend_EMA'], label='Trend Filter', color='blue', linestyle='--', alpha=0.6)

    entries = subset[(subset['Position'] == 1) & (subset['Position'].shift(1) == 0)]
    exits = subset[(subset['Position'] == 0) & (subset['Position'].shift(1) == 1)]
    
    ax3.scatter(entries.index, entries['close'], color='green', marker='^', s=80, label='Buy', zorder=5)
    ax3.scatter(exits.index, exits['close'], color='red', marker='x', s=80, label='Exit', zorder=5)
    ax3.set_title(f'Zoomed View')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()