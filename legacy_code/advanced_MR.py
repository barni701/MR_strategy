import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Strategy Logic (RSI + Adaptive Slope)
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss

    rsi = 100 - (100 / (1 + rs))

    rsi_ema = rsi.ewm(span=period, adjust=False).mean()
    return rsi, rsi_ema

def calculate_tsi(series, short_span=25, long_span=13):
    pc = series.diff()
    abs_pc = pc.abs()

    pc_smooth = pc.ewm(span=long_span, adjust=False).mean()
    pc_double_smooth = pc_smooth.ewm(span=short_span, adjust=False).mean()

    abs_pc_smooth = abs_pc.ewm(span=long_span, adjust=False).mean()
    abs_pc_double_smooth = abs_pc_smooth.ewm(span=short_span, adjust=False).mean()

    tsi = (pc_double_smooth / abs_pc_double_smooth) * 100
    
    signal = tsi.ewm(span=7, adjust=False).mean()
    return tsi, signal

def generate_intraday_signals(df, trend_period=200, tsi_long=25, tsi_short=13):
    """
    Generates Entry AND Exit signals optimized for 5-minute bars.
    """

    data = df.copy()

    if data.index.tz is None:
        times = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        times = df.index.tz_convert('US/Eastern')


    lunch = (times.hour > 11) & (times.hour < 13)
    
    # 1. Calculate Indicators
    data['TSI'], data['TSI_Signal'] = calculate_tsi(data['close'], long_span=tsi_long, short_span=tsi_short)
    data['RSI'], _ = calculate_rsi(data['close'])
    
    # 2. Trend Filter (EMA is better for intraday)
    data['Trend_EMA'] = data['close'].ewm(span=trend_period, adjust=False).mean()
    
    # 3. Volume Filter (20 period average)
    data['Vol_MA'] = data['volume'].rolling(window=20).mean()

    # ---------------------------
    # ENTRY LOGIC (Long Position)
    # ---------------------------
    
    # A. Trend is UP
    trend_condition = data['close'] > data['Trend_EMA']
    
    # B. TSI Momentum Trigger (Crossover)
    # Current TSI > Signal AND Previous TSI < Previous Signal
    tsi_cross_up = (data['TSI'] > data['TSI_Signal']) & (data['TSI'].shift(1) < data['TSI_Signal'].shift(1))
    
    # C. "Value" Condition
    # TSI must be negative (buying a pullback), but not necessarily "extreme" oversold (-25) 
    # because strong intraday trends don't always dip that deep.
    tsi_pullback = data['TSI'] < -15 
    
    # D. RSI Confluence (Not Overbought yet)
    rsi_room_to_grow = data['RSI'] < 60
    
    # E. Volume Confirmation (Relaxed for intraday)
    # Volume must be at least 80% of the average (avoids dead zones)
    vol_valid = data['volume'] > (data['Vol_MA'] * 0.8)

    data['Entry_Signal'] = np.where(
        trend_condition & tsi_cross_up & tsi_pullback & rsi_room_to_grow & vol_valid & (~lunch), 
        1, 0
    )

    # ---------------------------
    # EXIT LOGIC (Take Profit / Stop Momentum)
    # ---------------------------
    
    # Exit Condition 1: TSI Crosses Down (Momentum lost)
    tsi_cross_down = (data['TSI'] < data['TSI_Signal']) & (data['TSI'].shift(1) > data['TSI_Signal'].shift(1))
    
    # Exit Condition 2: RSI Extremity (Overbought)
    # If RSI hits 75 on a 5-min chart, the move is usually exhausted.
    rsi_overbought = data['RSI'] > 75
    
    # Exit Condition 3: Price falls below Trend EMA (Trend Change)
    trend_break = data['close'] < data['Trend_EMA']

    data['Exit_Signal'] = np.where(
        tsi_cross_down | rsi_overbought | trend_break,
        1, 0
    )

    data['Position'] = np.where(data['Entry_Signal'] == 1, 1, 0)
    data['Position'] = np.where(data['Exit_Signal'] == 1, -1, 0)
    data['Postion'] = data['Position'].shift(1).fillna(0)
    
    return data





def signal_generator(df, ema_span=100, trend_span=2000, entry_z=3.0, min_vol_bps=20.0, slope_thresh=0.05, cooldown=10):
    # 1. Indicators
    col_mean = f'EMA_{ema_span}'
    col_std = f'Std_{ema_span}'
    
    df[col_mean] = df['close'].ewm(span=ema_span, adjust=False).mean()
    df[col_std] = df['close'].rolling(window=ema_span).std()
    
    # 2. Z-Scores
    df['Z_Score'] = (df['close'] - df[col_mean]) / df[col_std]
    df['Vol_BPS'] = (df[col_std] / df['close']) * 10000
    
    # 3. RSI (Momentum Filter)
    # Critical for high vol stocks to confirm exhaustion
    df['RSI'] = calculate_rsi(df['close'])
    
    # 4. Regime (Adaptive Slope Hysteresis)
    df['Trend_EMA'] = df['close'].ewm(span=trend_span, adjust=False).mean()
    
    # Calculate Slope (Normalized)
    df['EMA_Slope'] = (df['Trend_EMA'].diff() / df['Trend_EMA']) * 10000
    
    # Hysteresis using DYNAMIC threshold
    conditions = [
        (df['EMA_Slope'] > slope_thresh),
        (df['EMA_Slope'] < -slope_thresh)
    ]
    choices = [1, -1]
    raw_regime = np.select(conditions, choices, default=np.nan)
    df['Regime'] = pd.Series(raw_regime, index=df.index).ffill().fillna(0)
    
    # 5. Time Filters
    if df.index.tz is None:
        times = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        times = df.index.tz_convert('US/Eastern')


    mask_lunch = (times.hour == 12)
    mask_eod = (times.hour == 15) & (times.minute >= 50)
    
    # 6. Execution Levels (Shifted)
    prev_mean = df[col_mean].shift(1)
    prev_std = df[col_std].shift(1)
    
    # Stop is relative to Entry Z.
    # Note: On COIN, stops need to be wide. We use a fixed buffer of 2.0 sigma for now.
    stop_level_z = -(entry_z + 2.0) 
    
    df['Stop_Price_Level'] = prev_mean + (prev_std * stop_level_z)
    # Target is MEAN (0). Don't get greedy on volatile assets.
    df['Target_Price_Level'] = prev_mean 
    
    # 7. Signal Generation
    df['Signal'] = 0 
    
    # ENTRY CONDITION (Long Only)
    long_condition = (
        (df['Z_Score'] < -entry_z) & 
        (df['RSI'] < 30) &               # RSI CONFIRMATION ADDED
        (df['Vol_BPS'] > min_vol_bps) &
        (df['Regime'] == 1) & 
        (~mask_lunch) & 
        (~mask_eod)
    )
    
    # Apply Cooldown
    signal_indices = np.where(long_condition)[0]
    valid_indices = []
    last_trade_idx = -cooldown
    
    for idx in signal_indices:
        if idx - last_trade_idx >= cooldown:
            valid_indices.append(idx)
            last_trade_idx = idx
            
    if valid_indices:
        df.iloc[valid_indices, df.columns.get_loc('Signal')] = 1
    
    # --- EXITS ---
    mask_hit_tp = df['high'] > df['Target_Price_Level']
    mask_hit_stop = df['low'] < df['Stop_Price_Level']
    
    df['Hit_TP'] = mask_hit_tp
    df['Hit_Stop'] = mask_hit_stop
    
    # Forward Fill
    state = np.zeros(len(df))
    holding = False
    
    sig_arr = df['Signal'].values
    tp_arr = mask_hit_tp.values
    stop_arr = mask_hit_stop.values
    eod_arr = np.array(mask_eod) 
    
    for i in range(1, len(df)):
        if holding:
            if eod_arr[i] or tp_arr[i] or stop_arr[i]:
                holding = False
                state[i] = 0
            else:
                state[i] = 1
        else:
            if sig_arr[i] == 1:
                holding = True
                state[i] = 1
    
    df['Signal'] = state
    df['Position'] = df['Signal'].shift(1).fillna(0)
    
    return df

# ==========================================
# 2. Strict Return Calculation
# ==========================================
def calc_robust_returns(df):
    opens = df['open']
    closes = df['close']
    prev_close = closes.shift(1).fillna(opens)
    
    gap_ret = (opens - prev_close) / prev_close
    body_ret = (closes - opens) / opens
    
    mask_stop = df['Hit_Stop']
    stop_prices = df['Stop_Price_Level']
    fill_stop = np.minimum(opens, stop_prices)
    ret_stop = (fill_stop - opens) / opens
    
    mask_tp = df['Hit_TP'] & (~mask_stop)
    target_prices = df['Target_Price_Level']
    fill_tp = np.maximum(opens, target_prices)
    ret_tp = (fill_tp - opens) / opens
    
    effective_body_ret = body_ret.copy()
    effective_body_ret.loc[mask_stop] = ret_stop
    effective_body_ret.loc[mask_tp] = ret_tp
    
    pos_body = df['Position']
    pos_gap = df['Position'].shift(1).fillna(0)
    
    strat_gap = pos_gap * gap_ret
    strat_body = pos_body * effective_body_ret
    
    total_ret = (1 + strat_gap) * (1 + strat_body) - 1
    return total_ret

# ==========================================
# 3. Metrics
# ==========================================
def calculate_metrics(df, strategy_col='Strategy_Return', bh_col='Buy_Hold_Return', cost_bps=1.0):
    metrics = {}
    bars_per_day = df.groupby(df.index.date).size().mode()[0]
    ann_factor = 252 * bars_per_day
    
    trades_count = df['Position'].diff().abs().fillna(0)
    cost_drag = trades_count * (cost_bps * 0.0001)
    
    df['Net_Strategy_Return'] = df[strategy_col] - cost_drag
    
    for name, col in [('Strategy', 'Net_Strategy_Return'), ('Buy & Hold', bh_col)]:
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

def get_trade_stats(df, cost_bps):
    df['Trade_Action'] = df['Position'].diff()
    entries = df[df['Trade_Action'] == 1].index
    exits = df[df['Trade_Action'] == -1].index
    trades = []
    
    if len(exits) > 0 and len(entries) > 0:
        if exits[0] < entries[0]:
            exits = exits[1:]
            
    for entry_dt, exit_dt in zip(entries, exits):
        entry_price = df.loc[entry_dt, 'open']
        hit_stop = df.loc[exit_dt, 'Hit_Stop']
        hit_tp = df.loc[exit_dt, 'Hit_TP']
        
        if hit_stop:
            stop_price = df.loc[exit_dt, 'Stop_Price_Level']
            bar_open = df.loc[exit_dt, 'open']
            exit_price = min(bar_open, stop_price)
            exit_reason = "Stop Loss"
        elif hit_tp:
            target_price = df.loc[exit_dt, 'Target_Price_Level']
            bar_open = df.loc[exit_dt, 'open']
            exit_price = max(bar_open, target_price)
            exit_reason = "Take Profit"
        else:
            exit_price = df.loc[exit_dt, 'open']
            exit_reason = "Signal/EOD"
            
        ret = (exit_price - entry_price) / entry_price
        net_ret = ret - (cost_bps * 2 * 0.0001)
        
        trades.append({
            'Entry_Time': entry_dt, 
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Exit_Reason': exit_reason,
            'Gross_Return_Pct': ret * 100,
            'Net_Return_Pct': net_ret * 100
        })
    return pd.DataFrame(trades)

# ==========================================
# 4. Optimizer (High Volatility Settings)
# ==========================================
def optimize_strategy(train_df, cost_bps):
    best_params = {'ema': 100, 'trend': 2000, 'entry_z': 3.0, 'vol': 30, 'slope': 0.05}
    best_perf = -np.inf
    
    # GRID SEARCH: Geared towards High Beta Stocks
    ema_opts = [50, 100]
    trend_opts = [2000]
    # Look for Deep Crashes
    entry_z_opts = [2.5, 3.0, 3.5]
    # Vol needs to be high to play
    vol_opts = [20, 40]
    # Optimize the Trend Sensitivity (Slope Threshold)
    slope_opts = [0.05, 0.15] # 0.05=Sensitive, 0.15=Requires Strong Trend
    
    for ema in ema_opts:
        for trend in trend_opts:
            for entry in entry_z_opts:
                for vol in vol_opts:
                    for slope in slope_opts:
                        
                        temp_df = train_df.copy()
                        '''temp_df = signal_generator(
                            temp_df, ema, trend, entry, vol, 
                            slope_thresh=slope
                        )'''

                        temp_df = generate_intraday_signals(
                            temp_df, trend_period=trend, 
                            tsi_long=25, tsi_short=13
                        )
                        
                        strat_ret = calc_robust_returns(temp_df)
                        trades = temp_df['Position'].diff().abs().fillna(0)
                        costs = trades * (cost_bps * 0.0001)
                        
                        net_simple = strat_ret - costs
                        net_simple = np.maximum(net_simple, -0.9999)
                        log_wealth = np.log(1 + net_simple).sum()
                        
                        if log_wealth > best_perf:
                            best_perf = log_wealth
                            best_params = {
                                'ema': ema, 'trend': trend, 
                                'entry_z': entry, 'vol': vol, 
                                'slope': slope
                            }
    return best_params

# ==========================================
# 5. WFV Engine
# ==========================================
def walk_forward_backtest(df, train_window_days=90, test_window_days=30, cost_bps=1.0):
    print(f"Starting WFV (High Volatility Mode)...")
    
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    test_results = []
    
    warmup_period = pd.Timedelta(days=30)
    current_date = start_date + pd.Timedelta(days=train_window_days)
    
    while current_date < end_date:
        train_start = current_date - pd.Timedelta(days=train_window_days)
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        
        train_data = df.loc[train_start:current_date].iloc[:-1].copy()
        
        # Optimize with wider, volatility-friendly grid
        if len(train_data) > 2000:
            params = optimize_strategy(train_data, cost_bps)
        else:
            params = {'ema': 100, 'trend': 2000, 'entry_z': 3.0, 'vol': 30, 'slope': 0.05}
        
        buffer_start = current_date - warmup_period
        test_data_buffered = df.loc[buffer_start:test_end].copy()
        
        '''test_data_buffered = signal_generator(
            test_data_buffered, 
            params['ema'], params['trend'], params['entry_z'], 
            params['vol'], slope_thresh=params['slope']
        )'''


        test_data_buffered = generate_intraday_signals(
            test_data_buffered, trend_period=params['trend'], 
            tsi_long=25, tsi_short=13
        )
        
        final_test_data = test_data_buffered.loc[current_date:test_end].copy()
        
        if len(test_results) > 0:
            last_ts = test_results[-1].index[-1]
            final_test_data = final_test_data[final_test_data.index > last_ts]
            
        if final_test_data.empty: break
            
        final_test_data['Opt_Entry_Z'] = params['entry_z']
        test_results.append(final_test_data)
        
        print(f"Period: {current_date.date()} | Entry: {params['entry_z']} | SlopeThresh: {params['slope']}")
        current_date = test_end

    return pd.concat(test_results)

# ==========================================
# 6. Main
# ==========================================
def main():
    FILEPATH = "Data_5min/SPY_5min_2019_2024.parquet" 
    TRADING_COST_BPS = 1.0 
    
    if not os.path.exists(FILEPATH):
        print("Data file not found.")
        return

    print("Loading data...")
    df = pd.read_parquet(FILEPATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    
    df['Buy_Hold_Return'] = df['close'].pct_change().fillna(0)
    df = df.iloc[1:]

    wfv_df = walk_forward_backtest(df, 90, 30, TRADING_COST_BPS)
    
    wfv_df['Strategy_Return'] = calc_robust_returns(wfv_df)
    
    results = calculate_metrics(wfv_df, cost_bps=TRADING_COST_BPS)
    
    print("\n" + "="*40)
    print(f"HIGH VOLATILITY RESULTS")
    print("="*40)
    print(f"Strategy Total: {results['Strategy_Total_Return']:.2f}%")
    print(f"Buy & Hold:     {results['Buy & Hold_Total_Return']:.2f}%")
    print(f"Sharpe Ratio:   {results['Strategy_Sharpe']:.2f}")
    
    trade_df = get_trade_stats(wfv_df, TRADING_COST_BPS)
    if not trade_df.empty:
        print("-" * 40)
        print(f"Total Trades: {len(trade_df)}")
        print(f"Win Rate:     {(len(trade_df[trade_df['Gross_Return_Pct'] > 0]) / len(trade_df) * 100):.2f}%")
        print(f"Avg PnL (Net):{trade_df['Net_Return_Pct'].mean():.3f}%")
        print(f"TP Hits:      {len(trade_df[trade_df['Exit_Reason'] == 'Take Profit'])}")

    wfv_df['Cum_Strat'] = (1 + wfv_df['Net_Strategy_Return']).cumprod()
    wfv_df['Cum_BH'] = (1 + wfv_df['Buy_Hold_Return']).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(wfv_df.index, wfv_df['Cum_Strat'], label='Strategy (Net)', color='green')
    plt.plot(wfv_df.index, wfv_df['Cum_BH'], label='Buy & Hold', color='gray', alpha=0.5)
    plt.title('High Vol Equity Curve (RSI + Adaptive Slope)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()