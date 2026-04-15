import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================

TICKER = "SPY"


RTH_START = "09:30"
RTH_END = "16:00" 
LUNCH_START = "11:00"
LUNCH_END = "13:00"
EOD_CUTOFF = "15:45"
EOD_EXIT = "15:55"

TP_PCT = 0.008  # Take profit level (0.5%)
SL_PCT = 0.003  # Stop loss level (0.3%)
INITIAL_CAPITAL = 100000.0  # Starting capital ($100k)
POSITION_SIZE_PCT = 0.90  # Use 90% of capital per trade


def prepare_data(df, source_tz='UTC', target_tz='US/Eastern'):
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    if data.index.tz is None:
        data.index = data.index.tz_localize(source_tz)
    data.index = data.index.tz_convert(target_tz)
    times = data.index.time
    rth_mask = (times >= pd.Timestamp(RTH_START).time()) & (times < pd.Timestamp(RTH_END).time())
    data = data[rth_mask].copy()
    data = data[~data.index.duplicated(keep='first')]
    return data

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi_ema = rsi.ewm(span=period, adjust=False).mean()
    return rsi, rsi_ema

def calculate_tsi(series, long_span=25, short_span=13, signal_span=7):
    pc = series.diff()
    abs_pc = pc.abs()
    pc_smooth = pc.ewm(span=long_span, adjust=False).mean()
    pc_double_smooth = pc_smooth.ewm(span=short_span, adjust=False).mean()
    abs_pc_smooth = abs_pc.ewm(span=long_span, adjust=False).mean()
    abs_pc_double_smooth = abs_pc_smooth.ewm(span=short_span, adjust=False).mean()
    tsi = (pc_double_smooth / abs_pc_double_smooth.replace(0, np.nan)) * 100
    tsi = tsi.fillna(0)
    signal = tsi.ewm(span=signal_span, adjust=False).mean()
    return tsi, signal


def add_boilinger_bands(df, period = 20, std =2 ):

    df['BB_MID'] = df['close'].rolling(window=period).mean()
    df['BB_STD'] = df['close'].rolling(window=period).std()
    df['BB_UPPER'] = df['BB_MID'] + (std * df['BB_STD'])
    df['BB_LOWER'] = df['BB_MID'] - (std * df['BB_STD'])
    return df




def generate_intraday_signals(df, trend_period=200, tsi_long=25, tsi_short=13, 
                               reset_position=True, start_flat=True):
    data = df.copy()
    times = data.index.time
    lunch = (times >= pd.Timestamp(LUNCH_START).time()) & (times < pd.Timestamp(LUNCH_END).time())
    eod_cutoff = times >= pd.Timestamp(EOD_CUTOFF).time()
    eod_exit = times >= pd.Timestamp(EOD_EXIT).time()
    
    # Entry conditions
    trend_condition = data['close'] > data['Trend_EMA'] *1.002
    tsi_cross_up = (data['TSI'] > data['TSI_Signal']) & \
                   (data['TSI'].shift(1) <= data['TSI_Signal'].shift(1))
    tsi_pullback = data['TSI'] < -5
    rsi_room_to_grow = data['RSI'] < 45
    vol_valid = data['volume'] > (data['Vol_MA'] * 0.9)
    
    data['Entry_Signal'] = np.where(
        trend_condition & tsi_cross_up & tsi_pullback & rsi_room_to_grow & 
        vol_valid & (~lunch) & (~eod_cutoff),
        1, 0
    )
    
    # Exit conditions
    tsi_cross_down = (data['TSI'] < data['TSI_Signal']) & \
                     (data['TSI'].shift(1) >= data['TSI_Signal'].shift(1))
    rsi_overbought = data['RSI'] > 60
    tsi_reverted = data['TSI'] > 5
    trend_break = (data['close'] < data['Trend_EMA']) & (data['close'].shift(1) < data['Trend_EMA'].shift(1))
    
    data['Exit_Signal'] = np.where(
        (trend_break | eod_exit) & (~lunch),
        1, 0
    )
    data.loc[eod_exit, 'Exit_Signal'] = 1
    
    # Signal action
    data['Signal_Action'] = 0
    data.loc[data['Exit_Signal'] == 1, 'Signal_Action'] = -1
    data.loc[(data['Entry_Signal'] == 1) & (data['Signal_Action'] == 0), 'Signal_Action'] = 1
    
    data['TP_Hit'] = False
    data['SL_Hit'] = False
    data['Entry_Price'] = np.nan
    
    # Build position state
    if start_flat:
        data['Position_State'] = 0
        in_position = False
        entry_price = 0.0
        
        pos_idx = data.columns.get_loc('Position_State')
        tp_idx = data.columns.get_loc('TP_Hit')
        sl_idx = data.columns.get_loc('SL_Hit')
        ref_idx = data.columns.get_loc('Entry_Price')
        
        for i in range(len(data)):
            action = data.iloc[i]['Signal_Action']
            
            if in_position:
                if entry_price == 0.0:
                    entry_price = data.iloc[i]['open']
                
                data.iloc[i, ref_idx] = entry_price
                current_high = data.iloc[i]['high']
                current_low = data.iloc[i]['low']
                
                # Check stop loss first
                if current_low <= entry_price * (1 - SL_PCT):
                    in_position = False
                    data.iloc[i, pos_idx] = 0
                    data.iloc[i, sl_idx] = True
                    entry_price = 0.0
                    continue
                
                # Check take profit
                if current_high >= entry_price * (1 + TP_PCT):
                    in_position = False
                    data.iloc[i, pos_idx] = 0
                    data.iloc[i, tp_idx] = True
                    entry_price = 0.0
                    continue
                
                if action == -1:
                    in_position = False
                    data.iloc[i, pos_idx] = 0
                    entry_price = 0.0
                else:
                    data.iloc[i, pos_idx] = 1
            else:
                if action == 1:
                    in_position = True
                    entry_price = 0.0
                    data.iloc[i, pos_idx] = 1
                    data.iloc[i, ref_idx] = np.nan
                else:
                    data.iloc[i, pos_idx] = 0
    else:
        position_changes = data['Signal_Action'].replace(0, np.nan)
        data['Position_State'] = position_changes.map({1: 1, -1: 0}).ffill().fillna(0)
    
    data['Position_Next'] = data['Position_State'].shift(1).fillna(0)
    return data

# ==========================================
# NEW: Capital-Based Return Calculation
# ==========================================
def calc_capital_returns(df, initial_capital=INITIAL_CAPITAL, position_size_pct=POSITION_SIZE_PCT, 
                         cost_bps=2.0, starting_costs=0.0):
    df = df.copy()
    
    # 1. Initialize columns
    df['Cash'] = initial_capital
    df['Shares_Held'] = 0.0
    df['Position_Value'] = 0.0
    
    # FIX START: Initialize Cumulative_Costs with NaN so we can ffill later
    df['Cumulative_Costs'] = np.nan
    # Set the first value so the ffill has a starting point
    df.iloc[0, df.columns.get_loc('Cumulative_Costs')] = starting_costs
    # FIX END
    
    df['Position_Change'] = df['Position_Next'].diff()
    entries = df[df['Position_Change'] == 1].index
    exits = df[df['Position_Change'] == -1].index
    
    cash = initial_capital
    total_costs = starting_costs
    
    equity_curve = pd.Series(initial_capital, index=df.index)
    
    if len(exits) > 0 and len(entries) > 0:
        if exits[0] < entries[0]: exits = exits[1:]
    
    if len(entries) == 0:
        # No trades? Just fill the costs down and return
        df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill() # Propagate starting_costs
        df['Total_Equity'] = initial_capital
        df['Equity_Return'] = 0.0
        df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)
        return df
    
    for i, entry_dt in enumerate(entries):
        entry_price = df.loc[entry_dt, 'open']
        
        if cash < 100: break
        
        shares = (cash * position_size_pct) / entry_price
        entry_cost = shares * entry_price * (cost_bps * 0.0001)
        
        cash -= (shares * entry_price + entry_cost)
        total_costs += entry_cost
        
        if i < len(exits):
            exit_dt = exits[i]
            entry_idx = df.index.get_loc(entry_dt)
            exit_idx = df.index.get_loc(exit_dt)
            
            holding_period_idx = range(entry_idx, exit_idx)
            holding_closes = df.iloc[holding_period_idx]['close'].values
            equity_curve.iloc[holding_period_idx] = cash + (shares * holding_closes)
            
            df.iloc[entry_idx:exit_idx, df.columns.get_loc('Cash')] = cash
            df.iloc[entry_idx:exit_idx, df.columns.get_loc('Shares_Held')] = shares
            df.iloc[entry_idx:exit_idx, df.columns.get_loc('Position_Value')] = shares * holding_closes
            
            # Record costs during the trade
            df.iloc[entry_idx:exit_idx, df.columns.get_loc('Cumulative_Costs')] = total_costs
            
            exit_price = df.loc[exit_dt, 'open']
            prev_idx = exit_idx - 1
            if prev_idx >= 0 and df.iloc[prev_idx]['TP_Hit']:
                exit_price = entry_price * (1 + TP_PCT)
            
            proceeds = shares * exit_price
            exit_cost = proceeds * (cost_bps * 0.0001)
            
            cash += (proceeds - exit_cost)
            total_costs += exit_cost
            
            equity_curve.iloc[exit_idx] = cash
            df.iloc[exit_idx, df.columns.get_loc('Cash')] = cash
            df.iloc[exit_idx, df.columns.get_loc('Shares_Held')] = 0.0
            df.iloc[exit_idx, df.columns.get_loc('Position_Value')] = 0.0
            
            # Record final cost after exit
            df.iloc[exit_idx, df.columns.get_loc('Cumulative_Costs')] = total_costs
            
        else:
            entry_idx = df.index.get_loc(entry_dt)
            remaining_idx = range(entry_idx, len(df))
            remaining_closes = df.iloc[remaining_idx]['close'].values
            equity_curve.iloc[remaining_idx] = cash + (shares * remaining_closes)
            
            df.iloc[entry_idx:, df.columns.get_loc('Cash')] = cash
            df.iloc[entry_idx:, df.columns.get_loc('Shares_Held')] = shares
            df.iloc[entry_idx:, df.columns.get_loc('Position_Value')] = shares * remaining_closes
            df.iloc[entry_idx:, df.columns.get_loc('Cumulative_Costs')] = total_costs
    
    # FIX START: Forward fill costs to cover flat periods between trades
    df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
    # FIX END
    
    flat_periods = df['Position_Next'] == 0
    equity_curve[flat_periods] = cash 
    
    df['Total_Equity'] = equity_curve
    df['Equity_Return'] = df['Total_Equity'].pct_change().fillna(0)
    df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)
    
    return df
# ==========================================
# NEW: Capital-Based Metrics
# ==========================================
def calculate_capital_metrics(df, initial_capital=INITIAL_CAPITAL):
    """
    Calculate metrics for capital-based system.
    """
    metrics = {}
    
    # Calculate bars per day
    bars_per_day_series = df.groupby(df.index.date).size()
    bars_per_day = int(bars_per_day_series.mode()[0]) if len(bars_per_day_series) > 0 else 78
    ann_factor = 252 * bars_per_day
    
    # Strategy metrics
    final_equity = df['Total_Equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # Drawdown
    running_max = df['Total_Equity'].cummax()
    drawdown = (df['Total_Equity'] / running_max) - 1
    max_dd = drawdown.min() * 100
    
    # Sharpe (using equity returns)
    equity_returns = df['Equity_Return'].dropna()
    if len(equity_returns) > 0 and equity_returns.std() > 0:
        sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0
    
    # Buy & Hold metrics (for comparison)
    bh_final = initial_capital * (df['open'].iloc[-1] / df['open'].iloc[0])
    bh_return = (bh_final - initial_capital) / initial_capital * 100
    
    bh_equity = initial_capital * (df['open'] / df['open'].iloc[0])
    bh_running_max = bh_equity.cummax()
    bh_drawdown = (bh_equity / bh_running_max) - 1
    bh_max_dd = bh_drawdown.min() * 100
    
    bh_returns = df['Buy_Hold_Return'].dropna()
    if len(bh_returns) > 0 and bh_returns.std() > 0:
        bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(ann_factor)
    else:
        bh_sharpe = 0
    
    metrics['Strategy_Total_Return'] = total_return
    metrics['Strategy_Final_Equity'] = final_equity
    metrics['Strategy_Max_DD'] = max_dd
    metrics['Strategy_Sharpe'] = sharpe
    metrics['Total_Costs'] = df['Cumulative_Costs'].iloc[-1]
    
    metrics['Buy_Hold_Total_Return'] = bh_return
    metrics['Buy_Hold_Final_Equity'] = bh_final
    metrics['Buy_Hold_Max_DD'] = bh_max_dd
    metrics['Buy_Hold_Sharpe'] = bh_sharpe
    
    return metrics, df

# ==========================================
# NEW: Capital-Based Trade Stats
# ==========================================
def get_capital_trade_stats(df, cost_bps):
    """
    Reconstruct trades with dollar P&L.
    """
    df = df.copy()
    
    df['Position_Change'] = df['Position_Next'].diff()
    entries = df[df['Position_Change'] == 1].index
    exits = df[df['Position_Change'] == -1].index
    
    trades = []
    
    if len(exits) > 0 and len(entries) > 0:
        if exits[0] < entries[0]:
            exits = exits[1:]
    
    for i, entry_dt in enumerate(entries):
        entry_price = df.loc[entry_dt, 'open']
        entry_cash = df.loc[entry_dt, 'Cash']
        
        # Get shares from next bar (when position is actually held)
        if entry_dt == df.index[-1]:
            shares = 0
        else:
            next_idx = df.index.get_loc(entry_dt) + 1
            shares = df.iloc[next_idx]['Shares_Held']
        
        if i < len(exits):
            exit_dt = exits[i]
            exit_price = df.loc[exit_dt, 'open']
            
            # Check exit reason
            prev_idx = df.index.get_loc(exit_dt) - 1
            if prev_idx >= 0:
                prev_bar = df.index[prev_idx]
                
                if df.loc[prev_bar, 'low'] <= entry_price * (1 - SL_PCT):
                    exit_reason = "Stop Loss"
                    exit_price = entry_price * (1 - SL_PCT)
                elif df.loc[prev_bar, 'high'] >= entry_price * (1 + TP_PCT):
                    exit_reason = "Take Profit"
                    exit_price = entry_price * (1 + TP_PCT)
                elif df.loc[prev_bar].name.time() >= pd.Timestamp(EOD_EXIT).time():
                    exit_reason = "EOD Force Exit"
                elif df.loc[prev_bar, 'close'] < df.loc[prev_bar, 'Trend_EMA']:
                    exit_reason = "Trend Break"
                elif df.loc[prev_bar, 'RSI'] > 70 and df.loc[prev_bar, 'TSI'] < df.loc[prev_bar, 'TSI_Signal']:
                    exit_reason = "RSI Overbought and TSI Cross Down"
                else:
                    exit_reason = "Other"
            else:
                exit_reason = "Unknown"
        else:
            exit_dt = df.index[-1]
            exit_price = df.loc[exit_dt, 'close']
            exit_reason = "Open at End"
        
        # Calculate P&L
        if shares > 0:
            gross_pnl = shares * (exit_price - entry_price)
            entry_cost = shares * entry_price * (cost_bps * 0.0001)
            exit_cost = shares * exit_price * (cost_bps * 0.0001)
            net_pnl = gross_pnl - entry_cost - exit_cost
            
            gross_return_pct = (exit_price - entry_price) / entry_price * 100
            net_return_pct = gross_return_pct - (2 * cost_bps * 0.01)
        else:
            gross_pnl = 0
            net_pnl = 0
            gross_return_pct = 0
            net_return_pct = 0
        
        holding_bars = df.index.get_loc(exit_dt) - df.index.get_loc(entry_dt)
        
        trades.append({
            'Entry_Time': entry_dt,
            'Exit_Time': exit_dt,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Shares': shares,
            'Holding_Bars': holding_bars,
            'Exit_Reason': exit_reason,
            'Gross_PnL': gross_pnl,
            'Net_PnL': net_pnl,
            'Gross_Return_Pct': gross_return_pct,
            'Net_Return_Pct': net_return_pct
        })
    
    return pd.DataFrame(trades)

# ==========================================
# Optimizer (modified for capital system)
# ==========================================
def optimize_strategy(train_df, cost_bps, initial_capital=INITIAL_CAPITAL):
    best_params = {'tsi_long': 25, 'tsi_short': 13, 'trend': 200}
    best_score = -np.inf
    
    tsi_long_opts = [20, 25, 30]
    tsi_short_opts = [10, 13]
    trend_opts = [175, 200, 225]
    
    for tsi_long in tsi_long_opts:
        for tsi_short in tsi_short_opts:
            for trend in trend_opts:
                
                if tsi_short >= tsi_long:
                    continue
                
                try:
                    # FIX: Copy FIRST (one line change!)
                    temp_df = train_df.copy()
                    
                    temp_df['TSI'], temp_df['TSI_Signal'] = calculate_tsi(temp_df['close'], tsi_long, tsi_short)
                    temp_df['RSI'], _ = calculate_rsi(temp_df['close'])
                    temp_df['Trend_EMA'] = temp_df['close'].ewm(span=trend, adjust=False).mean()
                    temp_df['Vol_MA'] = temp_df['volume'].rolling(window=20).mean()
                    
                    temp_df = generate_intraday_signals(temp_df, trend, tsi_long, tsi_short, start_flat=False)
                    temp_df = calc_capital_returns(temp_df, initial_capital, POSITION_SIZE_PCT, cost_bps)
                    
                    returns = temp_df['Equity_Return'].dropna()
                    
                    if len(returns) > 0 and returns.std() > 0:
                        score = returns.mean() / returns.std()
                    else:
                        score = -np.inf
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'tsi_long': tsi_long, 'tsi_short': tsi_short, 'trend': trend}
                
                except:
                    continue
    
    return best_params
# ==========================================
# Walk-Forward (same structure, different calc function)
# ==========================================
def walk_forward_backtest(df, train_window_days=90, test_window_days=30, cost_bps=2.0,
                          initial_capital=INITIAL_CAPITAL):
    print(f"Starting Walk-Forward Validation (Capital-Based)...")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.0f}% of capital")
    print(f"Transaction Cost: {cost_bps} bps per side")
    print("="*60)
    
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    
    warmup_days = 20
    warmup_period = pd.Timedelta(days=warmup_days)
    
    test_results = []
    current_date = start_date + pd.Timedelta(days=train_window_days)
    
    fold_num = 0
    running_capital = initial_capital  # Track capital across folds
    running_costs = 0.0  # Track costs across folds
    
    while current_date < end_date:
        fold_num += 1
        
        train_start = current_date - pd.Timedelta(days=train_window_days)
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        
        train_data = df.loc[train_start:current_date].iloc[:-1].copy()
        
        if len(train_data) < 1000:
            print(f"Fold {fold_num}: Insufficient training data, skipping...")
            current_date = test_end
            continue
        
        print(f"Fold {fold_num}: Optimizing on {train_start.date()} to {current_date.date()}...", end=" ")
        params = optimize_strategy(train_data, cost_bps, running_capital)
        print(f"TSI={params['tsi_long']}/{params['tsi_short']}, Trend={params['trend']}")
        
        buffer_start = current_date - warmup_period
        buffered_data = df.loc[buffer_start:test_end].copy()
        
        buffered_data['TSI'], buffered_data['TSI_Signal'] = calculate_tsi(buffered_data['close'], params['tsi_long'], params['tsi_short'])
        buffered_data['RSI'], _ = calculate_rsi(buffered_data['close'])
        buffered_data['Trend_EMA'] = buffered_data['close'].ewm(span=params['trend'], adjust=False).mean()
        buffered_data['Vol_MA'] = buffered_data['volume'].rolling(window=20).mean()
        
        oos_data = buffered_data.loc[current_date:test_end].copy()
        
        oos_data = generate_intraday_signals(oos_data, params['trend'], params['tsi_long'], params['tsi_short'], start_flat=True)
        
        if len(test_results) > 0:
            last_timestamp = test_results[-1].index[-1]
            oos_data = oos_data[oos_data.index > last_timestamp]
        
        if oos_data.empty:
            print(f"Fold {fold_num}: No new data in OOS period")
            break
        
        # Use running capital from previous fold
        oos_data = calc_capital_returns(oos_data, running_capital, POSITION_SIZE_PCT, cost_bps, running_costs)
        
        # Update running capital and costs for next fold
        running_capital = oos_data['Total_Equity'].iloc[-1]
        running_costs = oos_data['Cumulative_Costs'].iloc[-1]
        
        oos_data['Fold'] = fold_num
        oos_data['Opt_TSI_Long'] = params['tsi_long']
        oos_data['Opt_TSI_Short'] = params['tsi_short']
        oos_data['Opt_Trend'] = params['trend']
        
        test_results.append(oos_data)
        
        print(f"   OOS: {oos_data.index[0].date()} to {oos_data.index[-1].date()} | Equity: ${running_capital:,.0f}")
        
        current_date = test_end
    
    if not test_results:
        print("ERROR: Not enough data to run backtest")
        return pd.DataFrame()
    
    print("="*60)
    print(f"Completed {fold_num} folds")
    
    return pd.concat(test_results, axis=0)

# ==========================================
# Main
# ==========================================
def main():
    FILEPATH = f"Data_5min/{TICKER}_5min_2019_2024.parquet"
    SOURCE_TIMEZONE = 'UTC'
    TRADING_COST_BPS = 2
    
    print("="*60)
    print("TSI INTRADAY STRATEGY - CAPITAL-BASED VERSION")
    print("="*60)
    
    if not os.path.exists(FILEPATH):
        print(f"\nERROR: Data file not found at: {FILEPATH}")
        return
    
    print(f"Loading data from: {FILEPATH}")
    df = pd.read_parquet(FILEPATH)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    df = df.sort_index()
    df = prepare_data(df, source_tz=SOURCE_TIMEZONE, target_tz='US/Eastern')
    
    print(f"Data loaded: {len(df)} bars")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    bars_per_day = df.groupby(df.index.date).size().mode()[0]
    print(f"Bars per day: {bars_per_day}")
    print()
    
    wfv_df = walk_forward_backtest(df, 90, 30, TRADING_COST_BPS, INITIAL_CAPITAL)
    
    if wfv_df.empty:
        print("ERROR: No results from backtest")
        return
    
    print("\nCalculating performance metrics...")
    metrics, wfv_df = calculate_capital_metrics(wfv_df, INITIAL_CAPITAL)
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital:        ${INITIAL_CAPITAL:>12,.0f}")
    print(f"Final Equity:           ${metrics['Strategy_Final_Equity']:>12,.0f}")
    print(f"Total Return:           {metrics['Strategy_Total_Return']:>12.2f}%")
    print(f"Total Costs:            ${metrics['Total_Costs']:>12,.0f}")
    print(f"-" * 60)
    print(f"Buy & Hold Final:       ${metrics['Buy_Hold_Final_Equity']:>12,.0f}")
    print(f"Buy & Hold Return:      {metrics['Buy_Hold_Total_Return']:>12.2f}%")
    print(f"Outperformance:         {metrics['Strategy_Total_Return'] - metrics['Buy_Hold_Total_Return']:>12.2f}%")
    print(f"-" * 60)
    print(f"Strategy Max Drawdown:  {metrics['Strategy_Max_DD']:>12.2f}%")
    print(f"B&H Max Drawdown:       {metrics['Buy_Hold_Max_DD']:>12.2f}%")
    print(f"-" * 60)
    print(f"Strategy Sharpe:        {metrics['Strategy_Sharpe']:>12.2f}")
    print(f"B&H Sharpe:             {metrics['Buy_Hold_Sharpe']:>12.2f}")
    print("="*60)
    
    trade_df = get_capital_trade_stats(wfv_df, TRADING_COST_BPS)
    
    if not trade_df.empty:
        print("\nTRADE STATISTICS")
        print("="*60)
        print(f"Total Trades:           {len(trade_df):>8}")
        
        wins = trade_df[trade_df['Net_PnL'] > 0]
        losses = trade_df[trade_df['Net_PnL'] <= 0]
        
        win_rate = (len(wins) / len(trade_df) * 100) if len(trade_df) > 0 else 0
        print(f"Win Rate:               {win_rate:>8.2f}%")
        print(f"Winners / Losers:       {len(wins):>4} / {len(losses):>4}")
        print(f"-" * 60)
        print(f"Total Gross P&L:        ${trade_df['Gross_PnL'].sum():>12,.0f}")
        print(f"Total Net P&L:          ${trade_df['Net_PnL'].sum():>12,.0f}")
        print(f"Average Trade (Net):    ${trade_df['Net_PnL'].mean():>12,.2f}")
        print(f"Median Trade (Net):     ${trade_df['Net_PnL'].median():>12,.2f}")
        print(f"Best Trade (Net):       ${trade_df['Net_PnL'].max():>12,.2f}")
        print(f"Worst Trade (Net):      ${trade_df['Net_PnL'].min():>12,.2f}")
        print(f"-" * 60)
        
        if len(wins) > 0:
            print(f"Avg Winner:             ${wins['Net_PnL'].mean():>12,.2f}")
        if len(losses) > 0:
            print(f"Avg Loser:              ${losses['Net_PnL'].mean():>12,.2f}")
        
        avg_holding = trade_df['Holding_Bars'].mean()
        avg_holding_hours = avg_holding * 5 / 60
        print(f"Avg Holding Period:     {avg_holding_hours:>12.1f} hours")
        
        print(f"-" * 60)
        print("Exit Reasons:")
        exit_counts = trade_df['Exit_Reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = (count / len(trade_df) * 100)
            print(f"  {reason:<20} {count:>4} ({pct:>5.1f}%)")
        
        print("="*60)
    
    # Plotting
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Equity curves
    bh_equity = INITIAL_CAPITAL * (wfv_df['open'] / wfv_df['open'].iloc[0])
    
    axes[0].plot(wfv_df.index, wfv_df['Total_Equity'], 
                 label='Strategy Equity', color='green', linewidth=1.5)
    axes[0].plot(wfv_df.index, bh_equity, 
                 label='Buy & Hold Equity', color='gray', alpha=0.6, linewidth=1.5)
    axes[0].set_title('Equity Curve (Capital-Based)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Equity ($)', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    running_max_strat = wfv_df['Total_Equity'].cummax()
    drawdown_strat = (wfv_df['Total_Equity'] / running_max_strat - 1) * 100
    
    running_max_bh = bh_equity.cummax()
    drawdown_bh = (bh_equity / running_max_bh - 1) * 100
    
    axes[1].fill_between(wfv_df.index, drawdown_strat, 0, 
                          color='red', alpha=0.3, label='Strategy DD')
    axes[1].fill_between(wfv_df.index, drawdown_bh, 0, 
                          color='gray', alpha=0.2, label='B&H DD')
    axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=11)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nBacktest complete!")

if __name__ == "__main__":
    main()