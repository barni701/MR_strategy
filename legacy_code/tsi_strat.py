import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
RTH_START = "09:30"
RTH_END = "16:00" 
LUNCH_START = "11:00"
LUNCH_END = "13:00"
EOD_CUTOFF = "15:45" # No new entries after this time
EOD_EXIT = "15:55"   # Force exit before close

TP_PCT = 0.003  # Take profit level (0.3%)
# ==========================================
# 1. Data Preprocessing & Timezone Handling
# ==========================================
def prepare_data(df, source_tz='UTC', target_tz='US/Eastern'):
    """
    Properly handle timezone conversion and filter to regular trading hours.
    
    CRITICAL: source_tz must match the actual timezone of your data.
    If your data is already in US/Eastern, set source_tz='US/Eastern'.
    """
    data = df.copy()
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    
    # Handle timezone properly
    if data.index.tz is None:
        # Naive timestamps - localize to source timezone
        data.index = data.index.tz_localize(source_tz)
    
    # Convert to target timezone
    data.index = data.index.tz_convert(target_tz)
    
    # Filter to Regular Trading Hours ONLY
    times = data.index.time
    rth_mask = (times >= pd.Timestamp(RTH_START).time()) & (times < pd.Timestamp(RTH_END).time())
    data = data[rth_mask].copy()
    
    # Remove duplicates
    data = data[~data.index.duplicated(keep='first')]
    
    return data

# ==========================================
# 2. Indicator Calculations
# ==========================================
def calculate_rsi(series, period=14):
    """Calculate RSI with EMA smoothing"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Neutral RSI when no data
    
    rsi_ema = rsi.ewm(span=period, adjust=False).mean()
    return rsi, rsi_ema

def calculate_tsi(series, long_span=25, short_span=13, signal_span=7):
    """Calculate True Strength Index with signal line"""
    pc = series.diff()
    abs_pc = pc.abs()

    # Double smoothing
    pc_smooth = pc.ewm(span=long_span, adjust=False).mean()
    pc_double_smooth = pc_smooth.ewm(span=short_span, adjust=False).mean()

    abs_pc_smooth = abs_pc.ewm(span=long_span, adjust=False).mean()
    abs_pc_double_smooth = abs_pc_smooth.ewm(span=short_span, adjust=False).mean()

    # Avoid division by zero
    tsi = (pc_double_smooth / abs_pc_double_smooth.replace(0, np.nan)) * 100
    tsi = tsi.fillna(0)
    
    signal = tsi.ewm(span=signal_span, adjust=False).mean()
    return tsi, signal

# ==========================================
# 3. Strategy Signal Generation
# ==========================================
def generate_intraday_signals(df, trend_period=200, tsi_long=25, tsi_short=13, 
                               reset_position=True, start_flat=True):
    """
    Generates Entry AND Exit signals for 5-minute bars.
    
    Key Features:
    - Uses PREVIOUS bar data for all signal decisions (no look-ahead bias)
    - Filters out lunch hours for entries AND exits
    - Forces EOD exits
    - Prevents late-day entries
    - Can reset position state for WFV (start_flat=True)
    
    Returns:
    - DataFrame with Position_Next column indicating position to hold at NEXT bar's open
    """

    data = df.copy()
    
    # Extract time information (already in US/Eastern from prepare_data)
    times = data.index.time
    # Time-based filters
    lunch = (times >= pd.Timestamp(LUNCH_START).time()) & (times < pd.Timestamp(LUNCH_END).time())
    eod_cutoff = times >= pd.Timestamp(EOD_CUTOFF).time()
    eod_exit = times >= pd.Timestamp(EOD_EXIT).time()
    
    # ============================================
    # Define Signal Components (using PREVIOUS bar data)
    # ============================================
    
    # ENTRY CONDITIONS
    trend_condition = data['close'] > data['Trend_EMA']

    tsi_cross_up = (data['TSI'] > data['TSI_Signal']) & \
                   (data['TSI'].shift(1) <= data['TSI_Signal'].shift(1))
    tsi_pullback = data['TSI'] < -20
    rsi_room_to_grow = data['RSI'] < 55
    vol_valid = data['volume'] > (data['Vol_MA'] * 0.8)
    
    # Entry Signal (excluding lunch AND late day)
    data['Entry_Signal'] = np.where(
        trend_condition & 
        tsi_cross_up & 
        tsi_pullback & 
        rsi_room_to_grow & 
        vol_valid & 
        (~lunch) & 
        (~eod_cutoff),
        1, 0
    )

    
    # EXIT CONDITIONS (using PREVIOUS bar data)
    tsi_cross_down = (data['TSI'] < data['TSI_Signal']) & \
                     (data['TSI'].shift(1) >= data['TSI_Signal'].shift(1))
    rsi_overbought = data['RSI'] > 70
    trend_break = data['close'] < data['Trend_EMA']

    
    
    # Exit Signal (excluding lunch to avoid bad fills, but INCLUDING EOD forced exit)
    data['Exit_Signal'] = np.where(
        (tsi_cross_down | rsi_overbought | trend_break | eod_exit ) & (~lunch),
        1, 0
    )
    
    # Force exit at EOD 
    data.loc[eod_exit, 'Exit_Signal'] = 1
    
    # ============================================
    # Position State Management
    # ============================================
    # Priority: Exit takes precedence over Entry
    # If both signal on same bar, we EXIT (safety first)
    
    data['Signal_Action'] = 0  # 0 = no change, 1 = go long, -1 = go flat
    
    # Apply signals with proper priority
    # First mark exits
    data.loc[data['Exit_Signal'] == 1, 'Signal_Action'] = -1
    # Then mark entries (but don't overwrite exits if both triggered)
    data.loc[(data['Entry_Signal'] == 1) & (data['Signal_Action'] == 0), 'Signal_Action'] = 1
    

    data['TP_Hit'] = False
    data['Entry_Price'] = np.nan  


    # Build position state
    if start_flat:
        data['Position_State'] = 0
        in_position = False
        entry_price = 0.0 # Reset entry price
        
        # Optimizing column lookups
        pos_idx = data.columns.get_loc('Position_State')
        tp_idx = data.columns.get_loc('TP_Hit')
        ref_idx = data.columns.get_loc('Entry_Price')
        
        for i in range(len(data)):
            action = data.iloc[i]['Signal_Action']
            
            if in_position:
                # 1. CAPTURE ENTRY PRICE (The Fix)
                # If entry_price is 0, it means we just entered this bar.
                # We must set the Reference Price to THIS bar's Open (Execution Price).
                if entry_price == 0.0:
                    entry_price = data.iloc[i]['open']
                
                # Store it for calc_returns to use later
                data.iloc[i, ref_idx] = entry_price 

                # 2. CHECK TAKE PROFIT
                current_high = data.iloc[i]['high']
                
                # Compare High against the CORRECT Execution Price
                if current_high >= entry_price * (1 + TP_PCT):
                    in_position = False
                    data.iloc[i, pos_idx] = 0
                    data.iloc[i, tp_idx] = True # Mark TP Hit
                    continue 

                # 3. CHECK STANDARD EXITS
                if action == -1:
                    in_position = False
                    data.iloc[i, pos_idx] = 0
                else:
                    data.iloc[i, pos_idx] = 1 # Hold
                    
            else: # Not in position
                if action == 1:
                    in_position = True
                    # IMPORTANT: Do NOT set entry_price to Open[i] here.
                    # Open[i] is the Signal Bar Open. We want Open[i+1].
                    # We set it to 0.0 as a flag to capture it in the next iteration.
                    entry_price = 0.0 
                    
                    data.iloc[i, pos_idx] = 1 # Mark state as active (Signal fired)
                    data.iloc[i, ref_idx] = np.nan # Clear reference price until we capture it
                else:
                    data.iloc[i, pos_idx] = 0
                    
    else:
        # [Keep existing fallback code]
        position_changes = data['Signal_Action'].replace(0, np.nan)
        data['Position_State'] = position_changes.map({1: 1, -1: 0}).ffill().fillna(0)

    data['Position_Next'] = data['Position_State'].shift(1).fillna(0)


    return data

# ==========================================
# 4. Return Calculation (ALIGNED WITH EXECUTION)
# ==========================================
def calc_returns(df):
    """
    Calculate returns ALIGNED with execution model.
    
    Execution Model:
    - Signal generated at bar t close
    - Position taken at bar t+1 open
    - Held until next signal at bar t+n close
    - Exit at bar t+n+1 open
    
    Therefore: Returns are OPEN-TO-OPEN when in position
    """
    df = df.copy()
    
    # Calculate open-to-open returns
    # This captures the actual P&L from our execution model
    df['Open_to_Open_Return'] = df['open'].pct_change().fillna(0)
    
    # Strategy return is open-to-open return * position
    # Position_Next tells us what position we hold at THIS bar's open
    df['Strategy_Return'] = df['Position_Next'] * df['Open_to_Open_Return']


    tp_mask = df['TP_Hit'].shift(1) == True  

    if tp_mask.any():

        entry_ref = df.loc[tp_mask, 'Entry_Price'].shift(1)  # Entry price is open of the bar where TP was hit

        prev_open = df.loc[tp_mask, 'open'].shift(1)  # Previous open for return calculation

        target_price = entry_ref * (1 + TP_PCT)

        effective_exit = np.maximum(target_price, prev_open)  # If TP is above open, we use TP; otherwise, we use open (worst case)

        ret = (effective_exit - prev_open) / prev_open

        df.loc[tp_mask, 'Strategy_Return'] = ret.fillna(0)  # Override strategy return for TP hits

    # Also calculate buy & hold for comparison (using same open-to-open)
    df['Buy_Hold_Return'] = df['Open_to_Open_Return']
    
    return df

# ==========================================
# 5. Metrics Calculation
# ==========================================
def calculate_metrics(df, strategy_col='Strategy_Return', bh_col='Buy_Hold_Return', cost_bps=2.5):
    """
    Calculate strategy metrics with realistic costs.
    
    Cost Model: 2.5 bps per side (5 bps round-trip) is more realistic for:
    - Bid-ask spread
    - Market impact
    - Slippage on market orders
    """
    metrics = {}
    
    # Calculate bars per day using RTH data in correct timezone
    # Group by local date (already in US/Eastern)
    bars_per_day_series = df.groupby(df.index.date).size()
    bars_per_day = int(bars_per_day_series.mode()[0]) if len(bars_per_day_series) > 0 else 78
    
    # Annualization factor
    ann_factor = 252 * bars_per_day
    
    # Calculate transaction costs
    # A trade happens when Position_Next changes
    position_changes = df['Position_Next'].diff().abs().fillna(0)
    
    # Round-trip cost = entry cost + exit cost
    # We pay cost on every position change (0->1 is entry, 1->0 is exit)
    cost_per_trade = cost_bps * 0.0001  # Convert bps to decimal
    cost_drag = position_changes * cost_per_trade
    
    # Net strategy return
    df['Net_Strategy_Return'] = df[strategy_col] - cost_drag
    
    # Calculate metrics for both strategy and buy & hold
    for name, col in [('Strategy', 'Net_Strategy_Return'), ('Buy_Hold', bh_col)]:
        returns = df[col].dropna()
        
        if len(returns) == 0:
            continue
        
        # Cumulative Return
        cumulative = (1 + returns).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100
        
        # Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min() * 100
        
        # Sharpe Ratio (annualized)
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
        else:
            sharpe = 0
            
        # Store metrics
        metrics[f'{name}_Total_Return'] = total_return
        metrics[f'{name}_Max_DD'] = max_dd
        metrics[f'{name}_Sharpe'] = sharpe
        
    return metrics, df

# ==========================================
# 6. Trade Reconstruction
# ==========================================
def get_trade_stats(df, cost_bps):
    """
    Reconstruct individual trades from position changes.
    Handles open trades at the end by force-closing them.
    """
    df = df.copy()
    
    # Identify position changes
    df['Position_Change'] = df['Position_Next'].diff()
    
    # Entry = 0 -> 1 (change = +1)
    # Exit = 1 -> 0 (change = -1)
    entries = df[df['Position_Change'] == 1].index
    exits = df[df['Position_Change'] == -1].index
    
    trades = []
    
    # Handle case where we start in a position (shouldn't happen with start_flat=True)
    if len(exits) > 0 and len(entries) > 0:
        if exits[0] < entries[0]:
            exits = exits[1:]  # Remove orphaned exit
    
    # Match entries with exits
    for i, entry_dt in enumerate(entries):
        
        # Entry executed at this bar's open
        entry_price = df.loc[entry_dt, 'open']
        
        # Find corresponding exit
        if i < len(exits):
            exit_dt = exits[i]
            exit_price = df.loc[exit_dt, 'open']
            
            # Determine exit reason from previous bar's signals
            prev_idx = df.index.get_loc(exit_dt) - 1
            if prev_idx >= 0:
                prev_bar = df.index[prev_idx]
                
                # Check which exit condition triggered
                if (df.loc[prev_bar, 'high'] >= entry_price * (1 + TP_PCT)):
                    exit_reason = "Take Profit"
                    exit_price = entry_price * (1 + TP_PCT)  # Override exit price for take profit

                elif df.loc[prev_bar].name.time() >= pd.Timestamp(EOD_EXIT).time():
                    exit_reason = "EOD Force Exit"
                elif df.loc[prev_bar, 'close'] < df.loc[prev_bar, 'Trend_EMA']:
                    exit_reason = "Trend Break"
                elif df.loc[prev_bar, 'RSI'] > 75:
                    exit_reason = "RSI Overbought"
                elif (df.loc[prev_bar, 'TSI'] < df.loc[prev_bar, 'TSI_Signal']):
                    exit_reason = "TSI Cross Down"
                else:
                    exit_reason = "Other"
            else:
                exit_reason = "Unknown"
        else:
            # Open trade at end - force close at last bar
            exit_dt = df.index[-1]
            exit_price = df.loc[exit_dt, 'close']  # Use close for final trade
            exit_reason = "Open at End (Force Closed)"
        
        # Calculate returns
        gross_return = (exit_price - entry_price) / entry_price
        
        # Apply costs: entry cost + exit cost
        net_return = gross_return - (2 * cost_bps * 0.0001)
        
        # Holding period
        holding_bars = df.index.get_loc(exit_dt) - df.index.get_loc(entry_dt)
        
        trades.append({
            'Entry_Time': entry_dt,
            'Exit_Time': exit_dt,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Holding_Bars': holding_bars,
            'Exit_Reason': exit_reason,
            'Gross_Return_Pct': gross_return * 100,
            'Net_Return_Pct': net_return * 100
        })
    
    return pd.DataFrame(trades)

# ==========================================
# 7. Optimizer (with reduced overfitting)
# ==========================================
def optimize_strategy(train_df, cost_bps):
    """
    Grid search for TSI parameters and Trend period.
    
    Optimization Goal: Maximize risk-adjusted returns (Sharpe-like metric)
    to reduce overfitting to pure returns.
    """
    best_params = {'tsi_long': 25, 'tsi_short': 13, 'trend': 200}
    best_score = -np.inf
    
    # REDUCED grid to minimize overfitting
    # Only 18 combinations instead of 45
    tsi_long_opts = [20, 25, 30]
    tsi_short_opts = [10, 13]  # Reduced from 3 to 2 options
    trend_opts = [175, 200, 225]  # Reduced from 5 to 3 options
    
    for tsi_long in tsi_long_opts:
        for tsi_short in tsi_short_opts:
            for trend in trend_opts:
                
                # Skip invalid combinations
                if tsi_short >= tsi_long:
                    continue
                
                try:
                    
                    train_df['TSI'], train_df['TSI_Signal'] = calculate_tsi(train_df['close'], long_span=tsi_long, short_span=tsi_short)
                    train_df['RSI'], _ = calculate_rsi(train_df['close'])
                    train_df['Trend_EMA'] = train_df['close'].ewm(span=trend, adjust=False).mean()
                    train_df['Vol_MA'] = train_df['volume'].rolling(window=20).mean()

                    # Generate signals
                    temp_df = generate_intraday_signals(
                        train_df,
                        trend_period=trend,
                        tsi_long=tsi_long,
                        tsi_short=tsi_short,
                        start_flat=False  # Can use forward-fill for training
                    )
                    
                    # Calculate returns
                    temp_df = calc_returns(temp_df)
                    
                    # Apply costs
                    position_changes = temp_df['Position_Next'].diff().abs().fillna(0)
                    costs = position_changes * (cost_bps * 0.0001)
                    net_returns = temp_df['Strategy_Return'] - costs
                    
                    # Skip if no trades or all zeros
                    if net_returns.std() == 0 or position_changes.sum() == 0:
                        continue
                    
                    # Objective: Maximize Sharpe-like ratio
                    # This is more robust than pure returns
                    mean_ret = net_returns.mean()
                    std_ret = net_returns.std()
                    
                    if std_ret > 0:
                        score = mean_ret / std_ret
                    else:
                        score = 0
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'tsi_long': tsi_long,
                            'tsi_short': tsi_short,
                            'trend': trend
                        }
                
                except Exception as e:
                    # Skip this combination if it fails
                    continue
    
    return best_params

# ==========================================
# 8. Walk-Forward Validation (FIXED)
# ==========================================
def walk_forward_backtest(df, train_window_days=90, test_window_days=30, cost_bps=2.5):
    """
    Walk-forward validation with proper OOS treatment.
    
    Key Fixes:
    1. OOS segments always start FLAT (no position carry-over from warmup)
    2. Proper warmup for indicators without contaminating OOS results
    3. No overlap between consecutive OOS periods
    """
    print(f"Starting Walk-Forward Validation...")
    print(f"Train Window: {train_window_days} days | Test Window: {test_window_days} days")
    print(f"Transaction Cost: {cost_bps} bps per side")
    print("="*60)
    
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Warmup period for indicators (need ~1000 bars for 200-EMA)
    # At 78 bars/day, that's ~13 days
    warmup_days = 20  # Use 20 days to be safe
    warmup_period = pd.Timedelta(days=warmup_days)
    
    test_results = []
    current_date = start_date + pd.Timedelta(days=train_window_days)
    
    fold_num = 0
    
    while current_date < end_date:
        fold_num += 1
        
        # Define windows
        train_start = current_date - pd.Timedelta(days=train_window_days)
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        
        # Training data (used for optimization)
        train_data = df.loc[train_start:current_date].iloc[:-1].copy()
        
        # Skip if insufficient training data
        if len(train_data) < 1000:
            print(f"Fold {fold_num}: Insufficient training data, skipping...")
            current_date = test_end
            continue
        
        # Optimize on training data
        print(f"Fold {fold_num}: Optimizing on {train_start.date()} to {current_date.date()}...", end=" ")
        params = optimize_strategy(train_data, cost_bps)
        print(f"TSI={params['tsi_long']}/{params['tsi_short']}, Trend={params['trend']}")
        
        # ============================================
        # CRITICAL FIX: Proper OOS handling
        # ============================================
        
        # Step 1: Get data WITH warmup buffer for indicator calculation
        buffer_start = current_date - warmup_period
        buffered_data = df.loc[buffer_start:test_end].copy()
        
        # Step 2: calculate indocators for buffered data (for warmup)

        buffered_data['TSI'], buffered_data['TSI_Signal'] = calculate_tsi(buffered_data['close'], long_span=params['tsi_long'], short_span=params['tsi_short'])
        buffered_data['RSI'], _ = calculate_rsi(buffered_data['close'])
        buffered_data['Trend_EMA'] = buffered_data['close'].ewm(span=params['trend'], adjust=False).mean()
        buffered_data['Vol_MA'] = buffered_data['volume'].rolling(window=20).mean()
        
        # Step 3: Extract ONLY the OOS period
        oos_data = buffered_data.loc[current_date:test_end].copy()
        
        # Step 4: RESET position to start flat in OOS
        oos_data = generate_intraday_signals(
            oos_data,
            trend_period=params['trend'],
            tsi_long=params['tsi_long'],
            tsi_short=params['tsi_short'],
            start_flat=True  # Force starting flat
        )
        
        # Step 5: Ensure no overlap with previous OOS segment
        if len(test_results) > 0:
            last_timestamp = test_results[-1].index[-1]
            oos_data = oos_data[oos_data.index > last_timestamp]
        
        if oos_data.empty:
            print(f"Fold {fold_num}: No new data in OOS period")
            break
        
        # Calculate returns
        oos_data = calc_returns(oos_data)
        
        # Store metadata
        oos_data['Fold'] = fold_num
        oos_data['Opt_TSI_Long'] = params['tsi_long']
        oos_data['Opt_TSI_Short'] = params['tsi_short']
        oos_data['Opt_Trend'] = params['trend']
        
        test_results.append(oos_data)
        
        print(f"   OOS period: {oos_data.index[0].date()} to {oos_data.index[-1].date()} ({len(oos_data)} bars)")
        
        # Move to next window
        current_date = test_end
    
    if not test_results:
        print("ERROR: Not enough data to run backtest")
        return pd.DataFrame()
    
    print("="*60)
    print(f"Completed {fold_num} folds")
    
    return pd.concat(test_results, axis=0)

# ==========================================
# 9. Main Execution
# ==========================================
def main():
    """
    Main execution function with proper data handling.
    """
    # Configuration
    FILEPATH = "Data_5min/SPY_5min_2019_2024.parquet"
    SOURCE_TIMEZONE = 'UTC'  # CRITICAL: Set this to match your data's actual timezone
    TRADING_COST_BPS = 2   # Realistic cost: 2.5 bps per side = 5 bps round-trip
    
    print("="*60)
    print("TSI INTRADAY STRATEGY - FIXED VERSION")
    print("="*60)
    
    # ============================================
    # Load and Prepare Data
    # ============================================
    if not os.path.exists(FILEPATH):
        print(f"\nERROR: Data file not found at: {FILEPATH}")
        return

        
    else:
        print(f"Loading data from: {FILEPATH}")
        df = pd.read_parquet(FILEPATH)
        
        # Handle index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        df = df.sort_index()
        
        # Prepare data (timezone conversion + RTH filter)
        df = prepare_data(df, source_tz=SOURCE_TIMEZONE, target_tz='US/Eastern')
    
    print(f"Data loaded: {len(df)} bars")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Calculate bars per day
    bars_per_day = df.groupby(df.index.date).size().mode()[0]
    print(f"Bars per day: {bars_per_day}")
    print()
    
    # ============================================
    # Run Walk-Forward Backtest
    # ============================================
    wfv_df = walk_forward_backtest(
        df,
        train_window_days=90,
        test_window_days=30,
        cost_bps=TRADING_COST_BPS
    )
    
    if wfv_df.empty:
        print("ERROR: No results from backtest")
        return
    
    # ============================================
    # Calculate Final Metrics
    # ============================================
    print("\nCalculating performance metrics...")
    metrics, wfv_df = calculate_metrics(wfv_df, cost_bps=TRADING_COST_BPS)

    
    # ============================================
    # Print Results
    # ============================================
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Strategy Total Return:  {metrics['Strategy_Total_Return']:>8.2f}%")
    print(f"Buy & Hold Total Return:{metrics['Buy_Hold_Total_Return']:>8.2f}%")
    print(f"Outperformance:         {metrics['Strategy_Total_Return'] - metrics['Buy_Hold_Total_Return']:>8.2f}%")
    print(f"-" * 60)
    print(f"Strategy Max Drawdown:  {metrics['Strategy_Max_DD']:>8.2f}%")
    print(f"Buy & Hold Max Drawdown:{metrics['Buy_Hold_Max_DD']:>8.2f}%")
    print(f"-" * 60)
    print(f"Strategy Sharpe Ratio:  {metrics['Strategy_Sharpe']:>8.2f}")
    print(f"Buy & Hold Sharpe Ratio:{metrics['Buy_Hold_Sharpe']:>8.2f}")
    print("="*60)
    
    # ============================================
    # Trade Statistics
    # ============================================
    trade_df = get_trade_stats(wfv_df, TRADING_COST_BPS)
    
    if not trade_df.empty:
        print("\nTRADE STATISTICS")
        print("="*60)
        print(f"Total Trades:           {len(trade_df):>8}")
        
        wins = trade_df[trade_df['Net_Return_Pct'] > 0]
        losses = trade_df[trade_df['Net_Return_Pct'] <= 0]
        
        win_rate = (len(wins) / len(trade_df) * 100) if len(trade_df) > 0 else 0
        print(f"Win Rate:               {win_rate:>8.2f}%")
        print(f"Winners / Losers:       {len(wins):>4} / {len(losses):>4}")
        print(f"-" * 60)
        print(f"Average Trade (Net):    {trade_df['Net_Return_Pct'].mean():>8.3f}%")
        print(f"Median Trade (Net):     {trade_df['Net_Return_Pct'].median():>8.3f}%")
        print(f"Best Trade (Net):       {trade_df['Net_Return_Pct'].max():>8.3f}%")
        print(f"Worst Trade (Net):      {trade_df['Net_Return_Pct'].min():>8.3f}%")
        print(f"-" * 60)
        
        if len(wins) > 0:
            print(f"Avg Winner:             {wins['Net_Return_Pct'].mean():>8.3f}%")
        if len(losses) > 0:
            print(f"Avg Loser:              {losses['Net_Return_Pct'].mean():>8.3f}%")
        
        # Avg holding period
        avg_holding = trade_df['Holding_Bars'].mean()
        avg_holding_hours = avg_holding * 5 / 60  # 5-min bars to hours
        print(f"Avg Holding Period:     {avg_holding_hours:>8.1f} hours ({avg_holding:.0f} bars)")
        
        # Exit reason breakdown
        print(f"-" * 60)
        print("Exit Reasons:")
        exit_counts = trade_df['Exit_Reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = (count / len(trade_df) * 100)
            print(f"  {reason:<20} {count:>4} ({pct:>5.1f}%)")
        
        print("="*60)




    # ============================================
    # Plotting
    # ============================================
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Equity Curves
    wfv_df['Cum_Strategy'] = (1 + wfv_df['Net_Strategy_Return']).cumprod()
    wfv_df['Cum_BH'] = (1 + wfv_df['Buy_Hold_Return']).cumprod()
    
    axes[0].plot(wfv_df.index, wfv_df['Cum_Strategy'], 
                 label='Strategy (Net of Costs)', color='green', linewidth=1.5)
    axes[0].plot(wfv_df.index, wfv_df['Cum_BH'], 
                 label='Buy & Hold', color='gray', alpha=0.6, linewidth=1.5)
    axes[0].set_title('Cumulative Returns (TSI Intraday Strategy - Fixed)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    running_max_strat = wfv_df['Cum_Strategy'].cummax()
    drawdown_strat = (wfv_df['Cum_Strategy'] / running_max_strat - 1) * 100
    
    running_max_bh = wfv_df['Cum_BH'].cummax()
    drawdown_bh = (wfv_df['Cum_BH'] / running_max_bh - 1) * 100
    
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