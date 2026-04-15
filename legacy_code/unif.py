import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================

TICKER = "SPY"

RTH_START = "9:30"
RTH_END = "16:00" 
TRADING_START = "10:30"
TRADING_END = "15:30"
EOD_CUTOFF = "15:30"
EOD_EXIT = "15:55"

TP_PCT = 1 / 100  # Initial take profit level (1%)
SL_PCT = 0.75 / 100  # Initial stop loss level (0.75%)
TRAIL_TRIGGER = 0.5 / 100  # Start trailing after 0.4% profit
TRAIL_AMOUNT = 0.25 / 100  # Trail stop at 0.2% from peak

INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.90

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

def calculate_vwap(df):
    """
    Calculate VWAP - resets each day at market open.
    """
    df = df.copy()
    
    # Group by date
    df['date'] = df.index.date
    
    # Calculate typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate cumulative volume and price*volume for each day
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()

    pv = df['typical_price'] * df['volume']
    df['cum_pv'] = pv.groupby(df['date']).cumsum()
    
    # VWAP = cumulative price*volume / cumulative volume
    df['VWAP'] = df['cum_pv'] / df['cum_vol']
    
    # VWAP bands (standard deviation based)
    df['VWAP_Std'] = df.groupby('date')['typical_price'].transform(
        lambda x: x.expanding().std()
    )
    df['VWAP_Upper'] = df['VWAP'] + (2 * df['VWAP_Std'])
    df['VWAP_Lower'] = df['VWAP'] - (2 * df['VWAP_Std'])

    
    return df

def calculate_volume_profile(df, lookback=20):
    """
    Calculate volume metrics for filtering (intraday-only, resets each day).
    """
    df = df.copy()
    df['date'] = df.index.date

    # Intraday rolling metrics within each day
    df['Vol_MA'] = df.groupby('date')['volume'].transform(
        lambda s: s.rolling(window=lookback, min_periods=lookback).mean()
    )
    df['Vol_Std'] = df.groupby('date')['volume'].transform(
        lambda s: s.rolling(window=lookback, min_periods=lookback).std()
    )

    df['Vol_Ratio'] = df['volume'] / df['Vol_MA']
    return df


def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def calculate_ema(df, span = 200):
    df['EMA_200'] = df['close'].ewm(span=span, adjust=False).mean()
    df['EMA_100'] = df['close'].ewm(span=100, adjust=False).mean()
    return df

def add_60_min_200_ema(df_5m, ema_span=200):
    df = df_5m.sort_index().copy()

    ohlcv_60 = df.resample('60min', offset='30min', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    ema_60 = ohlcv_60['close'].ewm(span=ema_span, adjust=False).mean()

    # Map back to 5m bars: last completed 60m bar at or before time t
    df['EMA_200_60min'] = ema_60.reindex(df.index, method='ffill')

    
    df['EMA_200_60min'] = df['EMA_200_60min'].shift(1)

    return df

def generate_intraday_signals(
    df,
    vol_multiplier=1.5,
    rsi_oversold=30,
    reset_position=True,
    start_flat=True
):
    """
    VWAP Mean Reversion Strategy with Volume Filtering

    This version FIXES position state handling:
      - one position at a time
      - flat at day start (if start_flat=True)
      - hard exit at EOD (15:55)
      - optional cooldown after exit (if reset_position=True)
    """
    data = df.copy()

    # --- Time masks
    times = data.index.time
    valid_time = (times >= pd.Timestamp(TRADING_START).time()) & (times < pd.Timestamp(TRADING_END).time())
    eod_cutoff = times >= pd.Timestamp(EOD_CUTOFF).time()
    eod_exit = times >= pd.Timestamp(EOD_EXIT).time()

    # --- Day boundary helper
    day = pd.Series(data.index.date, index=data.index)
    is_new_day = day.ne(day.shift(1)).fillna(True).to_numpy()

    # ==============================================
    # FEATURES USED BY ENTRY CONDITIONS
    # ==============================================
    data['VWAP_Distance'] = (data['close'] - data['VWAP']) / data['VWAP']
    data['VWAP_Distance_Prev'] = data['VWAP_Distance'].shift(1)

    below_trend = (data['close'] <= data['EMA_200_60min'])

    stretch_lo = data['VWAP_Distance_Prev'] <= -0.0021
    stretch_hi = data['VWAP_Distance_Prev'] <= -0.0030
    stretch = np.where(below_trend, stretch_hi, stretch_lo)

    vol_ok = data['Vol_Ratio'] >= vol_multiplier

    rev_ok_lo = (data['close'] - data['open']) / data['open'] >= 0.0002
    rev_ok_hi = (data['close'] - data['open']) / data['open'] >= 0.0004
    reverting = np.where(below_trend, rev_ok_hi, rev_ok_lo)

    extra_conf = np.where(below_trend, vol_ok, True)

    prev_bar_down_raw = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1) < -0.00035
    prev_bar_down = np.where(below_trend, prev_bar_down_raw, True)

    closed_inside = data['close'] > data['VWAP_Lower']

    vwap_slope = (data['VWAP'] - data['VWAP'].shift(10)) / data['VWAP'].shift(10)
    strict = vwap_slope >= -0.00005
    loose = vwap_slope >= -0.0002

    trend_ok = data['close'] > data['EMA_100']
    vwap_trend_flat_or_up = np.where(trend_ok, loose, strict)

    # ==============================================
    # RAW ENTRY/EXIT SIGNALS (NOT POSITION-AWARE YET)
    # ==============================================
    entry_raw = (
        prev_bar_down &
        stretch &
        extra_conf &
        closed_inside &
        reverting &
        vwap_trend_flat_or_up &
        valid_time &
        (~eod_cutoff)
    )

    # Exit condition: touch VWAP+1*Std OR EOD_EXIT
    at_vwap = data['high'] >= (data['VWAP'] + 1 * data['VWAP_Std'])
    exit_raw = (at_vwap | eod_exit)

    # Convert to numpy for state machine
    entry_raw = pd.Series(entry_raw, index=data.index).fillna(False).to_numpy()
    exit_raw = pd.Series(exit_raw, index=data.index).fillna(False).to_numpy()
    eod_exit_np = pd.Series(eod_exit, index=data.index).to_numpy()

    # ==============================================
    # POSITION STATE MACHINE
    # ==============================================
    n = len(data)
    pos = np.zeros(n, dtype=np.int8)          # 0 flat, 1 long
    action = np.zeros(n, dtype=np.int8)       # -1 exit, +1 entry, 0 hold
    entry_sig = np.zeros(n, dtype=np.int8)    # 1 where an entry is actually taken
    exit_sig = np.zeros(n, dtype=np.int8)     # 1 where an exit is actually taken

    in_pos = 0
    cooldown = 0  # 1 means block entry for this bar (used when reset_position=True)

    for i in range(n):
        # Flat at day start
        if start_flat and is_new_day[i]:
            in_pos = 0
            cooldown = 0

        # Enforce hard EOD exit
        if in_pos == 1 and eod_exit_np[i]:
            action[i] = -1
            exit_sig[i] = 1
            in_pos = 0
            cooldown = 1 if reset_position else 0
            pos[i] = in_pos
            continue

        # Normal exit (only if in position)
        if in_pos == 1 and exit_raw[i]:
            action[i] = -1
            exit_sig[i] = 1
            in_pos = 0
            cooldown = 1 if reset_position else 0
            pos[i] = in_pos
            continue

        # Entry (only if flat, and cooldown not active)
        if in_pos == 0 and entry_raw[i] and (cooldown == 0):
            action[i] = 1
            entry_sig[i] = 1
            in_pos = 1
            pos[i] = in_pos
            continue

        # Otherwise hold state
        pos[i] = in_pos
        cooldown = 0  # cooldown only blocks the immediate bar after exit

    # Write outputs back
    data['Entry_Signal'] = entry_sig
    data['Exit_Signal'] = exit_sig
    data['Signal_Action'] = action
    data['Position_State'] = pos
    data['Position_Next'] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)

    return data


# ==========================================

def calculate_capital_metrics(df, initial_capital=INITIAL_CAPITAL):
    metrics = {}
    bars_per_day_series = df.groupby(df.index.date).size()
    bars_per_day = int(bars_per_day_series.mode()[0]) if len(bars_per_day_series) > 0 else 78
    ann_factor = 252 * bars_per_day
    final_equity = df['Total_Equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    running_max = df['Total_Equity'].cummax()
    drawdown = (df['Total_Equity'] / running_max) - 1
    max_dd = drawdown.min() * 100
    equity_returns = df['Equity_Return'].dropna()

    if len(equity_returns) > 0 and equity_returns.std() > 0:
        sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0

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


    days_in_backtest = (df.index[-1] - df.index[0]).days
    years = days_in_backtest / 365.25
    
    # Calculate Strategy CAGR
    if final_equity > 0:
        cagr = ((final_equity / initial_capital) ** (1 / years)) - 1
        cagr_pct = cagr * 100
    else:
        cagr_pct = -100.0

    # Calculate Buy & Hold CAGR for comparison
    if bh_final > 0:
        bh_cagr = ((bh_final / initial_capital) ** (1 / years)) - 1
        bh_cagr_pct = bh_cagr * 100
    else:
        bh_cagr_pct = -100.0
        
    metrics['Strategy_CAGR'] = cagr_pct
    metrics['Buy_Hold_CAGR'] = bh_cagr_pct
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

def calc_capital_returns_and_stats(df, initial_capital=INITIAL_CAPITAL, position_size_pct=POSITION_SIZE_PCT, 
                                   cost_bps=1.0, starting_costs=0.0):
    df = df.copy()
    df['Cash'] = initial_capital
    df['Shares_Held'] = 0.0
    df['Position_Value'] = 0.0
    df['Cumulative_Costs'] = np.nan
    df.iloc[0, df.columns.get_loc('Cumulative_Costs')] = starting_costs
    df['Position_Change'] = df['Position_Next'].diff()
    
    entries = df[df['Position_Change'] == 1].index
    signal_exits = df[df['Position_Change'] == -1].index
    
    cash = initial_capital
    total_costs = starting_costs
    equity_curve = pd.Series(np.nan, index=df.index)
    equity_curve.iloc[0] = initial_capital
    df.iloc[0, df.columns.get_loc('Cash')] = initial_capital
    
    trades = []  # Build the trade log inside the equity curve loop

    if len(entries) == 0:
        df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
        df['Total_Equity'] = initial_capital
        df['Equity_Return'] = 0.0
        df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)
        return df, pd.DataFrame(trades)

    for entry_dt in entries:
        entry_price = df.loc[entry_dt, 'open']
        if cash < 100:
            break
        
        shares = (cash * position_size_pct) / entry_price
        entry_cost = shares * entry_price * (cost_bps * 0.0001)
        cash -= (shares * entry_price + entry_cost)
        total_costs += entry_cost

        entry_idx = df.index.get_loc(entry_dt)

        # Find the next SIGNAL exit after this entry (upper bound on holding time)
        next_signal_exits = signal_exits[signal_exits > entry_dt]
        if len(next_signal_exits) == 0:
            # No signal exit available -> mark-to-market to end (same behavior as your original "open at end")
            remaining_idx = range(entry_idx, len(df))
            remaining_closes = df.iloc[remaining_idx]['close'].values
            equity_curve.iloc[remaining_idx] = cash + (shares * remaining_closes)

            df.iloc[entry_idx:, df.columns.get_loc('Cash')] = cash
            df.iloc[entry_idx:, df.columns.get_loc('Shares_Held')] = shares
            df.iloc[entry_idx:, df.columns.get_loc('Position_Value')] = shares * remaining_closes
            df.iloc[entry_idx:, df.columns.get_loc('Cumulative_Costs')] = total_costs
            continue

        exit_dt_signal = next_signal_exits[0]
        exit_idx_signal = df.index.get_loc(exit_dt_signal)

        # 1) Scan for TP/SL/Trailing BEFORE the signal exit (no flags needed)
        highest = entry_price
        trailing_active = False

        exit_idx_final = exit_idx_signal
        exit_reason = "Signal Exit"
        exit_price = df.iloc[exit_idx_signal]['open']  # baseline: next candle open on signal

        for j in range(entry_idx, exit_idx_signal):
            bar_high = df.iloc[j]['high']
            bar_low = df.iloc[j]['low']

            if bar_high > highest:
                highest = bar_high

            if (not trailing_active) and ((highest - entry_price) / entry_price >= TRAIL_TRIGGER):
                trailing_active = True

            sl_level = entry_price * (1 - SL_PCT)
            tp_level = entry_price * (1 + TP_PCT)
            trail_level = highest * (1 - TRAIL_AMOUNT) if trailing_active else None

            hit_sl = (bar_low <= sl_level)
            hit_tp = (bar_high >= tp_level)
            hit_trail = (trail_level is not None) and (bar_low <= trail_level)

            # Conservative ordering for same-bar hits on OHLC data
            if hit_sl:
                exit_reason = "Stop Loss"
                exit_idx_final = j + 1
                break
            if hit_trail:
                exit_reason = "Trailing Stop"
                exit_idx_final = j + 1
                break
            if hit_tp:
                exit_reason = "Take Profit"
                exit_idx_final = j + 1
                break

        if exit_idx_final >= len(df):
            exit_idx_final = len(df) - 1

        exit_dt = df.index[exit_idx_final]
        exit_price = df.iloc[exit_idx_final]['open']  # execute at next bar open (consistent with your engine)

        # 2) Process holding period up to the final exit
        holding_period_idx = range(entry_idx, exit_idx_final)
        holding_closes = df.iloc[holding_period_idx]['close'].values
        equity_curve.iloc[holding_period_idx] = cash + (shares * holding_closes)

        df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Cash')] = cash
        df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Shares_Held')] = shares
        df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Position_Value')] = shares * holding_closes
        df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Cumulative_Costs')] = total_costs

        # 3) Close the position
        proceeds = shares * exit_price
        exit_cost = proceeds * (cost_bps * 0.0001)
        cash += (proceeds - exit_cost)
        total_costs += exit_cost

        equity_curve.iloc[exit_idx_final] = cash
        df.iloc[exit_idx_final, df.columns.get_loc('Cash')] = cash
        df.iloc[exit_idx_final, df.columns.get_loc('Shares_Held')] = 0.0
        df.iloc[exit_idx_final, df.columns.get_loc('Position_Value')] = 0.0
        df.iloc[exit_idx_final, df.columns.get_loc('Cumulative_Costs')] = total_costs

        gross_pnl = shares * (exit_price - entry_price)
        net_pnl = gross_pnl - entry_cost - exit_cost
        gross_return_pct = (exit_price - entry_price) / entry_price * 100
        net_return_pct = gross_return_pct - (2 * cost_bps * 0.01)
        holding_bars = exit_idx_final - entry_idx

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

    df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
    equity_curve = equity_curve.ffill()
    df['Cash'] = df['Cash'].ffill()

    df['Total_Equity'] = equity_curve
    df['Equity_Return'] = df['Total_Equity'].pct_change().fillna(0)
    df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)

    return df, pd.DataFrame(trades)

# ==========================================
# OPTIMIZER - Updated for VWAP strategy
# ==========================================
def optimize_strategy(train_df, cost_bps, initial_capital=INITIAL_CAPITAL):
    best_params = {'vwap_distance': 0.01, 'vol_multiplier': 1.5, 'rsi_oversold': 30}
    best_score = -np.inf
    
    # Optimize VWAP distance threshold and volume multiplier
    vol_multiplier_opts = [0.8, 1.0, 1.2, 1.5]
    rsi_oversold_opts = [20, 25, 30]
    

    for vol_mult in vol_multiplier_opts:
        #for rsi_oversold in rsi_oversold_opts:
        try:
            temp_df = train_df.copy()
            temp_df = calculate_vwap(temp_df)
            temp_df = calculate_volume_profile(temp_df)
            temp_df = calculate_rsi(temp_df)
            temp_df = calculate_ema(temp_df)
            temp_df = add_60_min_200_ema(temp_df)

            temp_df = generate_intraday_signals(temp_df, vol_mult, start_flat=False) #, rsi_oversold=rsi_oversold
            temp_df, _ = calc_capital_returns_and_stats(temp_df, initial_capital, POSITION_SIZE_PCT, cost_bps)

            returns = temp_df['Equity_Return'].dropna()
            if len(returns) > 0 and returns.std() > 0:
                score = returns.mean() / returns.std()
            else:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_params = {'vol_multiplier': vol_mult} #, 'rsi_oversold': rsi_oversold
        except:
            continue
    return best_params

# ==========================================
# WALK-FORWARD - Updated for VWAP
# ==========================================
def walk_forward_backtest(df, train_window_days=90, test_window_days=30, cost_bps=2.0,
                          initial_capital=INITIAL_CAPITAL):
    print(f"Starting Walk-Forward Validation - VWAP Strategy...")
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
    running_capital = initial_capital
    running_costs = 0.0
    while current_date < end_date:
        fold_num += 1
        train_start = current_date - pd.Timedelta(days=train_window_days)
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        train_data = df.loc[train_start:current_date].iloc[:-1].copy()
        if len(train_data) < 1000:
            print(f"Fold {fold_num}: Insufficient training data, skipping...")
            current_date = test_end
            continue
        #print(f"Fold {fold_num}: Optimizing on {train_start.date()} to {current_date.date()}...", end=" ")
        #params = optimize_strategy(train_data, cost_bps, running_capital)
        #print(f"Vol_mult={params['vol_multiplier']:.1f}") #, RSI_oversold={params['rsi_oversold']}
        buffer_start = current_date - warmup_period
        buffered_data = df.loc[buffer_start:test_end].copy()
        buffered_data = calculate_vwap(buffered_data)
        buffered_data = calculate_rsi(buffered_data)
        buffered_data = calculate_ema(buffered_data)
        buffered_data = calculate_volume_profile(buffered_data)
        buffered_data = add_60_min_200_ema(buffered_data)


        # changes to this 
        buffered_data = generate_intraday_signals(buffered_data, vol_multiplier=1.1, start_flat=True) #, rsi_oversold=params['rsi_oversold']
        oos_data = buffered_data.loc[current_date:test_end].copy()
        #oos_data = buffered_data.loc[current_date:test_end].copy()
        #oos_data = generate_intraday_signals(oos_data, vol_multiplier=params['vol_multiplier'], rsi_oversold=params['rsi_oversold'], start_flat=True)

        if len(test_results) > 0:
            last_timestamp = test_results[-1].index[-1]
            oos_data = oos_data[oos_data.index > last_timestamp]
        if oos_data.empty:
            print(f"Fold {fold_num}: No new data in OOS period")
            break
        oos_data, _ = calc_capital_returns_and_stats(oos_data, running_capital, POSITION_SIZE_PCT, cost_bps, running_costs)
        running_capital = oos_data['Total_Equity'].iloc[-1]
        running_costs = oos_data['Cumulative_Costs'].iloc[-1]
        oos_data['Fold'] = fold_num
        test_results.append(oos_data)
        print(f"   OOS: {oos_data.index[0].date()} to {oos_data.index[-1].date()} | Equity: ${running_capital:,.0f}")
        current_date = test_end
    if not test_results:
        print("ERROR: Not enough data to run backtest")
        return pd.DataFrame()
    print("="*60)
    print(f"Completed {fold_num} folds")
    return pd.concat(test_results, axis=0)




def compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd

def plot_equity_curve(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    if 'Total_Equity' not in df.columns:
        raise ValueError("df must contain Total_Equity")
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Total_Equity'], label='Strategy Equity')
    plt.axhline(initial_capital, linestyle='--', linewidth=1, label='Initial Capital')
    plt.title("Strategy Equity Curve (Walk-Forward OOS)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_drawdown(df: pd.DataFrame):
    if 'Total_Equity' not in df.columns:
        raise ValueError("df must contain Total_Equity")
    dd = compute_drawdown(df['Total_Equity'])
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, dd * 100.0, label='Drawdown (%)')
    plt.title("Strategy Drawdown (Walk-Forward OOS)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def monte_carlo_block_bootstrap_from_trade_returns(
    trade_df: pd.DataFrame,
    initial_capital: float,
    n_sims: int = 2000,
    block_size: int = 10,
    seed: int = 42,
    use_net_returns: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Monte Carlo via BLOCK BOOTSTRAP on trade returns to preserve streaks/regime clustering.

    - Resamples contiguous blocks of trades (length = block_size) with replacement
      until we have n_trades returns for each simulation.
    - If use_net_returns=True: uses Net_Return_Pct and compounds equity.
      Else: uses Net_PnL and adds dollar PnL (less consistent with dynamic sizing).

    Returns:
      paths_df: (n_trades+1, n_sims) equity paths
      summary: percentile stats on final equity and max drawdown
    """
    if trade_df is None or trade_df.empty:
        raise ValueError("trade_df is empty; cannot run Monte Carlo.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")

    rng = np.random.default_rng(seed)

    if use_net_returns:
        if 'Net_Return_Pct' not in trade_df.columns:
            raise ValueError("trade_df must contain Net_Return_Pct for use_net_returns=True.")
        rets = trade_df['Net_Return_Pct'].dropna().to_numpy() / 100.0
        n_trades = len(rets)
        if n_trades < max(5, block_size):
            raise ValueError("Not enough trades for block bootstrap.")

        n_blocks = int(np.ceil(n_trades / block_size))
        max_start = n_trades - block_size
        if max_start < 0:
            max_start = 0

        paths = np.empty((n_trades + 1, n_sims), dtype=float)
        paths[0, :] = initial_capital

        # Build bootstrapped return matrix (n_trades, n_sims)
        boot = np.empty((n_trades, n_sims), dtype=float)
        for s in range(n_sims):
            pieces = []
            for _ in range(n_blocks):
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                pieces.append(rets[start:start + block_size])
            sim_rets = np.concatenate(pieces)[:n_trades]
            boot[:, s] = sim_rets

        for t in range(n_trades):
            paths[t + 1, :] = paths[t, :] * (1.0 + boot[t, :])

    else:
        if 'Net_PnL' not in trade_df.columns:
            raise ValueError("trade_df must contain Net_PnL for use_net_returns=False.")
        pnls = trade_df['Net_PnL'].dropna().to_numpy()
        n_trades = len(pnls)
        if n_trades < max(5, block_size):
            raise ValueError("Not enough trades for block bootstrap.")

        n_blocks = int(np.ceil(n_trades / block_size))
        max_start = n_trades - block_size
        if max_start < 0:
            max_start = 0

        paths = np.empty((n_trades + 1, n_sims), dtype=float)
        paths[0, :] = initial_capital

        boot = np.empty((n_trades, n_sims), dtype=float)
        for s in range(n_sims):
            pieces = []
            for _ in range(n_blocks):
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                pieces.append(pnls[start:start + block_size])
            sim_pnls = np.concatenate(pieces)[:n_trades]
            boot[:, s] = sim_pnls

        for t in range(n_trades):
            paths[t + 1, :] = paths[t, :] + boot[t, :]

    paths_df = pd.DataFrame(paths)

    final_equity = paths_df.iloc[-1].to_numpy()
    final_stats = {
        'p5': float(np.percentile(final_equity, 5)),
        'p25': float(np.percentile(final_equity, 25)),
        'p50': float(np.percentile(final_equity, 50)),
        'p75': float(np.percentile(final_equity, 75)),
        'p95': float(np.percentile(final_equity, 95)),
        'mean': float(np.mean(final_equity)),
    }

    running_max = np.maximum.accumulate(paths_df.to_numpy(), axis=0)
    dd = paths_df.to_numpy() / running_max - 1.0
    max_dd = dd.min(axis=0)
    dd_stats = {
        'p5': float(np.percentile(max_dd, 5) * 100.0),
        'p25': float(np.percentile(max_dd, 25) * 100.0),
        'p50': float(np.percentile(max_dd, 50) * 100.0),
        'p75': float(np.percentile(max_dd, 75) * 100.0),
        'p95': float(np.percentile(max_dd, 95) * 100.0),
        'mean': float(np.mean(max_dd) * 100.0),
    }

    summary = {'final_equity': final_stats, 'max_drawdown_pct': dd_stats}
    return paths_df, summary


def plot_monte_carlo(paths_df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
    """
    Plots:
      - a handful of sample paths
      - final equity histogram
    """
    arr = paths_df.to_numpy()

    # Sample paths
    plt.figure(figsize=(12, 5))
    k = min(50, arr.shape[1])
    for i in range(k):
        plt.plot(arr[:, i], linewidth=0.8, alpha=0.5)
    plt.axhline(initial_capital, linestyle='--', linewidth=1)
    plt.title(f"Monte Carlo Equity Paths (showing {k} of {arr.shape[1]})")
    plt.xlabel("Trade #")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()

    # Final equity distribution
    final_equity = arr[-1, :]
    plt.figure(figsize=(10, 4))
    plt.hist(final_equity, bins=60)
    plt.title("Monte Carlo Distribution of Final Equity")
    plt.xlabel("Final Equity ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()





def main():
    FILEPATH = f"Data_5min/{TICKER}_5min_2019_2024.parquet"
    SOURCE_TIMEZONE = 'UTC'
    TRADING_COST_BPS = 1
    print("="*60)
    print("VWAP MEAN REVERSION STRATEGY - CAPITAL-BASED")
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

    plot_equity_curve(wfv_df, INITIAL_CAPITAL)
    plot_drawdown(wfv_df)


    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial Capital:        ${INITIAL_CAPITAL:>12,.0f}")
    print(f"Final Equity:           ${metrics['Strategy_Final_Equity']:>12,.0f}")
    print(f"Annualized Return:      {metrics['Strategy_CAGR']:>12.2f}%")
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


    _, trade_df = calc_capital_returns_and_stats(wfv_df, INITIAL_CAPITAL, POSITION_SIZE_PCT, TRADING_COST_BPS)


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

        expected_return = (win_rate/100 * wins['Net_PnL'].mean()) + ((1 - win_rate/100) * losses['Net_PnL'].mean()) - (metrics['Total_Costs'] / len(trade_df) if len(trade_df) > 0 else 0)
        print(f"Expected Return (Net):  ${expected_return:>12.2f}")

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
    print("\nBacktest complete!")




    # --- Block bootstrap Monte Carlo (preserves streaks)
    if not trade_df.empty and len(trade_df) >= 50:
        BLOCK_SIZE = 10  # try 5, 10, 20; larger = more regime persistence
        mc_paths, mc_summary = monte_carlo_block_bootstrap_from_trade_returns(
            trade_df,
            initial_capital=INITIAL_CAPITAL,
            n_sims=2000,
            block_size=BLOCK_SIZE,
            seed=42,
            use_net_returns=True
        )

        print("\n" + "="*60)
        print(f"MONTE CARLO (BLOCK BOOTSTRAP, block_size={BLOCK_SIZE})")
        print("="*60)
        fe = mc_summary['final_equity']
        mdd = mc_summary['max_drawdown_pct']
        print("Final Equity ($):")
        print(f"  Mean: {fe['mean']:,.0f} | P5: {fe['p5']:,.0f} | P25: {fe['p25']:,.0f} | "
            f"P50: {fe['p50']:,.0f} | P75: {fe['p75']:,.0f} | P95: {fe['p95']:,.0f}")
        print("Max Drawdown (%):")
        print(f"  Mean: {mdd['mean']:.2f}% | P5: {mdd['p5']:.2f}% | P25: {mdd['p25']:.2f}% | "
            f"P50: {mdd['p50']:.2f}% | P75: {mdd['p75']:.2f}% | P95: {mdd['p95']:.2f}%")

        plot_monte_carlo(mc_paths, INITIAL_CAPITAL)
    else:
        print("\nBlock-bootstrap Monte Carlo skipped (not enough trades).")

if __name__ == "__main__":
    main()