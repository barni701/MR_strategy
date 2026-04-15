import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from model_trainer import train_xgboost_meta_labeler

# ==========================================
# CONFIGURATION
# ==========================================
TICKER = "SPY"
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.7

# ==========================================
# INDICATORS & ML FEATURES (DAILY TIMEFRAME)
# ==========================================
def compute_rsi2_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the core indicators for the Larry Connors RSI(2) Strategy.
    """
    x = df.copy()
    
    # 1. Trend Filter & Exit Indicators
    x['SMA_200'] = x['close'].rolling(window=200).mean()
    x['SMA_5'] = x['close'].rolling(window=5).mean()
    
    # 2. Wilder's RSI(2) Calculation
    delta = x['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's Smoothing
    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    x['RSI_2'] = 100 - (100 / (1 + rs))
    
    # 3. Daily ML Features
    x['ret_1'] = x['close'].pct_change()
    x['ret_3'] = x['close'].pct_change(3)
    x['ret_5'] = x['close'].pct_change(5)
    x['oc_ret'] = (x['close'] - x['open']) / x['open']
    x['hl_range'] = (x['high'] - x['low']) / x['open']
    x['SMA200_Dist'] = (x['close'] - x['SMA_200']) / x['SMA_200']
    x['SMA5_Dist'] = (x['close'] - x['SMA_5']) / x['SMA_5']
    
    return x

def drop_invalid_indicator_rows(df: pd.DataFrame) -> pd.DataFrame:
    req = ['SMA_200', 'SMA_5', 'RSI_2']
    x = df.copy()
    existing = [c for c in req if c in df.columns]
    x = x.dropna(subset=existing)
    return x

# ==========================================
# SIGNAL GENERATION
# ==========================================
def generate_rsi2_signals(
    df, 
    trade_direction='long_only', 
    use_vix_filter=False, 
    vix_threshold=30.0,
    rsi_os=10, # <-- NEW: Oversold Parameter
    rsi_ob=90  # <-- NEW: Overbought Parameter
):
    """
    Larry Connors' RSI(2) Strategy.
    Executes trades on the NEXT DAY'S OPEN.
    """
    data = df.copy()

    # 1. Trend Filters
    above_200 = data['close'] > data['SMA_200']
    below_200 = data['close'] < data['SMA_200']
    
    # 2. RSI Extremes (Now dynamic for ML training)
    oversold = data['RSI_2'] < rsi_os
    overbought = data['RSI_2'] > rsi_ob
    
    # 3. VIX Filter (Optional)
    if use_vix_filter and 'VIX' in data.columns:
        vix_ok = data['VIX'] < vix_threshold
    else:
        vix_ok = True 

    # 4. Entry & Exit Logic
    long_entry = above_200 & oversold & vix_ok
    short_entry = below_200 & overbought & vix_ok
    
    long_exit = data['close'] > data['SMA_5']
    short_exit = data['close'] < data['SMA_5']

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
    data['Position_Next'] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)

    return data

# ==========================================
# EXECUTION & KELLY SIZING
# ==========================================
def calculate_kelly_position_size(win_prob, empirical_wlr=1, max_leverage=2.0, kelly_multiplier=4.0):
    """
    Empirical Kelly using standard daily parameters.
    """
    q = 1.0 - win_prob
    raw_kelly = win_prob - (q / empirical_wlr)
    target_size = raw_kelly * kelly_multiplier

    if target_size <= 0:
        return 0.0 

    return round(min(target_size, max_leverage), 3)

def build_trade_database_from_signals(
    df_with_signals: pd.DataFrame,
    feature_cols: list[str],
    cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df_with_signals.copy()
    df['Position_Change'] = df['Position_Next'].diff().fillna(df['Position_Next'])
    entries = df.index[df['Position_Change'] == 1]
    signal_exits = df.index[df['Position_Change'] == -1]

    X_rows, y_rows = [], []

    for entry_dt in entries:
        entry_idx = df.index.get_loc(entry_dt)
        if entry_idx - 1 < 0:
            continue

        signal_dt = df.index[entry_idx - 1]  

        trade = simulate_trade_from_entry(df, entry_dt, signal_exits, cost_bps)
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
    df: pd.DataFrame,
    entry_dt: pd.Timestamp,
    signal_exits: pd.DatetimeIndex,
    cost_bps: float
) -> dict | None:
    entry_idx = df.index.get_loc(entry_dt)
    entry_price = float(df.loc[entry_dt, 'open'])

    next_signal_exits = signal_exits[signal_exits > entry_dt]
    if len(next_signal_exits) == 0:
        return None

    exit_dt_signal = next_signal_exits[0]
    exit_idx_signal = df.index.get_loc(exit_dt_signal)

    # For RSI(2), we exit STRICTLY on the SMA_5 signal
    exit_price = float(df.iloc[exit_idx_signal]['open'])
    exit_reason = "SMA_5 Signal Exit"

    gross_ret = (exit_price - entry_price) / entry_price
    net_ret = gross_ret - (2 * cost_bps * 1e-4)

    return {
        'Entry_Time': entry_dt,
        'Exit_Time': exit_dt_signal,
        'Entry_Price': entry_price,
        'Exit_Price': exit_price,
        'Holding_Bars': int(exit_idx_signal - entry_idx),
        'Exit_Reason': exit_reason,
        'Gross_Return': float(gross_ret),
        'Net_Return': float(net_ret),
        'y_win': int(net_ret > 0),
    }


def calc_capital_returns_and_stats(df, initial_capital=INITIAL_CAPITAL, position_size_pct=POSITION_SIZE_PCT, 
                                   cost_bps=1.0, starting_costs=0.0, ml_model=None, feature_cols=None):
    df = df.copy()
    df['Cash'] = initial_capital
    df['Shares_Held'] = 0.0
    df['Position_Value'] = 0.0
    df['Cumulative_Costs'] = np.nan
    df.iloc[0, df.columns.get_loc('Cumulative_Costs')] = starting_costs
    df['Position_Change'] = df['Position_Next'].diff().fillna(df['Position_Next'])
    
    entries = df[df['Position_Change'] == 1].index
    signal_exits = df[df['Position_Change'] == -1].index
    
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
        df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)
        return df, pd.DataFrame(trades)

    for entry_dt in entries:
        entry_price = df.loc[entry_dt, 'open']
        entry_idx = df.index.get_loc(entry_dt)
        if cash < 100 or entry_idx == 0:
            break
        
        signal_dt = df.index[entry_idx - 1] 
        
        # 1. Get ML Prediction
        if ml_model is not None and feature_cols is not None:
            feats = df.loc[signal_dt, feature_cols].to_frame().T.astype(float)
            win_prob = ml_model.predict_proba(feats)[0, 1]
            dynamic_size_pct = calculate_kelly_position_size(win_prob)
        else:
            win_prob = 0.0
            dynamic_size_pct = position_size_pct

        # 2. Simulate the trade theoretically to get hypothetical exit info
        trade_sim = simulate_trade_from_entry(df, entry_dt, signal_exits, cost_bps)
        if trade_sim is None:
            continue
            
        is_executed = dynamic_size_pct > 0
        
        # 3A. IF REJECTED BY ML: Log the hypothetical outcome for Accuracy Tracking
        if not is_executed:
            trades.append({
                'Entry_Time': entry_dt,
                'Exit_Time': trade_sim['Exit_Time'],
                'ML_Win_Prob': win_prob,         
                'Kelly_Size': 0.0, 
                'Entry_Price': trade_sim['Entry_Price'],
                'Exit_Price': trade_sim['Exit_Price'],
                'Shares': 0.0,
                'Holding_Bars': trade_sim['Holding_Bars'],
                'Exit_Reason': trade_sim['Exit_Reason'],
                'Gross_PnL': 0.0,
                'Net_PnL': 0.0,
                'Gross_Return_Pct': trade_sim['Gross_Return'] * 100,
                'Net_Return_Pct': trade_sim['Net_Return'] * 100,
                'Portfolio_Impact_Pct': 0.0,
                'Is_Executed': False,
                'Hypo_Win': trade_sim['y_win']
            })
            continue # Do not touch capital

        # 3B. IF APPROVED BY ML: Execute live and update capital
        exit_price = trade_sim['Exit_Price']
        exit_dt_final = trade_sim['Exit_Time']
        exit_idx_final = df.index.get_loc(exit_dt_final)

        shares = (cash * dynamic_size_pct) / entry_price
        entry_cost = shares * entry_price * (cost_bps * 0.0001)
        cash -= (shares * entry_price + entry_cost)
        total_costs += entry_cost

        holding_period_idx = range(entry_idx, exit_idx_final)
        if len(holding_period_idx) > 0:
            holding_closes = df.iloc[holding_period_idx]['close'].values
            equity_curve.iloc[holding_period_idx] = cash + (shares * holding_closes)
            
            df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Cash')] = cash
            df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Shares_Held')] = shares
            df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Position_Value')] = shares * holding_closes
            df.iloc[entry_idx:exit_idx_final, df.columns.get_loc('Cumulative_Costs')] = total_costs

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
        
        trades.append({
            'Entry_Time': entry_dt,
            'Exit_Time': exit_dt_final,
            'ML_Win_Prob': win_prob,         
            'Kelly_Size': dynamic_size_pct, 
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Shares': shares,
            'Holding_Bars': trade_sim['Holding_Bars'],
            'Exit_Reason': trade_sim['Exit_Reason'],
            'Gross_PnL': gross_pnl,
            'Net_PnL': net_pnl,
            'Gross_Return_Pct': trade_sim['Gross_Return'] * 100,
            'Net_Return_Pct': trade_sim['Net_Return'] * 100,
            'Portfolio_Impact_Pct': (trade_sim['Net_Return'] * 100) * dynamic_size_pct,
            'Is_Executed': True,
            'Hypo_Win': trade_sim['y_win']
        })

    df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
    equity_curve = equity_curve.ffill()
    df['Cash'] = df['Cash'].ffill()

    df['Total_Equity'] = equity_curve
    df['Equity_Return'] = df['Total_Equity'].pct_change().fillna(0)
    df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)

    return df, pd.DataFrame(trades)

# ==========================================
# METRICS & WALK-FORWARD ENGINE
# ==========================================
def calculate_capital_metrics(df, initial_capital=INITIAL_CAPITAL):
    metrics = {}
    ann_factor = 252 # Daily Trading Days
    final_equity = df['Total_Equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100
    running_max = df['Total_Equity'].cummax()
    drawdown = (df['Total_Equity'] / running_max) - 1
    max_dd = drawdown.min() * 100
    equity_returns = df['Equity_Return'].dropna()

    days_in_backtest = (df.index[-1] - df.index[0]).days
    years = days_in_backtest / 365.25

    if final_equity > 0:
        cagr = ((final_equity / initial_capital) ** (1 / years)) - 1
        cagr_pct = cagr * 100
    else:
        cagr_pct = -100.0

    sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(ann_factor) if len(equity_returns) > 0 and equity_returns.std() > 0 else 0
    downside_std = np.sqrt(np.mean(np.minimum(0, equity_returns)**2))
    sortino = (equity_returns.mean() / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0.0
    calmar = cagr_pct / abs(max_dd) if max_dd < 0 else 0.0

    # Buy and Hold Metrics
    bh_final = initial_capital * (df['open'].iloc[-1] / df['open'].iloc[0])
    bh_return = (bh_final - initial_capital) / initial_capital * 100
    bh_equity = initial_capital * (df['open'] / df['open'].iloc[0])
    bh_max_dd = ((bh_equity / bh_equity.cummax()) - 1).min() * 100
    bh_returns = df['Buy_Hold_Return'].dropna()

    if bh_final > 0:
        bh_cagr = ((bh_final / initial_capital) ** (1 / years)) - 1
        bh_cagr_pct = bh_cagr * 100
    else:
        bh_cagr_pct = -100.0

    bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(ann_factor) if len(bh_returns) > 0 and bh_returns.std() > 0 else 0
    bh_downside_returns = np.minimum(0, bh_returns)
    bh_downside_std = np.sqrt(np.mean(bh_downside_returns**2))
    bh_sortino = (bh_returns.mean() / bh_downside_std) * np.sqrt(ann_factor) if bh_downside_std > 0 else 0.0
    bh_calmar = bh_cagr_pct / abs(bh_max_dd) if bh_max_dd < 0 else 0.0

    metrics['Strategy_CAGR'] = cagr_pct
    metrics['Strategy_Total_Return'] = total_return
    metrics['Strategy_Final_Equity'] = final_equity
    metrics['Strategy_Max_DD'] = max_dd
    metrics['Strategy_Sharpe'] = sharpe
    metrics['Strategy_Sortino'] = sortino
    metrics['Strategy_Calmar'] = calmar
    metrics['Total_Costs'] = df['Cumulative_Costs'].iloc[-1]
    
    metrics['Buy_Hold_Total_Return'] = bh_return
    metrics['Buy_Hold_CAGR'] = bh_cagr_pct
    metrics['Buy_Hold_Max_DD'] = bh_max_dd
    metrics['Buy_Hold_Sharpe'] = bh_sharpe
    metrics['Buy_Hold_Sortino'] = bh_sortino
    metrics['Buy_Hold_Calmar'] = bh_calmar
    
    return metrics, df

def walk_forward_backtest(df, min_train_days=1095, test_window_days=30, cost_bps=1.0,
                          initial_capital=INITIAL_CAPITAL, feature_cols=None, use_ml=True):
    
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    current_date = start_date + pd.Timedelta(days=min_train_days)
    fold_num = 0
    
    running_capital = initial_capital
    running_costs = 0.0
    test_results, all_trades = [], []
    
    while current_date < end_date:
        fold_num += 1
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        
        if use_ml:
            # Use an EXPANDING window for daily data to maximize ML training size
            train_data = df.loc[start_date:current_date].iloc[:-1].copy()
            
            train_data = compute_rsi2_indicators(train_data)
            train_data = drop_invalid_indicator_rows(train_data)
            
            # THE FIX: Generate a massive training database using loose RSI(40) rules
            train_data = generate_rsi2_signals(train_data, use_vix_filter=True, rsi_os=60, rsi_ob=40)
            X_db, y_db = build_trade_database_from_signals(train_data, feature_cols, cost_bps)

            if len(y_db) < 30 or len(y_db['y_win'].unique()) < 2:
                print(f"Fold {fold_num} | ML too young ({len(y_db)} trades). Defaulting to static size.")
                ml_model = None  
            else:
                print(f"Fold {fold_num} | Training ML on {len(y_db)} historical trades...")
                ml_model = train_xgboost_meta_labeler(X_db=X_db, y_db=y_db)
        else:
            ml_model = None

        # OOS Testing (200 days buffer needed for SMA_200)
        buffer_start = current_date - pd.Timedelta(days=300)
        buffered_data = df.loc[buffer_start:test_end].copy()
        
        buffered_data = compute_rsi2_indicators(buffered_data)
        buffered_data = drop_invalid_indicator_rows(buffered_data)
        
        # LIVE EXECUTION: STRICT RSI(10) RULES
        buffered_data = generate_rsi2_signals(buffered_data, use_vix_filter=True, rsi_os=30, rsi_ob=70)
        
        oos_data = buffered_data.loc[current_date:test_end].copy()
        n_trades = int(oos_data['Entry_Signal'].sum())
        
        if len(test_results) > 0:
            oos_data = oos_data[oos_data.index > test_results[-1].index[-1]]
            
        if oos_data.empty:
            current_date = test_end
            continue
            
        oos_data, fold_trades = calc_capital_returns_and_stats(
            oos_data, running_capital, POSITION_SIZE_PCT, cost_bps, running_costs, ml_model, feature_cols
        )
        
        running_capital = oos_data['Total_Equity'].iloc[-1]
        running_costs = oos_data['Cumulative_Costs'].iloc[-1]
        test_results.append(oos_data)

        if not fold_trades.empty:
            all_trades.append(fold_trades)
        
        current_date = test_end

    final_wfv_df = pd.concat(test_results, axis=0)
    final_trade_df = pd.concat(all_trades, axis=0, ignore_index=True) if all_trades else pd.DataFrame()

    return final_wfv_df, final_trade_df

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
    """
    if trade_df is None or trade_df.empty:
        raise ValueError("trade_df is empty; cannot run Monte Carlo.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")

    rng = np.random.default_rng(seed)

    if use_net_returns:
        rets = trade_df['Portfolio_Impact_Pct'].dropna().to_numpy() / 100.0
        n_trades = len(rets)
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
                pieces.append(rets[start:start + block_size])
            sim_rets = np.concatenate(pieces)[:n_trades]
            boot[:, s] = sim_rets

        for t in range(n_trades):
            paths[t + 1, :] = paths[t, :] * (1.0 + boot[t, :])

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
    arr = paths_df.to_numpy()

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

    final_equity = arr[-1, :]
    plt.figure(figsize=(10, 4))
    plt.hist(final_equity, bins=60)
    plt.title("Monte Carlo Distribution of Final Equity")
    plt.xlabel("Final Equity ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def main():
    FILEPATH = f"Data_1D/{TICKER}_1d.parquet"
    TRADING_COST_BPS = 5

    print("="*60)
    print("DAILY LARRY CONNORS RSI(2) STRATEGY")
    print("="*60)

    if not os.path.exists(FILEPATH):
        print(f"\nERROR: Data file not found at: {FILEPATH}")
        return

    df = pd.read_parquet(FILEPATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    df = df.sort_index()
    
    # CRITICAL FIX: Resample the 5-minute data into clean Daily OHLCV candles
    df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"Resampled to Daily Data: {len(df)} days")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}\n")

    FEATURE_COLS = [
        "SMA_200", "SMA_5", "RSI_2",
        "ret_1", "ret_3", "ret_5", "oc_ret", "hl_range",
        "SMA200_Dist", "SMA5_Dist"
    ]

    print("RUNNING BASELINE RSI(2) (NO ML)...")
    base_wfv_df, base_trade_df = walk_forward_backtest(
        df, 1095, 30, TRADING_COST_BPS, INITIAL_CAPITAL, feature_cols=FEATURE_COLS, use_ml=False
    )

    print("\nRUNNING XGBOOST RSI(2)...")
    ml_wfv_df, ml_trade_df = walk_forward_backtest(
        df, 1095, 30, TRADING_COST_BPS, INITIAL_CAPITAL, feature_cols=FEATURE_COLS, use_ml=True
    )

    if base_wfv_df.empty or ml_wfv_df.empty:
        print("ERROR: Missing results")
        return

    base_metrics, base_wfv_df = calculate_capital_metrics(base_wfv_df, INITIAL_CAPITAL)
    ml_metrics, ml_wfv_df = calculate_capital_metrics(ml_wfv_df, INITIAL_CAPITAL)

    # Plot Equity Curves
    plt.figure(figsize=(12, 6))
    plt.plot(base_wfv_df.index, base_wfv_df['Total_Equity'], label=f"Baseline RSI(2) (Return: {base_metrics['Strategy_Total_Return']:.2f}%)", color='gray', alpha=0.7)
    plt.plot(ml_wfv_df.index, ml_wfv_df['Total_Equity'], label=f"ML Meta-Labeler (Return: {ml_metrics['Strategy_Total_Return']:.2f}%)", color='blue', linewidth=2)
    plt.axhline(INITIAL_CAPITAL, linestyle='--', color='black', alpha=0.5)
    plt.title(f"{TICKER} Daily RSI(2): Baseline vs. ML Dynamic Sizing", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Equity ($)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Detailed Backtest Results Printout
    print("\n" + "="*60)
    print("BACKTEST RESULTS (MACHINE LEARNING RSI2)")
    print("="*60)
    print(f"Initial Capital:        ${INITIAL_CAPITAL:>12,.0f}")
    print(f"Final Equity:           ${ml_metrics['Strategy_Final_Equity']:>12,.0f}")
    print(f"Annualized Return:      {ml_metrics['Strategy_CAGR']:>12.2f}%")
    print(f"Total Return:           {ml_metrics['Strategy_Total_Return']:>12.2f}%")
    print(f"Total Costs:            ${ml_metrics['Total_Costs']:>12,.0f}")

    print(f"-" * 60)
    print(f"Buy & Hold Return:      {ml_metrics['Buy_Hold_Total_Return']:>12.2f}%")
    print(f"Strategy Max Drawdown:  {ml_metrics['Strategy_Max_DD']:>12.2f}%")
    print(f"B&H Max Drawdown:       {ml_metrics['Buy_Hold_Max_DD']:>12.2f}%")
    print(f"-" * 60)
    print(f"Strategy Sharpe:        {ml_metrics['Strategy_Sharpe']:>12.2f}")
    print(f"Strategy Sortino:       {ml_metrics['Strategy_Sortino']:>12.2f}")
    print(f"Strategy Calmar:        {ml_metrics['Strategy_Calmar']:>12.2f}")
    print(f"B&H Sharpe:             {ml_metrics['Buy_Hold_Sharpe']:>12.2f}")
    print(f"B&H Sortino:            {ml_metrics['Buy_Hold_Sortino']:>12.2f}")
    print(f"B&H Calmar:             {ml_metrics['Buy_Hold_Calmar']:>12.2f}")
    print("="*60)

    if not ml_trade_df.empty:
        # ========================================================
        # NEW: MACHINE LEARNING CLASSIFICATION ACCURACY
        # ========================================================
        print("\n" + "="*60)
        print("MACHINE LEARNING CLASSIFICATION ACCURACY")
        print("="*60)
        
        total_signals = len(ml_trade_df)
        executed_mask = ml_trade_df['Is_Executed'] == True
        rejected_mask = ml_trade_df['Is_Executed'] == False
        
        executed_trades = ml_trade_df[executed_mask]
        rejected_trades = ml_trade_df[rejected_mask]
        
        # Confusion Matrix Logic
        true_positives = len(executed_trades[executed_trades['Hypo_Win'] == 1])
        false_positives = len(executed_trades[executed_trades['Hypo_Win'] == 0])
        true_negatives = len(rejected_trades[rejected_trades['Hypo_Win'] == 0])
        false_negatives = len(rejected_trades[rejected_trades['Hypo_Win'] == 1])
        
        accuracy = (true_positives + true_negatives) / total_signals * 100
        precision = true_positives / (true_positives + false_positives) * 100 if len(executed_trades) > 0 else 0
        rejected_win_rate = (false_negatives / len(rejected_trades) * 100) if len(rejected_trades) > 0 else 0
        
        print(f"Total OOS Signals Evaluated: {total_signals}")
        print(f"Signals Approved (Executed): {len(executed_trades)}")
        print(f"Signals Rejected (Skipped):  {len(rejected_trades)}")
        print("-" * 60)
        print(f"Overall Model Accuracy:      {accuracy:.2f}%")
        print(f"Model Precision (Win Rate):  {precision:.2f}%")
        print(f"Win Rate of REJECTED Trades: {rejected_win_rate:.2f}%")
        print("-" * 60)
        
        print("Probability Calibration (Deciles):")
        bins = [0, 0.4, 0.5, 0.6, 0.7, 1.0]
        labels = ['<40%', '40-50%', '50-60%', '60-70%', '>70%']
        ml_trade_df['Prob_Bucket'] = pd.cut(ml_trade_df['ML_Win_Prob'], bins=bins, labels=labels)
        
        calib = ml_trade_df.groupby('Prob_Bucket', observed=False).agg(
            Count=('Hypo_Win', 'size'),
            Actual_Win_Rate=('Hypo_Win', lambda x: x.mean() * 100)
        )
        for index, row in calib.iterrows():
            if row['Count'] > 0:
                print(f"  Predicted {index:<6} -> Actual Win Rate: {row['Actual_Win_Rate']:>5.1f}% (Count: {row['Count']})")

        # ========================================================
        # FINANCIAL TRADE STATISTICS (Only on Executed Trades)
        # ========================================================
        if not executed_trades.empty:
            print("\n" + "="*60)
            print("FINANCIAL TRADE STATISTICS (EXECUTED TRADES ONLY)")
            print("="*60)
            wins = executed_trades[executed_trades['Net_PnL'] > 0]
            losses = executed_trades[executed_trades['Net_PnL'] <= 0]
            win_rate = (len(wins) / len(executed_trades) * 100)
            
            print(f"Total Trades:           {len(executed_trades):>8}")
            print(f"Win Rate:               {win_rate:>8.2f}%")
            print(f"Winners / Losers:       {len(wins):>4} / {len(losses):>4}")
            print(f"-" * 60)

            print(f"Average Trade (Net):    ${executed_trades['Net_PnL'].mean():>12,.2f}")
            print(f"Average Kelly Size:     %{executed_trades['Kelly_Size'].mean():>12.2f}")
            print(f"Max Kelly Size:         %{executed_trades['Kelly_Size'].max():>12.2f}")

            print(f"Total Gross P&L:        ${executed_trades['Gross_PnL'].sum():>12,.0f}")
            print(f"Total Net P&L:          ${executed_trades['Net_PnL'].sum():>12,.0f}")
            print(f"Median Trade (Net):     ${executed_trades['Net_PnL'].median():>12,.2f}")
            print(f"Best Trade (Net):       ${executed_trades['Net_PnL'].max():>12,.2f}")
            print(f"Worst Trade (Net):      ${executed_trades['Net_PnL'].min():>12,.2f}")
            print(f"-" * 60)
            
            if len(wins) > 0:
                print(f"Avg Winner:             ${wins['Net_PnL'].mean():>12,.2f}")
                print(f"Avg Kelly Winner:       %{wins['Kelly_Size'].mean():>12.2f}")
            if len(losses) > 0:
                print(f"Avg Loser:              ${losses['Net_PnL'].mean():>12,.2f}")
                print(f"Avg Kelly Loser:        %{losses['Kelly_Size'].mean():>12.2f}")
                
            avg_holding = executed_trades['Holding_Bars'].mean()
            print(f"Avg Holding Period:     {avg_holding:>12.1f} Days")
            print("="*60)

    # --- Block bootstrap Monte Carlo ---
    # Must use ONLY the executed trades for the Monte Carlo
    executed_mc_trades = ml_trade_df[ml_trade_df['Is_Executed'] == True] if not ml_trade_df.empty else pd.DataFrame()
    
    if not executed_mc_trades.empty and len(executed_mc_trades) >= 20:
        BLOCK_SIZE = 5 
        try:
            mc_paths, mc_summary = monte_carlo_block_bootstrap_from_trade_returns(
                executed_mc_trades,
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
        except Exception as e:
            print(f"\nMonte Carlo skipped due to error: {e}")
    else:
        print("\nBlock-bootstrap Monte Carlo skipped (not enough executed trades).")

if __name__ == "__main__":
    main()