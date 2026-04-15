import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from model_trainer import train_xgboost_meta_labeler

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

TP_PCT = 0.45 / 100  # Initial take profit level (1%)
SL_PCT = 0.5 / 100  # Initial stop loss level (0.75%)
TRAIL_TRIGGER = 0.25 / 100  # Start trailing after 0.4% profit
TRAIL_AMOUNT = 0.15 / 100  # Trail stop at 0.2% from peak

INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.9

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

    df['VWAP_Std'] = df['VWAP_Std'].replace(0, np.nan)

    df['VWAP_Upper'] = df['VWAP'] + (2 * df['VWAP_Std'])
    df['VWAP_Lower'] = df['VWAP'] - (2 * df['VWAP_Std'])

    df['VWAP_Z'] = (df['close'] - df['VWAP']) / df['VWAP_Std']
    df['VWAP_Z_Prev'] = df['VWAP_Z'].shift(1)
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


def add_atr(df, n=14):
    x = df.copy()
    x['date'] = x.index.date

    prev_close = x.groupby('date')['close'].shift(1)
    prev_close = prev_close.fillna(x['open'])  # first bar of day: no overnight gap

    tr = np.maximum(
        x['high'] - x['low'],
        np.maximum((x['high'] - prev_close).abs(), (x['low'] - prev_close).abs())
    )

    x['ATR'] = tr.groupby(x['date']).rolling(n, min_periods=n).mean().reset_index(level=0, drop=True)
    x['ATR_PCT'] = x['ATR'] / x['close']
    return x


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

# ============================================================
# NEW: Feature set for ML + trade database builder (no leakage)
# ============================================================

def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ML-friendly features that only use information available up to the bar timestamp.
    No forward-looking / future data.
    """
    x = df.copy()

    # Returns / ranges
    x['ret_1'] = x['close'].pct_change()
    x['ret_3'] = x['close'].pct_change(3)
    x['ret_6'] = x['close'].pct_change(6)
    x['oc_ret'] = (x['close'] - x['open']) / x['open']
    x['hl_range'] = (x['high'] - x['low']) / x['open']
    x['range_z'] = (x['hl_range'] - x['hl_range'].rolling(50).mean()) / x['hl_range'].rolling(50).std()

    # VWAP distances (decision-time features)
    x['VWAP_Distance'] = (x['close'] - x['VWAP']) / x['VWAP']
    x['VWAP_Distance_Prev'] = x['VWAP_Distance'].shift(1)
    x['VWAP_Slope_10'] = (x['VWAP'] - x['VWAP'].shift(10)) / x['VWAP'].shift(10)
    x['VWAP_Slope_30'] = (x['VWAP'] - x['VWAP'].shift(30)) / x['VWAP'].shift(30)

    # Trend distances
    x['EMA100_Dist'] = (x['close'] - x['EMA_100']) / x['EMA_100']
    x['EMA200_60_Dist'] = (x['close'] - x['EMA_200_60min']) / x['EMA_200_60min']

    # Time-of-day (minutes since 9:30)
    minutes = (x.index.hour * 60 + x.index.minute).astype(int)
    rth_open_min = 9 * 60 + 30
    x['min_since_open'] = np.maximum(minutes - rth_open_min, 0)

    return x




def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single pipeline to compute everything your strategy + ML DB needs.
    Keeps your existing indicator logic unchanged.
    """
    x = df.copy()
    x = calculate_vwap(x)
    x = calculate_volume_profile(x)
    x = calculate_rsi(x)
    x = calculate_ema(x)
    x = add_60_min_200_ema(x)
    x = add_ml_features(x)
    x = add_atr(x)
    return x



def generate_intraday_signals(
    df,
    vol_multiplier=1,
    reset_position=True,
    start_flat=True,
    exit_buffer_k=0.0  # Added: 0.0 exits exactly at VWAP.
):
    """
    VWAP Mean Reversion Strategy with Dynamic Volatility (De-Curve-Fit)
    
    This version FIXES position state handling:
      - one position at a time
      - flat at day start (if start_flat=True)
      - hard exit at EOD (15:55)
      - optional cooldown after exit (if reset_position=True)
    """
    data = df.copy()

    # Ensure required columns exist
    req_cols = ['VWAP', 'VWAP_Std', 'VWAP_Lower', 'Vol_Ratio', 'EMA_200_60min', 'ATR_PCT']
    missing = [c for c in req_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for signals: {missing}")

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
    
    # 1. MACRO TREND
    uptrend = (data['close'] > data['EMA_200_60min'])

    # 2. DYNAMIC Z-SCORE STRETCH 
    vwap_std = data['VWAP_Std'].replace(0, np.nan)
    data['VWAP_Z'] = (data['close'] - data['VWAP']) / vwap_std
    data['VWAP_Z_Prev'] = data['VWAP_Z'].shift(1)
    
    # Target the statistical sweet spot 
    stretch = (data['VWAP_Z_Prev'] <= -1.5) & (data['VWAP_Z_Prev'] >= -2.5) 

    # 3. DYNAMIC REVERSAL & DIP 
    reverting = (data['close'] - data['open']) / data['open'] >= (data['ATR_PCT'] * 0.1)
    prev_bar_down = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1) <= -(data['ATR_PCT'].shift(1) * 0.2)

    # 4. CONFIRMATION FILTERS
    vol_ok = data['Vol_Ratio'] >= vol_multiplier
    closed_inside = data['close'] > data['VWAP_Lower']

    # Keep VWAP slope filter to ensure the VWAP itself isn't crashing
    vwap_slope = (data['VWAP'] - data['VWAP'].shift(10)) / data['VWAP'].shift(10)
    vwap_trend_ok = vwap_slope.fillna(0) >= -0.0002

    # ==============================================
    # RAW ENTRY/EXIT SIGNALS 
    # ==============================================
    entry_raw = (
        uptrend &           # Must be in a macro uptrend
        stretch &           # Z-Score based stretch
        prev_bar_down &     # Legitimate dip relative to VIX/ATR
        vol_ok &            # Volume surge confirmation
        closed_inside &     # Closing back inside the 2nd deviation band
        reverting &         # Legitimate reversal relative to VIX/ATR
        vwap_trend_ok &     # VWAP is stable
        valid_time &
        (~eod_cutoff)
    )

    exit_level = data['VWAP'] + (exit_buffer_k * data['VWAP_Std'])
    at_vwap = data['high'] >= exit_level
    

    exit_raw = (at_vwap | eod_exit)

    # Convert to numpy for state machine
    entry_raw = pd.Series(entry_raw, index=data.index).fillna(False).to_numpy()
    exit_raw = pd.Series(exit_raw, index=data.index).fillna(False).to_numpy()
    eod_exit_np = pd.Series(eod_exit, index=data.index).to_numpy()

    # ==============================================
    # POSITION STATE MACHINE (Untouched - Great Logic)
    # ==============================================
    n = len(data)
    pos = np.zeros(n, dtype=np.int8)          # 0 flat, 1 long
    action = np.zeros(n, dtype=np.int8)       # -1 exit, +1 entry, 0 hold
    entry_sig = np.zeros(n, dtype=np.int8)    # 1 where an entry is actually taken
    exit_sig = np.zeros(n, dtype=np.int8)     # 1 where an exit is actually taken

    in_pos = 0
    cooldown = 0  

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




def generate_intraday_signals_without_ML(
    df,
    vol_multiplier=1,
    reset_position=True,
    start_flat=True,
    exit_buffer_k=0.0  # Added: 0.0 exits exactly at VWAP.
):
    """
    VWAP Mean Reversion Strategy with Dynamic Volatility (De-Curve-Fit)
    
    This version FIXES position state handling:
      - one position at a time
      - flat at day start (if start_flat=True)
      - hard exit at EOD (15:55)
      - optional cooldown after exit (if reset_position=True)
    """
    data = df.copy()

    # Ensure required columns exist
    req_cols = ['VWAP', 'VWAP_Std', 'VWAP_Lower', 'Vol_Ratio', 'EMA_200_60min', 'ATR_PCT']
    missing = [c for c in req_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for signals: {missing}")

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
    
    # 1. MACRO TREND
    uptrend = (data['close'] > data['EMA_200_60min'])

    # 2. DYNAMIC Z-SCORE STRETCH 
    vwap_std = data['VWAP_Std'].replace(0, np.nan)
    data['VWAP_Z'] = (data['close'] - data['VWAP']) / vwap_std
    data['VWAP_Z_Prev'] = data['VWAP_Z'].shift(1)
    
    # Target the statistical sweet spot 
    stretch = (data['VWAP_Z_Prev'] <= -2.0) & (data['VWAP_Z_Prev'] >= -3.5)

    # 3. DYNAMIC REVERSAL & DIP
    reverting = (data['close'] - data['open']) / data['open'] >= (data['ATR_PCT'] * 0.1)
    prev_bar_down = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1) <= -(data['ATR_PCT'].shift(1) * 0.3)

    # 4. CONFIRMATION FILTERS
    vol_ok = data['Vol_Ratio'] >= vol_multiplier
    closed_inside = data['close'] > data['VWAP_Lower']

    # Keep VWAP slope filter to ensure the VWAP itself isn't crashing
    vwap_slope = (data['VWAP'] - data['VWAP'].shift(10)) / data['VWAP'].shift(10)
    vwap_trend_ok = vwap_slope.fillna(0) >= -0.0002

    # ==============================================
    # RAW ENTRY/EXIT SIGNALS 
    # ==============================================
    entry_raw = (
        uptrend &           # Must be in a macro uptrend
        stretch &           # Z-Score based stretch
        prev_bar_down &     # Legitimate dip relative to VIX/ATR
        vol_ok &            # Volume surge confirmation
        closed_inside &     # Closing back inside the 2nd deviation band
        reverting &         # Legitimate reversal relative to VIX/ATR
        vwap_trend_ok &     # VWAP is stable
        valid_time &
        (~eod_cutoff)
    )

    # 5. REALISTIC EXIT (Exit at VWAP, don't demand a 1-Std overshoot)
    exit_level = data['VWAP'] + (exit_buffer_k * data['VWAP_Std'])
    at_vwap = data['high'] >= exit_level
    


    exit_raw = (at_vwap | eod_exit)

    # Convert to numpy for state machine
    entry_raw = pd.Series(entry_raw, index=data.index).fillna(False).to_numpy()
    exit_raw = pd.Series(exit_raw, index=data.index).fillna(False).to_numpy()
    eod_exit_np = pd.Series(eod_exit, index=data.index).to_numpy()

    # ==============================================
    # POSITION STATE MACHINE (Untouched - Great Logic)
    # ==============================================
    n = len(data)
    pos = np.zeros(n, dtype=np.int8)          # 0 flat, 1 long
    action = np.zeros(n, dtype=np.int8)       # -1 exit, +1 entry, 0 hold
    entry_sig = np.zeros(n, dtype=np.int8)    # 1 where an entry is actually taken
    exit_sig = np.zeros(n, dtype=np.int8)     # 1 where an exit is actually taken

    in_pos = 0
    cooldown = 0  

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

def calculate_kelly_position_size(win_prob, tp_pct=TP_PCT, sl_pct=SL_PCT, max_leverage=8, kelly_multiplier=12):
    b = tp_pct / sl_pct 
    q = 1.0 - win_prob
    raw_kelly = win_prob - (q / b)
    target_size = raw_kelly * kelly_multiplier

    if target_size <= 0:
        return 0.0 

    return round(min(target_size, max_leverage), 3)



def generate_data(
    df,
    vol_multiplier=0.0,
    reset_position=True,
    start_flat=True,
    exit_buffer_k=0.0  # Added: 0.0 exits exactly at VWAP.
):
    """
    VWAP Mean Reversion Strategy with Dynamic Volatility (De-Curve-Fit)
    
    This version FIXES position state handling:
      - one position at a time
      - flat at day start (if start_flat=True)
      - hard exit at EOD (15:55)
      - optional cooldown after exit (if reset_position=True)
    """
    data = df.copy()

    # Ensure required columns exist
    req_cols = ['VWAP', 'VWAP_Std', 'VWAP_Lower', 'Vol_Ratio', 'EMA_200_60min', 'ATR_PCT']
    missing = [c for c in req_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for signals: {missing}")

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
    
    # 1. MACRO TREND (Crucial for win rate)
    uptrend = (data['close'] > data['EMA_200_60min'])

    # 2. DYNAMIC Z-SCORE STRETCH (Replaces static percentage distance)
    vwap_std = data['VWAP_Std'].replace(0, np.nan)
    data['VWAP_Z'] = (data['close'] - data['VWAP']) / vwap_std
    data['VWAP_Z_Prev'] = data['VWAP_Z'].shift(1)
    
    # Target the statistical sweet spot (avoiding -3.0 falling knives)
    stretch = (data['VWAP_Z_Prev'] <= -1.0) & (data['VWAP_Z_Prev'] >= -3.5)

    # 3. DYNAMIC REVERSAL & DIP (ATR-Based instead of hardcoded 0.02%)
    # Current bar must bounce by at least 30% of the current ATR
    reverting = (data['close'] - data['open']) / data['open'] >= (data['ATR_PCT'] * 0.05)
    
    # Previous bar must have actually dropped by at least 30% of its ATR
    prev_bar_down = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1) <= -(data['ATR_PCT'].shift(1) * 0.05)

    # 4. CONFIRMATION FILTERS
    vol_ok = data['Vol_Ratio'] >= vol_multiplier
    closed_inside = data['close'] > data['VWAP_Lower']

    # Keep VWAP slope filter to ensure the VWAP itself isn't crashing
    vwap_slope = (data['VWAP'] - data['VWAP'].shift(10)) / data['VWAP'].shift(10)
    vwap_trend_ok = vwap_slope.fillna(0) >= -0.0002

    # ==============================================
    # RAW ENTRY/EXIT SIGNALS 
    # ==============================================
    entry_raw = (          
        stretch &           # Z-Score based stretch
        prev_bar_down &     # Legitimate dip relative to VIX/ATR
        vol_ok &            # Volume surge confirmation
        closed_inside &     # Closing back inside the 2nd deviation band
        reverting &         # Legitimate reversal relative to VIX/ATR
        vwap_trend_ok &     # VWAP is stable
        valid_time &
        (~eod_cutoff)
    )

    # 5. REALISTIC EXIT (Exit at VWAP, don't demand a 1-Std overshoot)
    exit_level = data['VWAP'] + (exit_buffer_k * data['VWAP_Std'])
    at_vwap = data['high'] >= exit_level
    
    exit_raw = (at_vwap | eod_exit)

    # Convert to numpy for state machine
    entry_raw = pd.Series(entry_raw, index=data.index).fillna(False).to_numpy()
    exit_raw = pd.Series(exit_raw, index=data.index).fillna(False).to_numpy()
    eod_exit_np = pd.Series(eod_exit, index=data.index).to_numpy()

    # ==============================================
    # POSITION STATE MACHINE (Untouched - Great Logic)
    # ==============================================
    n = len(data)
    pos = np.zeros(n, dtype=np.int8)          # 0 flat, 1 long
    action = np.zeros(n, dtype=np.int8)       # -1 exit, +1 entry, 0 hold
    entry_sig = np.zeros(n, dtype=np.int8)    # 1 where an entry is actually taken
    exit_sig = np.zeros(n, dtype=np.int8)     # 1 where an exit is actually taken

    in_pos = 0
    cooldown = 0  

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

def build_trade_database_from_signals(
    df_with_signals: pd.DataFrame,
    feature_cols: list[str],
    cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df_with_signals.copy()

    if 'Position_Next' not in df.columns:
        raise ValueError("df must include Position_Next (run generate_intraday_signals first).")

    df['Position_Change'] = df['Position_Next'].diff().fillna(df['Position_Next'])
    entries = df.index[df['Position_Change'] == 1]
    signal_exits = df.index[df['Position_Change'] == -1]

    X_rows, y_rows = [], []

    for entry_dt in entries:
        entry_idx = df.index.get_loc(entry_dt)
        if entry_idx - 1 < 0:
            continue

        signal_dt = df.index[entry_idx - 1]  # decision bar (t)

        trade = simulate_trade_from_entry(df, entry_dt, signal_exits, cost_bps)
        if trade is None:
            continue

        feats = df.loc[signal_dt, feature_cols]
        if feats.isna().any():
            continue

        X_rows.append({
            "Signal_Time": signal_dt,
            "Entry_Time": entry_dt,
            **feats.to_dict()
        })

        y_rows.append({
            "Signal_Time": signal_dt,
            **trade
        })

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
    """
    Deterministic trade simulation:
      - entry at entry_dt OPEN
      - upper bound exit at next signal exit OPEN
      - TP/SL/Trail checked intrabar; exit executed at next bar OPEN (j+1 open)
    Returns a trade dict (times, returns, reason), or None if cannot simulate.
    """
    entry_idx = df.index.get_loc(entry_dt)
    entry_price = float(df.loc[entry_dt, 'open'])

    next_signal_exits = signal_exits[signal_exits > entry_dt]
    if len(next_signal_exits) == 0:
        return None

    exit_dt_signal = next_signal_exits[0]
    exit_idx_signal = df.index.get_loc(exit_dt_signal)

    highest = entry_price
    trailing_active = False

    exit_idx_final = exit_idx_signal
    exit_reason = "Signal Exit"

    # DEFAULT EXIT PRICE: If it doesn't hit a stop/TP, it will exit at the signal's open
    exit_price = float(df.iloc[exit_idx_signal]['open'])

    for j in range(entry_idx, exit_idx_signal):
        bar_high = float(df.iloc[j]['high'])
        bar_low = float(df.iloc[j]['low'])


        sl_level = entry_price * (1 - SL_PCT)
        tp_level = entry_price * (1 + TP_PCT)
        trail_level = highest * (1 - TRAIL_AMOUNT) if trailing_active else None

        hit_sl = (bar_low <= sl_level)
        hit_tp = (bar_high >= tp_level)
        hit_trail = (trail_level is not None) and (bar_low <= trail_level)




        if hit_sl:
            exit_reason = "Stop Loss"
            exit_idx_final = j
            exit_price = sl_level  # assume we get filled at the SL level for better realism
            break
        if hit_trail:
            exit_reason = "Trailing Stop"
            exit_idx_final = j
            exit_price = trail_level  # assume we get filled at the trail level for better realism
            break
        if hit_tp:
            exit_reason = "Take Profit"
            exit_idx_final = j
            exit_price = tp_level  # assume we get filled at the TP level for better realism
            break


        if bar_high > highest:
            highest = bar_high

        if (not trailing_active) and ((highest - entry_price) / entry_price >= TRAIL_TRIGGER):
            trailing_active = True


    if exit_idx_final >= len(df):
        exit_idx_final = len(df) - 1

    exit_dt = df.index[exit_idx_final]

    gross_ret = (exit_price - entry_price) / entry_price
    net_ret = gross_ret - (2 * cost_bps * 1e-4)

    return {
        'Entry_Time': entry_dt,
        'Exit_Time': exit_dt,
        'Entry_Price': entry_price,
        'Exit_Price': exit_price,
        'Holding_Bars': int(exit_idx_final - entry_idx),
        'Exit_Reason': exit_reason,
        'Gross_Return': float(gross_ret),
        'Net_Return': float(net_ret),
        'y_win': int(net_ret > 0),
    }



def purge_overlapping_trades(
    X_db: pd.DataFrame,
    y_db: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    embargo_bars: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purge training trades whose [Entry_Time, Exit_Time] overlaps the test interval,
    plus an optional embargo (in bars) after test_start.

    embargo_bars:
      - if >0, also purge trades that exit within embargo after test_start.
    """
    x = X_db.copy()
    y = y_db.copy()

    entry = pd.to_datetime(y['Entry_Time'])
    exit_ = pd.to_datetime(y['Exit_Time'])

    embargo_end = test_start
    if embargo_bars > 0:
        # assumes fixed 5-min bars; adjust if needed
        embargo_end = test_start + pd.Timedelta(minutes=5 * embargo_bars)

    # overlap if entry <= test_end and exit >= test_start
    overlap = (entry <= test_end) & (exit_ >= test_start)

    # embargo: trades exiting shortly after test_start are also removed
    embargo = (exit_ >= test_start) & (exit_ <= embargo_end)

    keep = ~(overlap | embargo)

    return x.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)



def drop_invalid_indicator_rows(df: pd.DataFrame) -> pd.DataFrame:
    req = ['VWAP_Std', 'Vol_Ratio', 'EMA_100', 'EMA_200_60min', 'VWAP_Lower']

    x = df.copy()
    existing = [c for c in req if c in df.columns]
    x = x.dropna(subset=existing)
    return x



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



    days_in_backtest = (df.index[-1] - df.index[0]).days
    years = days_in_backtest / 365.25

    #Strategy metrics
    # Calculate Strategy CAGR
    if final_equity > 0:
        cagr = ((final_equity / initial_capital) ** (1 / years)) - 1
        cagr_pct = cagr * 100
    else:
        cagr_pct = -100.0


    #Sharpe Ratio (annualized)
    if len(equity_returns) > 0 and equity_returns.std() > 0:
        sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0

    #Sortino Ratio (annualized)
    downside_returns = np.minimum(0, equity_returns)
    downside_std = np.sqrt(np.mean(downside_returns**2))
    sortino = (equity_returns.mean() / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0.0

    #Calmar ratio
    if max_dd < 0:
        calmar = cagr_pct / abs(max_dd)
    else:
        calmar = 0.0


    bh_final = initial_capital * (df['open'].iloc[-1] / df['open'].iloc[0])
    bh_return = (bh_final - initial_capital) / initial_capital * 100
    bh_equity = initial_capital * (df['open'] / df['open'].iloc[0])
    bh_running_max = bh_equity.cummax()
    bh_drawdown = (bh_equity / bh_running_max) - 1
    bh_max_dd = bh_drawdown.min() * 100
    bh_returns = df['Buy_Hold_Return'].dropna()


    #Buy and hold metrics

    # Calculate Buy & Hold CAGR for comparison
    if bh_final > 0:
        bh_cagr = ((bh_final / initial_capital) ** (1 / years)) - 1
        bh_cagr_pct = bh_cagr * 100
    else:
        bh_cagr_pct = -100.0

    #Sharpe Ratio (annualized)
    if len(bh_returns) > 0 and bh_returns.std() > 0:
        bh_sharpe = (bh_returns.mean() / bh_returns.std()) * np.sqrt(ann_factor)
    else:
        bh_sharpe = 0

    # Buy & Hold Sortino
    bh_downside_returns = np.minimum(0, bh_returns)
    bh_downside_std = np.sqrt(np.mean(bh_downside_returns**2))
    bh_sortino = (bh_returns.mean() / bh_downside_std) * np.sqrt(ann_factor) if bh_downside_std > 0 else 0.0

    # Buy & Hold Calmar
    if bh_max_dd < 0:
        bh_calmar = bh_cagr_pct / abs(bh_max_dd)
    else:
        bh_calmar = 0.0
    
        
    metrics['Strategy_CAGR'] = cagr_pct
    metrics['Buy_Hold_CAGR'] = bh_cagr_pct
    metrics['Strategy_Total_Return'] = total_return
    metrics['Strategy_Final_Equity'] = final_equity
    metrics['Strategy_Max_DD'] = max_dd
    metrics['Strategy_Sharpe'] = sharpe
    metrics['Strategy_Sortino'] = sortino
    metrics['Strategy_Calmar'] = calmar
    metrics['Total_Costs'] = df['Cumulative_Costs'].iloc[-1]
    metrics['Buy_Hold_Total_Return'] = bh_return
    metrics['Buy_Hold_Final_Equity'] = bh_final
    metrics['Buy_Hold_Max_DD'] = bh_max_dd
    metrics['Buy_Hold_Sharpe'] = bh_sharpe
    metrics['Buy_Hold_Sortino'] = bh_sortino
    metrics['Buy_Hold_Calmar'] = bh_calmar
    return metrics, df

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
    
    trades = []  # Build the trade log inside the equity curve loop

    if len(entries) == 0:
        df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
        df['Total_Equity'] = initial_capital
        df['Equity_Return'] = 0.0
        df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)
        return df, pd.DataFrame(trades)

    for entry_dt in entries:
        entry_price = df.loc[entry_dt, 'open']
        entry_idx = df.index.get_loc(entry_dt)
        if cash < 100 or entry_idx ==0:
            break
        

        # ML postion sizing 

        signal_dt = df.index[entry_idx - 1] # The decision bar
        
        if ml_model is not None and feature_cols is not None:
            # Grab features for the decision bar
            feats = df.loc[signal_dt, feature_cols].to_frame().T
            
            # THE FIX: Force the data back to float (fixes the Pandas .T object bug)
            feats = feats.astype(float)
            
            # Predict Probability
            win_prob = ml_model.predict_proba(feats)[0, 1]
            dynamic_size_pct = calculate_kelly_position_size(win_prob)

            
            if dynamic_size_pct <= 0:
                continue # ML OVERRIDE: Skip this trade
        else:
            win_prob = 0.0
            dynamic_size_pct = position_size_pct


        shares = (cash * dynamic_size_pct) / entry_price
        entry_cost = shares * entry_price * (cost_bps * 0.0001)
        cash -= (shares * entry_price + entry_cost)
        total_costs += entry_cost

        entry_idx = df.index.get_loc(entry_dt)

        # Find the next SIGNAL exit after this entry (upper bound on holding time)
        next_signal_exits = signal_exits[signal_exits > entry_dt]
        if len(next_signal_exits) == 0:
            # No signal exit available -> mark-to-market to end
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

            sl_level = entry_price * (1 - SL_PCT)
            tp_level = entry_price * (1 + TP_PCT)
            trail_level = highest * (1 - TRAIL_AMOUNT) if trailing_active else None

            hit_sl = (bar_low <= sl_level)
            hit_tp = (bar_high >= tp_level)
            hit_trail = (trail_level is not None) and (bar_low <= trail_level)


            # Conservative ordering for same-bar hits on OHLC data
            if hit_sl:
                exit_reason = "Stop Loss"
                exit_idx_final = j
                exit_price = sl_level  # assume we get filled at the SL level for better realism
                break
            if hit_trail:
                exit_reason = "Trailing Stop"
                exit_idx_final = j
                exit_price = trail_level  # assume we get filled at the trail level for better realism
                break
            if hit_tp:
                exit_reason = "Take Profit"
                exit_idx_final = j
                exit_price = tp_level  # assume we get filled at the TP level for better realism
                break

            if bar_high > highest:
                highest = bar_high

            if (not trailing_active) and ((highest - entry_price) / entry_price >= TRAIL_TRIGGER):
                trailing_active = True

        if exit_idx_final >= len(df):
            exit_idx_final = len(df) - 1

        exit_dt = df.index[exit_idx_final]

        
        # 2) Process holding period up to the final exit
        holding_period_idx = range(entry_idx, exit_idx_final)


        if len(holding_period_idx) > 0:
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
        portfolio_impact_pct = net_return_pct * dynamic_size_pct
        holding_bars = exit_idx_final - entry_idx

        trades.append({
            'Entry_Time': entry_dt,
            'Exit_Time': exit_dt,
            'ML_Win_Prob': win_prob,         
            'Kelly_Size': dynamic_size_pct, 
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Shares': shares,
            'Holding_Bars': holding_bars,
            'Exit_Reason': exit_reason,
            'Gross_PnL': gross_pnl,
            'Net_PnL': net_pnl,
            'Gross_Return_Pct': gross_return_pct,
            'Net_Return_Pct': net_return_pct,
            'Portfolio_Impact_Pct': portfolio_impact_pct
        })

    df['Cumulative_Costs'] = df['Cumulative_Costs'].ffill()
    equity_curve = equity_curve.ffill()
    df['Cash'] = df['Cash'].ffill()

    df['Total_Equity'] = equity_curve
    df['Equity_Return'] = df['Total_Equity'].pct_change().fillna(0)
    df['Buy_Hold_Return'] = df['open'].pct_change().fillna(0)

    return df, pd.DataFrame(trades)


# ==========================================
# WALK-FORWARD - Updated for VWAP
# ==========================================
def walk_forward_backtest(df, min_train_days=1095, test_window_days=30, cost_bps=1.0,
                          initial_capital=INITIAL_CAPITAL, feature_cols=None, use_ml=True):
    
    print(f"\nStarting Rolling Window Walk-Forward ML Backtest...")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Transaction Cost: {cost_bps} bps per side")
    print(f"Initial ML Warmup Period: {min_train_days} days (Gathering initial trades...)")
    print("="*80)
    
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Start testing ONLY after we have built up a 2-year history for the first ML model
    current_date = start_date + pd.Timedelta(days=min_train_days)
    fold_num = 0
    
    running_capital = initial_capital
    running_costs = 0.0
    test_results = []

    all_trades = []
    
    while current_date < end_date:
        fold_num += 1
        test_end = min(current_date + pd.Timedelta(days=test_window_days), end_date)
        
        # =======================================================
        # STEP 1: STRICTLY PAST DATA FOR ML TRAINING
        # =======================================================
        # Rolling window of up to 3 years (1095 days) to keep the ML model focused on recent market dynamics

        if use_ml:
            rolling_start = current_date - pd.Timedelta(days = 1095)

            actual_start = max(rolling_start, start_date)

            train_data = df.loc[actual_start:current_date].iloc[:-1].copy()

            
            # Compute indicators and use the LOOSE strategy (generate_data) to gather a big database
            train_data = compute_all_indicators(train_data)
            train_data = drop_invalid_indicator_rows(train_data)
            train_data = generate_data(train_data, start_flat=True)
            
            X_db, y_db = build_trade_database_from_signals(train_data, feature_cols, cost_bps)

            MIN_ML_TRADES = 100 # Need enough data so XGBoost doesn't curve-fit
            
            if len(y_db) < MIN_ML_TRADES or len(y_db['y_win'].unique()) < 2:
                print(f"Fold {fold_num} | Only {len(y_db)} trades. ML is too young. Defaulting to base rules.")
                ml_model = None  # Passing None forces calc_capital... to use POSITION_SIZE_PCT
            else:
                print(f"\nFold {fold_num} | Training ML on {len(y_db)} historical trades...")
                ml_model = train_xgboost_meta_labeler(X_db=X_db, y_db=y_db)
                #ml_model = train_random_forest_meta_labeler(X_db=X_db, y_db=y_db)
                #ml_model = train_logistic_meta_labeler(X_db=X_db, y_db=y_db)

            
            buffer_start = current_date - pd.Timedelta(days=60)
            buffered_data = df.loc[buffer_start:test_end].copy()
            
            buffered_data = compute_all_indicators(buffered_data)
            buffered_data = drop_invalid_indicator_rows(buffered_data)
            
            buffered_data = generate_intraday_signals(buffered_data, start_flat=True)
        
        else:
            ml_model = None
            
            buffer_start = current_date - pd.Timedelta(days=60)
            buffered_data = df.loc[buffer_start:test_end].copy()
            
            buffered_data = compute_all_indicators(buffered_data)
            buffered_data = drop_invalid_indicator_rows(buffered_data)
            
            buffered_data = generate_intraday_signals_without_ML(buffered_data, start_flat=True)

        # =======================================================
        # STEP 2: OUT-OF-SAMPLE LIVE TESTING
        # =======================================================
        # We only grab the 30-day test window (plus a 60-day buffer to warm up the EMAs)
        '''
        buffer_start = current_date - pd.Timedelta(days=60)
        buffered_data = df.loc[buffer_start:test_end].copy()
        
        buffered_data = compute_all_indicators(buffered_data)
        buffered_data = drop_invalid_indicator_rows(buffered_data)
        
        # Use your normal/strict signals for the live execution
        buffered_data = generate_intraday_signals(buffered_data, start_flat=True)'''
        
        # Slice off the warmup buffer so we only execute inside the exact 30-day window
        oos_data = buffered_data.loc[current_date:test_end].copy()

        n_trades = int(oos_data['Entry_Signal'].sum())
        
        if len(test_results) > 0:
            last_timestamp = test_results[-1].index[-1]
            oos_data = oos_data[oos_data.index > last_timestamp]
            
        if oos_data.empty:
            print(f"   OOS: {current_date.date()} to {test_end.date()} | Trades: 0 | Equity: ${running_capital:,.0f}")
            current_date = test_end
            continue
            
        # Execute the trades using the freshly trained ML model
        oos_data, fold_trades = calc_capital_returns_and_stats(
            oos_data, 
            running_capital, 
            POSITION_SIZE_PCT, 
            cost_bps, 
            running_costs, 
            ml_model=ml_model, 
            feature_cols=feature_cols
        )

        executed_trades = len(fold_trades) if not fold_trades.empty else 0
        
        running_capital = oos_data['Total_Equity'].iloc[-1]
        running_costs = oos_data['Cumulative_Costs'].iloc[-1]
        oos_data['Fold'] = fold_num
        test_results.append(oos_data)

        if not fold_trades.empty:
            all_trades.append(fold_trades)
        
        print(f"   OOS: {oos_data.index[0].date()} to {oos_data.index[-1].date()} | Raw Signals: {n_trades} | Executed Trades: {executed_trades} | Equity: ${running_capital:,.0f}")
        
        # Move forward 30 days
        current_date = test_end

    if not test_results:
        print("ERROR: Not enough data to run backtest")
        return pd.DataFrame()
        
    print("\n" + "="*80)
    print(f"Completed {fold_num} Rolling Window Folds")

    final_wfv_df = pd.concat(test_results, axis=0)
    final_trade_df = pd.concat(all_trades, axis=0, ignore_index=True) if all_trades else pd.DataFrame()

    return final_wfv_df, final_trade_df


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
        if 'Portfolio_Impact_Pct' not in trade_df.columns:
            raise ValueError("trade_df must contain Portfolio_Impact_Pct for use_net_returns=True.")
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





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_vwap_statistical_analysis(df: pd.DataFrame):
    print("\n" + "="*60)
    print("VWAP MEAN REVERSION - STATISTICAL EDGE ANALYSIS")
    print("="*60)
    
    # Work on a copy to avoid altering the main dataset
    data = df.copy()
    
    # Ensure necessary columns exist
    req_cols = ['VWAP_Z', 'close', 'open', 'high', 'low', 'EMA_200_60min']
    missing = [c for c in req_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for analysis: {missing}")

    # 1. Realistic Entry Price (Open of the NEXT bar)
    data['next_open'] = data['open'].shift(-1)
    
    # 2. Forward Returns (6 bars = 30m, 12 bars = 1h, 24 bars = 2h)
    data['fwd_ret_6'] = (data['close'].shift(-6) - data['next_open']) / data['next_open']
    data['fwd_ret_12'] = (data['close'].shift(-12) - data['next_open']) / data['next_open']
    data['fwd_ret_24'] = (data['close'].shift(-24) - data['next_open']) / data['next_open']
    
    # 3. Maximum Favorable/Adverse Excursion (over the next 24 bars / 2 hours)
    # We reverse the series, do a rolling window, and reverse it back to look *forward*
    fwd_max_24 = data['high'].iloc[::-1].rolling(window=24, min_periods=1).max().iloc[::-1].shift(-1)
    fwd_min_24 = data['low'].iloc[::-1].rolling(window=24, min_periods=1).min().iloc[::-1].shift(-1)
    
    data['MFE_24'] = (fwd_max_24 - data['next_open']) / data['next_open']  # Max Run-up
    data['MAE_24'] = (fwd_min_24 - data['next_open']) / data['next_open']  # Max Drawdown
    
    # Drop NaNs at the very end of the dataset to avoid skewed stats
    data = data.dropna(subset=['fwd_ret_24', 'MFE_24', 'MAE_24'])

    # ---------------------------------------------------------
    # ANALYSIS 1: VWAP Z-Score Bucketing
    # ---------------------------------------------------------
    # We group Z-scores into bins to see if extreme deviations actually revert.
    bins = [-np.inf, -3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    labels = ['< -3.0', '-3.0 to -2.5', '-2.5 to -2.0', '-2.0 to -1.5', '-1.5 to -1.0', 
              'Neutral', '1.0 to 1.5', '1.5 to 2.0', '2.0 to 2.5', '2.5 to 3.0', '> 3.0']
    
    data['Z_Bucket'] = pd.cut(data['VWAP_Z'], bins=bins, labels=labels)
    
    z_stats = data.groupby('Z_Bucket', observed=False).agg(
        Count=('fwd_ret_12', 'size'),
        Avg_Return_1h=('fwd_ret_12', lambda x: x.mean() * 100),
        Win_Rate_1h=('fwd_ret_12', lambda x: (x > 0).mean() * 100)
    )
    
    print("\n1. FORWARD RETURNS BY VWAP Z-SCORE (1 Hour Hold)")
    print(z_stats.to_string())
    
    # Plot Z-Score Win Rates
    z_stats['Win_Rate_1h'].plot(kind='bar', figsize=(10, 4), color='teal', alpha=0.7)
    plt.axhline(50, color='red', linestyle='--', linewidth=1)
    plt.title("1-Hour Win Rate by VWAP Z-Score Bucket")
    plt.ylabel("Win Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # ANALYSIS 2: MFE vs MAE (Stop Loss / Take Profit Optimization)
    # ---------------------------------------------------------
    # Filter only for the extreme long signals you are interested in (e.g., Z < -2.0)
    extreme_longs = data[data['VWAP_Z'] <= -2.0].copy()
    
    print(f"\n2. EXCURSION ANALYSIS (Based on {len(extreme_longs)} extreme dips: Z <= -2.0)")
    print(f"  Average Max Run-up (MFE) within 2 hrs:   {extreme_longs['MFE_24'].mean()*100:.2f}%")
    print(f"  Average Max Drawdown (MAE) within 2 hrs: {extreme_longs['MAE_24'].mean()*100:.2f}%")
    print(f"  90th Percentile Run-up (Target TP):      {extreme_longs['MFE_24'].quantile(0.90)*100:.2f}%")
    print(f"  10th Percentile Drawdown (Target SL):    {extreme_longs['MAE_24'].quantile(0.10)*100:.2f}%")
    
    # Scatter plot of MFE vs MAE
    plt.figure(figsize=(8, 6))
    plt.scatter(extreme_longs['MAE_24'] * 100, extreme_longs['MFE_24'] * 100, alpha=0.3, color='purple')
    plt.axvline(0, color='black', linewidth=1)
    plt.axhline(0, color='black', linewidth=1)
    plt.title("MFE vs MAE for Extreme VWAP Dips (2-Hour Window)")
    plt.xlabel("Max Drawdown (%) - Set Stop Loss Here")
    plt.ylabel("Max Run-up (%) - Set Take Profit Here")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # ANALYSIS 3: Trend Regime Conditioning
    # ---------------------------------------------------------
    # Does buying the dip work better in a macro uptrend or downtrend?
    extreme_longs['Macro_Trend'] = np.where(extreme_longs['close'] > extreme_longs['EMA_200_60min'], 'Uptrend', 'Downtrend')
    
    regime_stats = extreme_longs.groupby('Macro_Trend').agg(
        Count=('fwd_ret_12', 'size'),
        Avg_Return_1h=('fwd_ret_12', lambda x: x.mean() * 100),
        Win_Rate_1h=('fwd_ret_12', lambda x: (x > 0).mean() * 100)
    )
    
    print("\n3. LONG SIGNAL PERFORMANCE BY MACRO TREND (1 Hour Hold)")
    print(regime_stats.to_string())

    # ---------------------------------------------------------
    # ANALYSIS 4: Time of Day Seasonality
    # ---------------------------------------------------------
    extreme_longs['Hour'] = extreme_longs.index.hour
    time_stats = extreme_longs.groupby('Hour').agg(
        Count=('fwd_ret_12', 'size'),
        Win_Rate_1h=('fwd_ret_12', lambda x: (x > 0).mean() * 100)
    )
    
    time_stats['Win_Rate_1h'].plot(kind='bar', figsize=(10, 4), color='coral', alpha=0.8)
    plt.axhline(50, color='red', linestyle='--', linewidth=1)
    plt.title("1-Hour Win Rate of Dips (Z < -2.0) by Hour of Day")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("Hour of Day (Eastern Time)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    print("\nStatistical Analysis Complete.")



# ============================================
# CHANGE 2: In main(), after loading+prepare_data, add DB build
# ============================================

def main():
    FILEPATH = f"Data_5min/{TICKER}_5min.parquet"
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


    df_ind = compute_all_indicators(df)
    df_ind = drop_invalid_indicator_rows(df_ind)

    df_ind = generate_intraday_signals(df_ind, start_flat=True)

    FEATURE_COLS = [
        # core indicators
        "VWAP", "VWAP_Std", "Vol_Ratio", "RSI",
        "EMA_100", "EMA_200", "EMA_200_60min",
        "VWAP_Distance", "VWAP_Distance_Prev", "VWAP_Slope_10", "VWAP_Slope_30",
        # ML features
        "ret_1", "ret_3", "ret_6", "oc_ret", "hl_range", "range_z",
        "EMA100_Dist", "EMA200_60_Dist",
        "min_since_open",
    ]



    '''
    ml_dat = generate_data(df_ind, start_flat=True)

    X_db, y_db = build_trade_database_from_signals(
        ml_dat,
        feature_cols=FEATURE_COLS,
        cost_bps=TRADING_COST_BPS,
    )

    print(f"Trade DB built: X={X_db.shape}, y={y_db.shape}")
    if not y_db.empty:
        X_db.to_parquet(f"ML_Data/{TICKER}_trade_features.parquet", index=False)
        y_db.to_parquet(f"ML_Data/{TICKER}_trade_outcomes.parquet", index=False)
        print(f"Saved: {TICKER}_trade_features.parquet, {TICKER}_trade_outcomes.parquet")'''

    # ----------------------------
    # Existing: walk-forward backtest (baseline evaluation)
    # ----------------------------


    #wfv_df, trade_df = walk_forward_backtest(df, test_window_days=30, cost_bps=TRADING_COST_BPS, initial_capital=INITIAL_CAPITAL, feature_cols=FEATURE_COLS)
    
    print("\n" + "="*60)
    print("RUNNING BASELINE STRATEGY (NO ML)")
    print("="*60)
    base_wfv_df, base_trade_df = walk_forward_backtest(
        df, 1095, 30, TRADING_COST_BPS, INITIAL_CAPITAL, feature_cols=FEATURE_COLS, use_ml=False
    )

    print("\n" + "="*60)
    print("RUNNING MACHINE LEARNING STRATEGY")
    print("="*60)
    ml_wfv_df, ml_trade_df = walk_forward_backtest(
        df, 1095, 30, TRADING_COST_BPS, INITIAL_CAPITAL, feature_cols=FEATURE_COLS, use_ml=True
    )


    if base_wfv_df.empty or ml_wfv_df.empty:
        print("ERROR: Missing results from backtests")
        return

    print("\nCalculating performance metrics...")

    base_metrics, base_wfv_df = calculate_capital_metrics(base_wfv_df, INITIAL_CAPITAL)
    ml_metrics, ml_wfv_df = calculate_capital_metrics(ml_wfv_df, INITIAL_CAPITAL)


    plt.figure(figsize=(12, 6))
    plt.plot(base_wfv_df.index, base_wfv_df['Total_Equity'], label=f"Baseline (Return: {base_metrics['Strategy_Total_Return']:.2f}%)", color='gray', alpha=0.7)
    plt.plot(ml_wfv_df.index, ml_wfv_df['Total_Equity'], label=f"ML Meta-Labeler (Return: {ml_metrics['Strategy_Total_Return']:.2f}%)", color='blue', linewidth=2)
    
    plt.axhline(INITIAL_CAPITAL, linestyle='--', color='black', alpha=0.5)
    plt.title(f"{TICKER} VWAP Mean Reversion: Baseline vs. ML Dynamic Sizing", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Equity ($)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
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

    #_, trade_df = calc_capital_returns_and_stats(wfv_df, INITIAL_CAPITAL, POSITION_SIZE_PCT, TRADING_COST_BPS, ml_model=meta_model, feature_cols=FEATURE_COLS)

    
    if not ml_trade_df.empty:
        print("\nTRADE STATISTICS")
        print("="*60)
        print(f"Total Trades:           {len(ml_trade_df):>8}")
        wins = ml_trade_df[ml_trade_df['Net_PnL'] > 0]
        losses = ml_trade_df[ml_trade_df['Net_PnL'] <= 0]
        win_rate = (len(wins) / len(ml_trade_df) * 100) if len(ml_trade_df) > 0 else 0
        print(f"Win Rate:               {win_rate:>8.2f}%")
        print(f"Winners / Losers:       {len(wins):>4} / {len(losses):>4}")
        print(f"-" * 60)


        print(f"Average Trade (Net):    ${ml_trade_df['Net_PnL'].mean():>12,.2f}")

        print(f"Average Kelly Size:     %{ml_trade_df['Kelly_Size'].mean():>12.2f}")
        print(f"Max Kelly Size:         %{ml_trade_df['Kelly_Size'].max():>12.2f}")


        print(f"Total Gross P&L:        ${ml_trade_df['Gross_PnL'].sum():>12,.0f}")
        print(f"Total Net P&L:          ${ml_trade_df['Net_PnL'].sum():>12,.0f}")
        print(f"Median Trade (Net):     ${ml_trade_df['Net_PnL'].median():>12,.2f}")
        print(f"Best Trade (Net):       ${ml_trade_df['Net_PnL'].max():>12,.2f}")
        print(f"Worst Trade (Net):      ${ml_trade_df['Net_PnL'].min():>12,.2f}")
        print(f"-" * 60)
        if len(wins) > 0:
            print(f"Avg Winner:             ${wins['Net_PnL'].mean():>12,.2f}")
            print(f"Avg Kelly Winner:       %{wins['Kelly_Size'].mean():>12.2f}")
        if len(losses) > 0:
            print(f"Avg Loser:              ${losses['Net_PnL'].mean():>12,.2f}")
            print(f"Avg Kelly Loser:        %{losses['Kelly_Size'].mean():>12.2f}")
        avg_holding = ml_trade_df['Holding_Bars'].mean()
        avg_holding_hours = avg_holding * 5 / 60
        print(f"Avg Holding Period:     {avg_holding_hours:>12.1f} hours")
        print(f"-" * 60)
        print("Exit Reasons:")
        exit_counts = ml_trade_df['Exit_Reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = (count / len(ml_trade_df) * 100)
            print(f"  {reason:<20} {count:>4} ({pct:>5.1f}%)")

        print("="*60)
    print("\nBacktest complete!")




    # --- Block bootstrap Monte Carlo (preserves streaks)
    if not ml_trade_df.empty and len(ml_trade_df) >= 50:
        BLOCK_SIZE = 10  # try 5, 10, 20; larger = more regime persistence
        mc_paths, mc_summary = monte_carlo_block_bootstrap_from_trade_returns(
            ml_trade_df,
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
