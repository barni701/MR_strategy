import pandas as pd
import numpy as np
import os
from glob import glob
from statsmodels.tsa.stattools import adfuller
import warnings

# Suppress warnings from statsmodels if series are perfectly stationary/non-stationary
warnings.filterwarnings("ignore")

# ==========================================
# Metric Calculations
# ==========================================

def get_hurst_exponent(time_series, max_lag=20):
    """
    Returns the Hurst Exponent of the time series vector ts.
    H < 0.5 = Mean Reverting
    H = 0.5 = Random Walk
    H > 0.5 = Trending
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    
    # Use linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0

def get_half_life(time_series):
    """
    Calculates the half-life of mean reversion using an Ornstein-Uhlenbeck process.
    Formula: dy(t) = -lambda * (y(t) - mean) * dt + sigma * dW
    """
    # 1. Create the lag and delta
    ts_lag = np.roll(time_series, 1)
    ts_lag[0] = 0
    ts_ret = time_series - ts_lag
    ts_ret[0] = 0
    
    # 2. Run regression of delta (y) against lag (x)
    ts_lag = ts_lag[1:]
    ts_ret = ts_ret[1:]
    
    # Using numpy polyfit for speed (degree 1)
    slope, intercept = np.polyfit(ts_lag, ts_ret, 1)
    
    # 3. Calculate Half Life
    lam = -slope
    
    if lam <= 0:
        return np.inf # Non-stationary / Trending (No mean reversion)
        
    half_life = -np.log(2) / slope
    return half_life

def analyze_file(filepath):
    try:
        # Load Data
        df = pd.read_parquet(filepath)
        
        # Ensure 'close' exists
        if 'close' not in df.columns:
            return None
            
        # Use the last 50,000 bars (approx 2.5 years of 5-min data)
        subset_price = df['close'].tail(50000).values
        
        # --- CRITICAL UPDATE: Analyze Raw Log Prices ---
        # We use Log Prices so the variance scales properly over time.
        log_price = np.log(subset_price)
        
        # 1. Hurst Exponent
        hurst = get_hurst_exponent(log_price, max_lag=50)
        
        # 2. Half-Life (in Bars)
        half_life_bars = get_half_life(log_price)
        
        # 3. Augmented Dickey-Fuller Test (The Statistical Gold Standard)
        # Tests the null hypothesis that a unit root is present (random walk).
        # A p-value < 0.05 indicates the series is stationary (mean-reverting).
        adf_result = adfuller(log_price, maxlag=1)
        adf_pvalue = adf_result[1]
        
        # 4. Volatility (Annualized for 5-minute bars)
        # 252 trading days * 78 five-minute bars per day = 19,656 bars per year
        log_rets = np.diff(log_price)
        min_vol = np.std(log_rets)
        ann_vol = min_vol * np.sqrt(252 * 78) * 100 # In Percent
        
        avg_price = np.mean(subset_price)
        filename = os.path.basename(filepath)
        symbol = filename.split('_')[0] 
        
        return {
            'Symbol': symbol,
            'Price': f"${avg_price:.2f}",
            'Hurst': round(hurst, 3),
            'Half_Life_Bars': round(half_life_bars, 1) if not np.isinf(half_life_bars) else "Infinite",
            'ADF_pvalue': round(adf_pvalue, 4),
            'Ann_Volatility': f"{ann_vol:.1f}%",
            'Score': 0 # Placeholder
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# ==========================================
# Main Execution
# ==========================================
def main():
    # PATH TO YOUR DATA
    DATA_DIR = "Data_5min" 
    
    files = glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"Found {len(files)} files. Analyzing structural properties of last 50k bars...")
    print("-" * 100)
    
    results = []
    
    for f in files:
        res = analyze_file(f)
        if res:
            # SCORING ALGORITHM
            # 1. Hurst should be LOW (closer to 0.5 or below)
            # 2. ADF p-value should be LOW (stronger rejection of random walk)
            # 3. Volatility should be HIGH (pays the trading costs)
            
            h = res['Hurst']
            vol = float(res['Ann_Volatility'].strip('%'))
            pval = res['ADF_pvalue']
            
            # If a stock is perfectly trending (Hurst > 0.65 or infinite half-life), penalize it heavily.
            if res['Half_Life_Bars'] == "Infinite" or h > 0.65:
                score = 0
            else:
                # We want high volatility, but heavily punish high Hurst and high p-values
                # We add 0.001 to pval to prevent division by zero
                score = vol / (h * (pval + 0.001))
                
            res['Score'] = round(score, 2)
            results.append(res)
            print(f"Processed {res['Symbol']}")

    # Create DataFrame and Sort
    res_df = pd.DataFrame(results)
    
    if not res_df.empty:
        # Sort by Custom Score (Descending)
        res_df = res_df.sort_values(by='Score', ascending=False)
        
        print("\n" + "="*100)
        print("INSTITUTIONAL MEAN REVERSION SCREENER (RAW PRICE ANALYSIS)")
        print("="*100)
        print("Guide:")
        print(" - Hurst: < 0.5 is Mean Reverting, ~0.5 is Random Walk, > 0.5 is Trending")
        print(" - ADF_pvalue: < 0.05 mathematically proves mean reversion (Stationarity)")
        print(" - Ann_Volatility: Needs to be high enough to outpace transaction costs")
        print(" - Score: Custom ratio optimizing for Volatility while penalizing non-stationarity")
        print("-" * 100)
        print(res_df.to_string(index=False))
        print("="*100)
    else:
        print("No results found.")

if __name__ == "__main__":
    main()