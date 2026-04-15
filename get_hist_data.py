import os
import requests
from dotenv import load_dotenv
import datetime as dt
import pandas as pd
import time
import io

# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

SYMBOLS = ["VIX"]
TIMEFRAME = "1D"
FEED = "sip"
ADJUSTMENT = "all"

START_DATE = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc)
END_DATE   = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)


CHUNK_DAYS = 30
OUT_DIR = "Data_1D" if TIMEFRAME == "1D" else "Data"

os.makedirs(OUT_DIR, exist_ok=True)




def iso(dt_obj):
    return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")


def download_vix_from_cboe():
    response = requests.get(CBOE_VIX_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df.columns = df.columns.str.strip().str.lower()

    required_cols = {"date", "open", "high", "low", "close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Cboe VIX CSV is missing columns: {sorted(missing_cols)}")

    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = 0
    df["symbol"] = "VIX"

    return df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]






for symbol in SYMBOLS:
    print(f"\nDownloading {symbol}...")
    if symbol == "VIX":
        df = download_vix_from_cboe()
    else:
        all_rows = []
        chunk_start = START_DATE

        while chunk_start < END_DATE:
            chunk_end = min(chunk_start + dt.timedelta(days=CHUNK_DAYS), END_DATE)

            params = {
                "symbols": symbol,
                "timeframe": TIMEFRAME,
                "start": iso(chunk_start),
                "end": iso(chunk_end),
                "limit": 10000,
                "adjustment": ADJUSTMENT,
                "feed": FEED,
                "sort": "asc"
            }

            next_page_token = None

            while True:
                if next_page_token:
                    params["page_token"] = next_page_token

                response = requests.get(BASE_URL, headers=HEADERS, params=params)
                response.raise_for_status()
                data = response.json()

                bars = data.get("bars", {}).get(symbol, [])
                if not bars:
                    break

                for bar in bars:
                    all_rows.append({
                        "timestamp": bar["t"],
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                        "symbol": symbol
                    })

                next_page_token = data.get("next_page_token")
                if not next_page_token:
                    break

                time.sleep(0.2)  # polite pacing

            chunk_start = chunk_end
            time.sleep(0.5)

        df = pd.DataFrame(all_rows)

    if df.empty:
        print(f"  No data for {symbol}")
        continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()

    out_path = os.path.join(
        OUT_DIR,
        f"{symbol}_{TIMEFRAME.lower()}.parquet"
    )

    df.to_parquet(out_path, index=True)
    print(f"  Saved {len(df):,} rows -> {out_path}")
