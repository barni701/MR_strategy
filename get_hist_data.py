import os
import requests
from dotenv import load_dotenv
import datetime as dt
import pandas as pd
import time

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

SYMBOLS = ["AAPL", "LMT", "AMZN", "VZ"]   # <-- add more here
TIMEFRAME = "1Min"
FEED = "sip"
ADJUSTMENT = "raw"

START_DATE = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
END_DATE   = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)


CHUNK_DAYS = 30
OUT_DIR = "alpaca_1min_parquet"

os.makedirs(OUT_DIR, exist_ok=True)




def iso(dt_obj):
    return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")






for symbol in SYMBOLS:
    print(f"\nDownloading {symbol}...")
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




    # Save to Parquet

    df = pd.DataFrame(all_rows)

    if df.empty:
        print(f"  No data for {symbol}")
        continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)

    out_path = os.path.join(
        OUT_DIR,
        f"{symbol}_1min_2019_2024.parquet"
    )

    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df):,} rows → {out_path}")
