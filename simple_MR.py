import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def signal_generator(df):
    '''Generates trading signals based on moving average crossover strategy.
    Long only!'''


    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    #df['SMA_50'] = df['close'].rolling(window=50).mean()
    #df['SMA_200'] = df['close'].rolling(window=200).mean()

    prev_close = df['close'].shift(1)


    df['Signal'] = np.where(df['close'] > df['EMA20'], 1, 0)

    df['Position'] = df['Signal'].shift(1)

    return df







if __name__ == "__main__":


    print("Loading VZ 1-Minute data from Parquet...")
    df = pd.read_parquet("Data/VZ_1min_2019_2024.parquet", columns=["timestamp", "close"])

    print("Generating signals...")
    df = signal_generator(df)

    df = df[
        (df["timestamp"] >= "2022-01-01") &
        (df["timestamp"] <  "2022-01-07")
    ].copy()

    # Example plot
    plt.figure(figsize=(14,7))
    plt.plot(df['timestamp'], df['close'], label='Close Price')
    plt.plot(df['timestamp'], df['EMA20'], label='EMA20')
    plt.title('VZ Close Price with EMA20 and Trading Signals')
    plt.xlabel('Time')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()


