from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_data(start_date: datetime, end_date: datetime):
    def calculate_moving_averages(df):
        window_sizes = [10, 20, 50, 60, 100, 200]
        # Sort by datetime to ensure correct calculation
        df = df.sort_values(by='datetime')
        for window_size in window_sizes:
            df[f'moving_average_{window_size}'] = df['close'].rolling(window=window_size).mean()
        return df

    # Convert string dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Check if CSV exists and read it
    csv_path = 'BTC_USD.csv'
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
        
        # Check if we have all the data we need
        if (existing_df['datetime'].min() <= start_date and 
            existing_df['datetime'].max() >= end_date):
            # We have all the data we need
            mask = (existing_df['datetime'] >= start_date) & (existing_df['datetime'] <= end_date)
            return existing_df[mask]
        
        # If we need more recent data, only fetch from the last available date
        if existing_df['datetime'].max() < end_date:
            fetch_start = existing_df['datetime'].max()
            fetch_end = end_date
        else:
            # If we need older data, fetch from start_date to the earliest available date
            fetch_start = start_date
            fetch_end = existing_df['datetime'].min()
    else:
        # If no CSV exists, fetch the entire range
        fetch_start = start_date
        fetch_end = end_date
        existing_df = None

    # Fetch new data from Alpaca
    client = CryptoHistoricalDataClient()

    # Creating request object
    request_params = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USD"],
        timeframe=TimeFrame.Minute,
        start=fetch_start,
        end=fetch_end
    )

    # Retrieve daily bars for Bitcoin in a DataFrame
    btc_bars = client.get_crypto_bars(request_params)

    # Convert to dataframe
    df = btc_bars.df

    # Add epoch timestamp column by accessing the timestamp from the index
    df['datetime'] = pd.to_datetime(df.index.get_level_values('timestamp').astype('int64') // 10**9, unit='s')

    # Drop the unneeded columns
    df = df.drop(columns=['open', 'high', 'low', 'volume', 'trade_count', 'vwap'])

    # Sort by epoch
    df = df.sort_values(by='datetime')

    # If we have existing data, merge with it
    if existing_df is not None:
        # Combine old and new data, removing duplicates
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df.drop_duplicates(subset=['datetime'])
        combined_df = combined_df.sort_values(by='datetime')
        
        # Recalculate moving averages for the entire dataset
        combined_df = calculate_moving_averages(combined_df)
        
        # Save the combined data
        combined_df.to_csv(csv_path, index=False)
        
        # Return only the requested date range
        mask = (combined_df['datetime'] >= start_date) & (combined_df['datetime'] <= end_date)
        return combined_df[mask]
    else:
        # Calculate moving averages for new data
        df = calculate_moving_averages(df)
        
        # Save new data
        df.to_csv(csv_path, index=False)
        return df

if __name__ == "__main__":
    get_data('2020-01-01', '2023-01-01')
