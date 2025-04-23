import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def get_tesla_data(interval='1d'):
    # Get Tesla stock data
    tesla = yf.Ticker("TSLA")
    
    # Get historical data for the last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch the data with specified interval
    data = tesla.history(start=start_date, end=end_date, interval=interval)
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Rename columns to match the desired format
    data = data.rename(columns={
        'Date': 'start',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # data = data.drop(columns=['Dividends', 'Stock Splits'])

    # data['start'] = data['start'].astype(np.int64) // 10**9
    
    # Save to CSV with interval in filename
    filename = f'TSLA_{interval}.csv'
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Available intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    get_tesla_data(interval='90m')  # Example: 1-hour candles
