import numpy as np
import pandas as pd
from tqdm import tqdm
from get_data import get_data
import time
from transact import buy_btc, sell_btc

while True:
    success, df = get_data("BTC-USD", start_epoch=int(time.time()) - 3600 * 24 * 365)
    
    if not success:
        print("Error: Failed to fetch data. Please check your internet connection and try again.")
        exit(1)
    
    if df.empty:
        print("Error: No data available. Please check your internet connection and try again.")
        exit(1)
    
    print(f"Successfully loaded {len(df)} daily data points of Bitcoin price data")
    
    # Extract dates and price values from the DataFrame
    dates = df['start'].values
    price_values = df['close'].values
    
    current_price = price_values[-1]
    current_date = dates[-1]
    current_date_ts = pd.Timestamp(current_date).timestamp()
    
    # Check if we have enough data
    if len(dates) < 10:
        print("Error: Not enough data points. Please check your internet connection and try again.")
        exit(1)
    
    def calculate_moving_averages(prices, window_sizes):
        """Pre-calculate moving averages for all window sizes"""
        moving_averages = {}
        for window_size in window_sizes:
            # Use pandas rolling mean for efficient calculation
            ma = pd.Series(prices).rolling(window=window_size, min_periods=1).mean().values
            moving_averages[window_size] = ma
        return moving_averages
    
    def run_simulation(buy_percent, sell_percent, window_size, moving_averages, start_epoch, end_epoch):
        """Optimized simulation function using vectorized operations"""
        
        # Convert dates to Unix timestamps for comparison
        # For numpy.datetime64 objects, we need to convert to pandas Timestamp first
        dates_ts = np.array([pd.Timestamp(d).timestamp() for d in dates])
        
        # Convert epochs to indices
        start_idx = np.searchsorted(dates_ts, start_epoch)
        end_idx = np.searchsorted(dates_ts, end_epoch)
        
        # Slice data for the specified time period
        dates_slice = dates[start_idx:end_idx]
        prices_slice = price_values[start_idx:end_idx]
        ma_slice = moving_averages[window_size][start_idx:end_idx]
        
        # Simulation parameters
        initial_balance = 10000
        
        # Pre-calculate buy and sell prices
        buy_prices = ma_slice * buy_percent
        last_buy_price = 0
        sell_prices = ma_slice * sell_percent
        
        # Initialize arrays for tracking
        portfolio_values = np.zeros_like(prices_slice)
        btc_balance = np.zeros_like(prices_slice)
        usd_balance = np.zeros_like(prices_slice)
        usd_balance[0] = initial_balance
        
        # Track trades
        trades = []
        
        # Vectorized trading logic
        for i in range(1, len(prices_slice)):
            current_price = prices_slice[i]
            current_date = dates_slice[i]
            
            # Buy decision
            if current_price <= buy_prices[i] and usd_balance[i-1] > 0:
                btc_to_buy = usd_balance[i-1] / current_price
                btc_balance[i] = btc_to_buy
                usd_balance[i] = 0
                
                last_buy_price = current_price
                trades.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': current_price,
                    'amount': btc_to_buy,
                    'value': usd_balance[i-1]
                })
            else:
                btc_balance[i] = btc_balance[i-1]
                usd_balance[i] = usd_balance[i-1]
            
            # Sell decision
            if current_price >= sell_prices[i] and btc_balance[i] > 0:
                usd_value = btc_balance[i] * current_price
                usd_balance[i] += usd_value
                btc_balance[i] = 0
                
                if btc_balance[i-1] > last_buy_price:
                    trades.append({
                        'date': current_date,
                        'type': 'SELL',
                        'price': current_price,
                        'amount': btc_balance[i-1],
                        'value': usd_value
                })
            
            # Calculate portfolio value
            portfolio_values[i] = usd_balance[i] + (btc_balance[i] * current_price)
        
        # Calculate performance metrics
        final_portfolio_value = portfolio_values[-1]
        total_return = ((final_portfolio_value - initial_balance) / initial_balance) * 100
        
        return total_return, trades, portfolio_values
    
    # Define parameter ranges (optimized for faster testing)
    buy_percentages = np.array([0.98])
    sell_percentages = np.array([1.02])
    window_sizes = np.array([60])  # Single window size to test
    
    # Pre-calculate moving averages for all window sizes
    print("\nPre-calculating moving averages...")
    moving_averages = calculate_moving_averages(price_values, window_sizes)
    
    # Initialize best results tracking
    best_results = {
        'return': float('-inf'),
        'buy_percent': None,
        'sell_percent': None,
        'window_size': None,
        'trades': None,
        'portfolio_values': None
    }
    
    print("\nRunning simulation...")
    total_simulations = len(buy_percentages) * len(sell_percentages) * len(window_sizes)
    simulation_count = 0
    
    # Create a progress bar for the simulation
    pbar = tqdm(total=total_simulations, desc="Simulation Progress")
    
    for buy_percent in buy_percentages:
        for sell_percent in sell_percentages:
            if buy_percent >= sell_percent:
                continue
                
            for window_size in window_sizes:
                simulation_count += 1
                pbar.update(1)
                
                # Run single simulation for the entire year
                start_time = 1713661140
                end_time = current_date_ts
                total_return, trades, portfolio_values = run_simulation(
                    buy_percent, sell_percent, window_size, moving_averages, start_time, end_time
                )
                
                if total_return > best_results['return']:
                    best_results.update({
                        'return': total_return,
                        'buy_percent': buy_percent,
                        'sell_percent': sell_percent,
                        'window_size': window_size,
                        'trades': len(trades),
                        'trades_list': trades,
                        'portfolio_values': portfolio_values
                    })
                    
                pbar.set_postfix({
                    'Buy': f"{buy_percent:.2f}", 
                    'Sell': f"{sell_percent:.2f}", 
                    'Window': window_size, 
                    'Return': f"{total_return:.2f}%"
                })
    
    pbar.close()
    
    # Print best results
    print("\nBest Performance:")
    print(f"Buy Percentage: {best_results['buy_percent']:.2f}")
    print(f"Sell Percentage: {best_results['sell_percent']:.2f}")
    print(f"Window Size: {best_results['window_size']}")
    print(f"Total Return: {best_results['return']:.2f}%")
    print(f"Number of Trades: {best_results['trades']}")
    
    print(best_results['trades_list'][-1]['date'], current_date)
    if (best_results['trades_list'][-1]['date'] == current_date):
        if best_results['trades_list'][-1]['type'] == 'BUY':
            buy_btc()
        else:
            sell_btc()
    
    if current_date_ts + 55 - time.time() > 0:
        time.sleep(current_date_ts + 55 - time.time())
    
    while time.time() < current_date_ts + 62:
        print(current_date_ts + 62 - time.time())
        time.sleep(0.5)
