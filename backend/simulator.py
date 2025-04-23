import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

# store the data
data_file = "./BTC_USD.csv"

# Read data from CSV
df = pd.read_csv(data_file)
dates = df.iloc[:, 0].values
price_values = df.iloc[:, 1].values

def calculate_moving_averages(prices, window_sizes):
    """Pre-calculate moving averages for all window sizes"""
    if os.path.exists('moving_averages.json'):
        with open('moving_averages.json', 'r') as f:
            moving_averages = json.load(f)
        return moving_averages

    moving_averages = {}
    for window_size in window_sizes:
        # Use pandas rolling mean for efficient calculation
        ma = pd.Series(prices).rolling(window=window_size, min_periods=1).mean().values
        moving_averages[window_size] = ma

    # with open('moving_averages.json', 'w') as f:
    #     json.dump(moving_averages, f)
    return moving_averages

def calculate_weighted_moving_averages(prices, window_sizes):
    """Pre-calculate weighted moving averages for all window sizes"""
    if os.path.exists('weighted_moving_averages.json'):
        with open('weighted_moving_averages.json', 'r') as f:
            weighted_moving_averages = json.load(f)
        return weighted_moving_averages

    weighted_moving_averages = {}
    for window_size in window_sizes:
        # Create weights array
        weights = np.arange(1, window_size + 1)
        weights = weights / weights.sum()  # Normalize weights
        
        # Use pandas rolling with weights
        weighted_ma = pd.Series(prices).rolling(
            window=window_size,
            min_periods=1
        ).apply(lambda x: np.sum(x * weights[:len(x)]), raw=True).values
        
        weighted_moving_averages[window_size] = weighted_ma

    with open('weighted_moving_averages.json', 'w') as f:
        data_to_save = {}
        for window_size in weighted_moving_averages:
            data_to_save[window_size] = weighted_moving_averages[window_size].tolist()
        json.dump(data_to_save, f)
    return weighted_moving_averages

def run_simulation(buy_percent, sell_percent, window_size, moving_averages, weighted_moving_averages, start_epoch, end_epoch):
    """Optimized simulation function using vectorized operations"""

    # Convert epochs to indices
    start_idx = np.searchsorted(dates, start_epoch)
    end_idx = np.searchsorted(dates, end_epoch)
    
    # Slice data for the specified time period
    dates_slice = dates[start_idx:end_idx]
    prices_slice = price_values[start_idx:end_idx]
    ma_slice = moving_averages[window_size][start_idx:end_idx]
    weighted_ma_slice = weighted_moving_averages[str(10 * 3600 * 24)][start_idx:end_idx]
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
# buy_percentages = np.arange(0.97, 1.00, 0.005)  # Reduced granularity
# sell_percentages = np.arange(1.01, 1.03, 0.005)  # Reduced granularity
# window_sizes = np.array([3, 5, 10, 20, 30, 60, 120])  # Reduced window sizes to test

buy_percentages = np.array([0.98])
sell_percentages = np.array([1.02])
window_sizes = np.array([60])  # Reduced window sizes to test
time_windows = 1
time_window_length_months = 12
time_window_length = time_window_length_months * 60 * 60 * 24 * 30

# Pre-calculate moving averages for all window sizes
print("\nPre-calculating moving averages...")
moving_averages = calculate_moving_averages(price_values, window_sizes)

print("\nPre-calculating weighted moving averages...")
weighted_moving_averages = calculate_weighted_moving_averages(price_values, [10 * 3600 * 24])

# Initialize best results tracking
best_results = {
    'return': float('-inf'),
    'buy_percent': None,
    'sell_percent': None,
    'window_size': None,
    'trades': None,
    'portfolio_values': None
}

print("\nRunning simulations with different parameters...")
total_simulations = len(buy_percentages) * len(sell_percentages) * len(window_sizes)
simulation_count = 0

# Create a progress bar for the entire simulation
pbar = tqdm(total=total_simulations, desc="Simulation Progress")

for buy_percent in buy_percentages:
    for sell_percent in sell_percentages:
        if buy_percent >= sell_percent:
            continue
            
        for window_size in window_sizes:
            simulation_count += 1
            pbar.update(1)
            
            sim_results = []
            for time_window in range(time_windows):
                start_time = 1650659523
                end_time = 1682195523
                time_range = end_time - start_time
                lerp_amount = time_window / time_windows if time_windows > 1 else 0
                selected_time = start_time + (time_range * lerp_amount)
                total_return, trades, portfolio_values = run_simulation(
                    buy_percent, sell_percent, window_size, moving_averages, weighted_moving_averages, selected_time, selected_time + time_window_length
                )
                sim_results.append((total_return, trades, portfolio_values))
            
            returns, trades_list, portfolio_values_list = zip(*sim_results)
            total_return = sum(returns) / len(returns)
            trade_avg = sum([len(t) for t in trades_list]) / len(trades_list)
            
            # Instead of averaging portfolio values, we'll use the last simulation's values
            portfolio_values = portfolio_values_list[-1]  # Use the last simulation's values

            if total_return > best_results['return']:
                best_results.update({
                    'return': total_return,
                    'buy_percent': buy_percent,
                    'sell_percent': sell_percent,
                    'window_size': window_size,
                    'trades': trade_avg,
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
print(f"Total Return 12 Months: {best_results['return'] * (12 / time_window_length_months):.2f}%")
print(f"Number of Trades: {(best_results['trades'])}")

print(dates.shape)

# Visualize results
plt.figure(figsize=(12, 8))
plt.plot(dates, price_values, label='BTC Price', alpha=0.5)

# Add buy and sell markers
buy_dates = [trade['date'] for trade in trades if trade['type'] == 'BUY']
buy_prices = [trade['price'] for trade in trades if trade['type'] == 'BUY']
sell_dates = [trade['date'] for trade in trades if trade['type'] == 'SELL']
sell_prices = [trade['price'] for trade in trades if trade['type'] == 'SELL']

try:
    # plt.plot(dates, moving_averages[window_size], label=f'{window_size}-day Moving Average', linestyle='--', color='purple')
    # plt.plot(dates, moving_averages[window_size] * best_results["buy_percent"], label=f'Buy Line ({best_results["buy_percent"]*100:.2f}% of MA)', linestyle='--', color='green', alpha=0.7)
    # plt.plot(dates, moving_averages[window_size] * best_results["sell_percent"], label=f'Sell Line ({best_results["sell_percent"]*100:.2f}% of MA)', linestyle='--', color='red', alpha=0.7)

    plt.plot(dates, weighted_moving_averages[str(10 * 3600 * 24)], label=f'Weighted Moving Average', linestyle='--', color='purple')

    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', alpha=0.7)
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', alpha=0.7)

    plt.title('Bitcoin Trading Simulation')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('btc_simulation_results.png')
    plt.show()

except KeyboardInterrupt:
    print("Simulation interrupted by user")

# Print simulation results
print("\nBitcoin Trading Simulation Results:")
print("===================================")
# print(f"Initial Balance: ${initial_balance:,.2f}")
# print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Number of Trades: {len(trades)}")