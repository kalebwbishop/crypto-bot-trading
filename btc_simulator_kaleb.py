import requests
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import yfinance as yf
import os
import pandas as pd

# Calculate date range (past 5 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 1)

# List of cryptocurrencies to simulate
# cryptocurrencies = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOT-USD", "LINK-USD", "BCH-USD", "LTC-USD"]
cryptocurrencies = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD"]

# Add S&P 500 to the list of assets
assets = cryptocurrencies + []  # ^GSPC is the Yahoo Finance symbol for S&P 500

# Function to check if data file exists and is recent (less than 1 day old)
def should_fetch_new_data(symbol):
    data_file = f"{symbol.replace('-', '_').replace('^', '')}_price_data.csv"
    if not os.path.exists(data_file):
        return True
    
    # Check if file is less than 1 day old
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(data_file))
    if datetime.now() - file_mod_time > timedelta(days=1):
        return True
    
    # Check if the file contains the correct interval data
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        # If the file is empty or doesn't have enough data points, fetch new data
        if len(df) < 10:
            return True
    except:
        return True
    
    return True

# Function to fetch or load cryptocurrency price data
def fetch_crypto_data(symbol):
    data_file = f"{symbol.replace('-', '_').replace('^', '')}_price_data.csv"
    interval = "1d"  # Data interval: "1d" for daily, "4h" for 4-hour intervals
    
    if should_fetch_new_data(symbol):
        print(f"Fetching {symbol} price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} with {interval} intervals...")
        asset = yf.Ticker(symbol)
        hist = asset.history(start=start_date, end=end_date, interval=interval)
        
        # Save data to CSV
        hist.to_csv(data_file)
        print(f"Data saved to {data_file}")
    else:
        print(f"Loading {symbol} price data from {data_file}...")
        hist = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Extract price data
    dates = hist.index.to_list()
    price_values = hist['Close'].to_list()
    
    print(f"Successfully loaded {len(dates)} data points of {symbol} price data")
    
    # Check if we have enough data
    if len(dates) < 10:
        print(f"Error: Not enough data points for {symbol}. Please check your internet connection and try again.")
        return None, None
    
    return dates, price_values

# Function to run simulation with specific buy and sell percentages
def run_simulation(dates, price_values, buy_percent, sell_percent, days_threshold, strategy1_percent, window_size=30):
    # Simulation parameters
    initial_balance = 10000  # Starting with $10,000
    crypto_balance = 0
    usd_balance = initial_balance
    trades = []
    portfolio_values = []
    days_in_market = 0

    # Calculate moving average for the line
    # window_size is now passed as a parameter

    # Function to calculate moving average up to a specific index
    def calculate_moving_average(prices, up_to_index, window_size):
        if up_to_index < window_size:
            # If we don't have enough data, use the average of what we have
            return sum(prices[:up_to_index+1]) / (up_to_index+1)
        else:
            return sum(prices[up_to_index-window_size+1:up_to_index+1]) / window_size

    # Initialize the first moving average
    current_moving_average = calculate_moving_average(price_values, 0, window_size)
    moving_averages = [current_moving_average]
    buy_lines = [current_moving_average * buy_percent]  # Buy line at specified percent of moving average
    sell_lines = [current_moving_average * sell_percent]  # Sell line at specified percent of moving average

    for i in range(1, len(price_values)):
        current_price = price_values[i]
        current_date = dates[i]
        
        # Only recalculate moving average when looking to buy
        if usd_balance > 0:
            current_moving_average = calculate_moving_average(price_values, i, window_size)
        moving_averages.append(current_moving_average)
        buy_lines.append(current_moving_average * buy_percent)
        sell_lines.append(current_moving_average * sell_percent)
            
        # Trading logic
        buy_price = current_moving_average * buy_percent
        if current_price <= buy_price and usd_balance > 0:
            # Calculate crypto amount to buy
            crypto_to_buy = usd_balance / current_price
            
            # Execute trade
            usd_balance -= usd_balance
            crypto_balance += crypto_to_buy
            days_in_market = 0  # Reset days in market counter
            
            trades.append({
                'date': current_date,
                'type': 'BUY',
                'price': current_price,
                'amount': crypto_to_buy,
                'value': usd_balance
            })

        # Sell decision
        sell_price = current_moving_average * sell_percent
        if current_price >= sell_price and crypto_balance > 0:
            # Calculate USD value
            usd_value = crypto_balance * current_price
            
            # Execute trade
            usd_balance += usd_value
            crypto_balance -= crypto_balance
            
            trades.append({
                'date': current_date,
                'type': 'SELL',
                'price': current_price,
                'amount': crypto_balance,
                'value': usd_value
            })

        # Update days in market
        if crypto_balance > 0:
            days_in_market += 1
        
        # Calculate total portfolio value (USD + crypto value)
        portfolio_value = usd_balance + (crypto_balance * current_price)
        portfolio_values.append(portfolio_value)

    # Calculate performance metrics
    final_portfolio_value = portfolio_values[-1]
    total_return = ((final_portfolio_value - initial_balance) / initial_balance) * 100  # Multiply by 100 to convert to percentage
    
    return total_return, trades, portfolio_values

# Define the ranges of buy and sell percentages to test
buy_percentages = np.arange(0.60, 1.00, 0.005)  # From 60% to 100%
sell_percentages = np.arange(1.01, 1.15, 0.005)  # From 101% to 115%
window_sizes = np.arange(70, 80, 10)  # From 70 to 80

# Dictionary to store results for each cryptocurrency
results = {}

# Run simulation for each cryptocurrency
for symbol in assets:
    print(f"\n{'='*50}")
    print(f"SIMULATING {symbol}")
    print(f"{'='*50}")
    
    # Fetch data for this cryptocurrency
    dates, price_values = fetch_crypto_data(symbol)
    if dates is None or price_values is None:
        print(f"Skipping {symbol} due to insufficient data")
        continue
    
    best_return = float('-inf')
    best_buy_percent = None
    best_sell_percent = None
    best_window_size = None
    best_trades = None
    best_portfolio_values = None

    print("\nRunning simulations with different parameters...")
    for buy_percent in buy_percentages:
        for sell_percent in sell_percentages:
            for window_size in window_sizes:
                if buy_percent >= sell_percent:
                    continue  # Skip invalid combinations where buy price is higher than sell price
                    
                total_return, trades, portfolio_values = run_simulation(dates, price_values, buy_percent, sell_percent, 0, 1.0, window_size)
                
                if total_return > best_return:
                    best_return = total_return
                    best_buy_percent = buy_percent
                    best_sell_percent = sell_percent
                    best_window_size = window_size
                    best_trades = trades
                    best_portfolio_values = portfolio_values
                    
                print(f"Buy: {buy_percent*100:.2f}%, Sell: {sell_percent*100:.2f}%, Window: {window_size}, Return: {total_return:.2f}%")

    print("\nBest Performance:")
    print(f"Buy Percentage: {best_buy_percent*100:.2f}%")
    print(f"Sell Percentage: {best_sell_percent*100:.2f}%")
    print(f"Window Size: {best_window_size}")
    print(f"Total Return: {best_return:.2f}%")
    
    # Store results
    results[symbol] = {
        'buy_percent': best_buy_percent,
        'sell_percent': best_sell_percent,
        'window_size': best_window_size,
        'return': best_return,
        'trades': best_trades,
        'portfolio_values': best_portfolio_values,
        'dates': dates,
        'price_values': price_values
    }
    
    # Visualize results with best parameters
    plt.figure(figsize=(12, 8))
    plt.plot(dates[1:], best_portfolio_values, label='Portfolio Value')
    plt.plot(dates, price_values, label=f'{symbol} Price', alpha=0.5)

    # Calculate and plot moving average and buy/sell lines for best parameters
    window_size = best_window_size
    moving_averages = []
    buy_lines = []
    sell_lines = []

    for i in range(len(price_values)):
        if i < window_size:
            ma = sum(price_values[:i+1]) / (i+1)
        else:
            ma = sum(price_values[i-window_size+1:i+1]) / window_size
        moving_averages.append(ma)
        buy_lines.append(ma * best_buy_percent)
        sell_lines.append(ma * best_sell_percent)

    plt.plot(dates, moving_averages, label=f'{window_size}-day Moving Average', linestyle='--', color='purple')
    plt.plot(dates, buy_lines, label=f'Buy Line ({best_buy_percent*100:.2f}% of MA)', linestyle='--', color='green', alpha=0.7)
    plt.plot(dates, sell_lines, label=f'Sell Line ({best_sell_percent*100:.2f}% of MA)', linestyle='--', color='red', alpha=0.7)

    # Add buy and sell markers
    buy_dates = [trade['date'] for trade in best_trades if trade['type'] == 'BUY']
    buy_prices = [trade['price'] for trade in best_trades if trade['type'] == 'BUY']
    sell_dates = [trade['date'] for trade in best_trades if trade['type'] == 'SELL']
    sell_prices = [trade['price'] for trade in best_trades if trade['type'] == 'SELL']

    try:
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', alpha=0.7)
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', alpha=0.7)

        plt.title(f'{symbol} Trading Simulation (Best Parameters)\nBuy: {best_buy_percent*100:.2f}%, Sell: {best_sell_percent*100:.2f}%, Return: {best_return:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{symbol.replace("-", "_")}_simulation_results.png')
        # plt.show()

    except KeyboardInterrupt:
        print("Simulation interrupted by user")

    # Print simulation results
    print(f"\n{symbol} Trading Simulation Results (Best Parameters):")
    print("===================================")
    print(f"Initial Balance: $10,000.00")
    print(f"Final Portfolio Value: ${best_portfolio_values[-1]:,.2f}")
    print(f"Total Return: {best_return:.2f}%")
    print(f"Return Doing Nothing: {(price_values[-1] / price_values[0] - 1) * 100:.2f}%")
    print(f"Number of Trades: {len(best_trades)}")

# Print summary of all cryptocurrencies
print("\n" + "="*50)
print("SUMMARY OF ALL CRYPTOCURRENCIES")
print("="*50)
print(f"{'Cryptocurrency':<15} {'Buy %':<10} {'Sell %':<10} {'Window':<10} {'Return %':<10} {'Buy & Hold %':<15}")
print("-"*70)

for symbol in cryptocurrencies:
    if symbol in results:
        buy_hold_return = (results[symbol]['price_values'][-1] / results[symbol]['price_values'][0] - 1) * 100
        print(f"{symbol:<15} {results[symbol]['buy_percent']*100:.2f}%    {results[symbol]['sell_percent']*100:.2f}%    {results[symbol]['window_size']:<10} {results[symbol]['return']:.2f}%     {buy_hold_return:.2f}%")

# Create a combined portfolio visualization
plt.figure(figsize=(14, 10))

# Define different line styles and colors for each cryptocurrency
styles = {
    'BTC-USD': {'color': '#F7931A', 'linestyle': '-', 'linewidth': 2},
    'ETH-USD': {'color': '#627EEA', 'linestyle': '--', 'linewidth': 2},
    'XRP-USD': {'color': '#23292F', 'linestyle': '-.', 'linewidth': 2},
    'SOL-USD': {'color': '#00FFA3', 'linestyle': ':', 'linewidth': 2},
    'ADA-USD': {'color': '#30A7D7', 'linestyle': '-', 'linewidth': 2},
    'DOT-USD': {'color': '#E86626', 'linestyle': '--', 'linewidth': 2},
    'LINK-USD': {'color': '#2A5ADA', 'linestyle': '-.', 'linewidth': 2},
    'BCH-USD': {'color': '#8DC351', 'linestyle': ':', 'linewidth': 2},
    'LTC-USD': {'color': '#A6A9AA', 'linestyle': '-', 'linewidth': 2},
    '^GSPC': {'color': '#000000', 'linestyle': '-', 'linewidth': 3},  # S&P 500 with black color and thicker line
    'GLD': {'color': '#FFD700', 'linestyle': '-', 'linewidth': 2}  # Gold with gold color and thicker line
}

# Plot portfolio values for each cryptocurrency
for symbol in assets:
    if symbol in results:
        portfolio_values = results[symbol]['portfolio_values']
        dates = results[symbol]['dates'][1:]
        
        # Calculate final return percentage
        final_return = results[symbol]['return']
        
        # Create a more descriptive label for S&P 500
        label = f'S&P 500 ({final_return:.1f}%)' if symbol == '^GSPC' else f'{symbol} ({final_return:.1f}%)'
        
        # Plot the line with custom style
        plt.plot(dates, portfolio_values, 
                label=label, 
                **styles.get(symbol, {}))
        
        # Add markers for highest and lowest points
        max_value = max(portfolio_values)
        min_value = min(portfolio_values)
        max_idx = portfolio_values.index(max_value)
        min_idx = portfolio_values.index(min_value)
        
        plt.scatter(dates[max_idx], max_value, 
                   color=styles[symbol]['color'], 
                   marker='^', s=100)
        plt.scatter(dates[min_idx], min_value, 
                   color=styles[symbol]['color'], 
                   marker='v', s=100)

plt.title('Combined Cryptocurrency Trading Simulation Results\nPortfolio Performance Comparison', 
         fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)

# Enhance the legend
plt.legend(title='Cryptocurrencies (Total Return)', 
          title_fontsize=12, 
          fontsize=10, 
          loc='upper left', 
          bbox_to_anchor=(1.05, 1))

# Enhance the grid
plt.grid(True, linestyle='--', alpha=0.7)

# Format y-axis with dollar signs and commas
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Rotate x-axis dates for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the enhanced visualization
plt.savefig('combined_crypto_simulation_results.png', 
            bbox_inches='tight', 
            dpi=300)

try:
    plt.show()
except KeyboardInterrupt:
    pass

# Calculate combined portfolio performance
print("\n" + "="*50)
print("COMBINED PORTFOLIO PERFORMANCE")
print("="*50)

# Calculate the best performing cryptocurrency
best_crypto = max(results, key=lambda x: results[x]['return'])
print(f"Best Performing Asset: {best_crypto} with {results[best_crypto]['return']:.2f}% return")

# Calculate the worst performing cryptocurrency
worst_crypto = min(results, key=lambda x: results[x]['return'])
print(f"Worst Performing Asset: {worst_crypto} with {results[worst_crypto]['return']:.2f}% return")

# Calculate average return for cryptocurrencies only
crypto_returns = {symbol: results[symbol]['return'] for symbol in cryptocurrencies if symbol in results}
if crypto_returns:
    avg_crypto_return = sum(crypto_returns.values()) / len(crypto_returns)
    print(f"Average Cryptocurrency Return: {avg_crypto_return:.2f}%")

# Calculate S&P 500 return if available
if '^GSPC' in results:
    sp500_return = results['^GSPC']['return']
    print(f"S&P 500 Return: {sp500_return:.2f}%")
    
    # Compare crypto performance to S&P 500
    if crypto_returns:
        crypto_vs_sp500 = avg_crypto_return - sp500_return
        print(f"Cryptocurrencies vs S&P 500: {crypto_vs_sp500:+.2f}%")

# Calculate average return across all assets
avg_return = sum(results[symbol]['return'] for symbol in results) / len(results)
print(f"Average Return (All Assets): {avg_return:.2f}%")

# Calculate buy and hold returns
buy_hold_returns = {symbol: (results[symbol]['price_values'][-1] / results[symbol]['price_values'][0] - 1) * 100 for symbol in results}
avg_buy_hold = sum(buy_hold_returns.values()) / len(buy_hold_returns)
print(f"Average Buy & Hold Return: {avg_buy_hold:.2f}%")

# Calculate strategy outperformance
outperformance = avg_return - avg_buy_hold
print(f"Strategy Outperformance: {outperformance:.2f}%")

# Calculate total number of trades across all assets
total_trades = sum(len(results[symbol]['trades']) for symbol in results)
print(f"\nTotal Number of Trades Across All Assets: {total_trades}")

