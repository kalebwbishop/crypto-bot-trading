import requests
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import ta

# --- CONFIGURATION ---
RISK_PER_TRADE = 0.01  # 1% per trade
INITIAL_BALANCE = 10000  # Starting with $10,000

# Grid Trading Configuration
GRID_PERCENT_RANGE = 5  # 5% above/below initial
NUM_GRIDS = 20
TRAILING_STOP_TRIGGER = 1.05  # 5% pump activates trailing stop
TRAILING_STOP_OFFSET = 0.02   # 2% under highest peak

# Calculate date range (past year)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Fetch Bitcoin price data from CoinGecko API
url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
response = requests.get(url)
data = response.json()

# Extract price data
prices = data['prices']
dates = [datetime.fromtimestamp(timestamp/1000) for timestamp, _ in prices]
price_values = [price for _, price in prices]

# Create DataFrame for easier manipulation
df = pd.DataFrame({
    'timestamp': dates,
    'close': price_values,
    'high': price_values,  # Using close as high for simplicity
    'low': price_values,   # Using close as low for simplicity
    'volume': [random.uniform(1000, 5000) for _ in range(len(price_values))]  # Simulated volume
})

# Calculate indicators
df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
df['rsi'] = ta.momentum.rsi(df['close'], window=14)
bb = ta.volatility.BollingerBands(df['close'])
df['upper_band'] = bb.bollinger_hband()
df['middle_band'] = bb.bollinger_mavg()
df['lower_band'] = bb.bollinger_lband()
df['ema8'] = ta.trend.ema_indicator(df['close'], window=8)

# Grid Trading Functions
def calculate_order_size(price, capital, num_grids):
    return (capital / num_grids) / price

def rebuild_grid(center_price, capital, num_grids, grid_percent_range):
    lower_bound = center_price * (1 - grid_percent_range/100)
    upper_bound = center_price * (1 + grid_percent_range/100)
    spacing = (upper_bound - lower_bound) / num_grids
    order_book = []

    for i in range(num_grids + 1):
        price = lower_bound + i * spacing
        side = 'buy' if price < center_price else 'sell'
        order_book.append({'price': price, 'side': side})
    
    return order_book, lower_bound, upper_bound, spacing

# Grid Trading Simulation
def run_grid_simulation(df, initial_capital, num_grids, grid_percent_range, trailing_stop_trigger, trailing_stop_offset):
    available_usd = initial_capital
    available_asset = 0
    portfolio_values = []
    trades = []
    
    # Initialize grid
    initial_price = df['close'].iloc[0]
    order_book, lower_bound, upper_bound, spacing = rebuild_grid(
        initial_price, initial_capital, num_grids, grid_percent_range
    )
    
    trailing_active = False
    highest_price = initial_price
    grid_center = initial_price
    
    for i in range(len(df)):
        current_row = df.iloc[i]
        price = current_row['close']
        
        # Dynamic Rebalance if breakout
        if price < lower_bound or price > upper_bound:
            order_book, lower_bound, upper_bound, spacing = rebuild_grid(
                price, initial_capital, num_grids, grid_percent_range
            )
            grid_center = price
            if not trailing_active:
                highest_price = price  # Reset peak
        
        # Trailing Stop Activation
        if not trailing_active and price >= grid_center * trailing_stop_trigger:
            trailing_active = True
            highest_price = price
        
        if trailing_active:
            if price > highest_price:
                highest_price = price
            trail_stop_price = highest_price * (1 - trailing_stop_offset)
            
            if price < trail_stop_price:
                # Trigger Trailing Stop: Sell all assets, exit
                available_usd += available_asset * price
                available_asset = 0
                trades.append({
                    'date': current_row['timestamp'],
                    'type': 'SELL',
                    'price': price,
                    'amount': available_asset,
                    'value': available_asset * price
                })
                break
        
        # Execute grid orders
        for order in order_book:
            if order['side'] == 'buy' and price <= order['price'] and available_usd >= order['price'] * calculate_order_size(order['price'], initial_capital, num_grids):
                qty = calculate_order_size(order['price'], initial_capital, num_grids)
                available_usd -= qty * order['price']
                available_asset += qty
                order['price'] += spacing  # Move order up
                order['side'] = 'sell'
                
                trades.append({
                    'date': current_row['timestamp'],
                    'type': 'BUY',
                    'price': order['price'],
                    'amount': qty,
                    'value': qty * order['price']
                })
                
            elif order['side'] == 'sell' and price >= order['price'] and available_asset >= calculate_order_size(order['price'], initial_capital, num_grids):
                qty = calculate_order_size(order['price'], initial_capital, num_grids)
                available_usd += qty * order['price']
                available_asset -= qty
                order['price'] -= spacing  # Move order down
                order['side'] = 'buy'
                
                trades.append({
                    'date': current_row['timestamp'],
                    'type': 'SELL',
                    'price': order['price'],
                    'amount': qty,
                    'value': qty * order['price']
                })
        
        # Track portfolio value
        total_value = available_usd + available_asset * price
        portfolio_values.append(total_value)
    
    return portfolio_values, trades

# Run the grid trading simulation
grid_portfolio_values, grid_trades = run_grid_simulation(
    df, INITIAL_BALANCE, NUM_GRIDS, GRID_PERCENT_RANGE, 
    TRAILING_STOP_TRIGGER, TRAILING_STOP_OFFSET
)

# Calculate performance metrics
final_portfolio_value = grid_portfolio_values[-1]
total_return = ((final_portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
buy_hold_return = ((price_values[-1] - price_values[0]) / price_values[0]) * 100

# Print a sample of trades
print("\nSample Grid Trading Trades:")
print("===========================")
for trade in grid_trades[:10]:  # Show first 10 trades
    print(f"{trade['date'].strftime('%Y-%m-%d')}: {trade['type']} {trade['amount']:.6f} BTC at ${trade['price']:,.2f} (Value: ${trade['value']:,.2f})")

# Visualize results
plt.figure(figsize=(12, 8))
plt.plot(dates[:len(grid_portfolio_values)], grid_portfolio_values, label='Grid Trading Portfolio Value')
plt.plot(dates, price_values, label='BTC Price', alpha=0.5)

# Add buy and sell markers
buy_dates = [trade['date'] for trade in grid_trades if trade['type'] == 'BUY']
buy_prices = [trade['price'] for trade in grid_trades if trade['type'] == 'BUY']
sell_dates = [trade['date'] for trade in grid_trades if trade['type'] == 'SELL']
sell_prices = [trade['price'] for trade in grid_trades if trade['type'] == 'SELL']

try:
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', alpha=0.7)
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', alpha=0.7)

    plt.title('Bitcoin Grid Trading Simulation')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('btc_grid_simulation_results.png')
    plt.show()

except KeyboardInterrupt:
    print("Simulation interrupted by user")

# Print simulation results
print("\nBitcoin Grid Trading Simulation Results:")
print("=======================================")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Number of Trades: {len(grid_trades)}")