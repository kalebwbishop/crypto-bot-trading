import requests
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import random
import numpy as np

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

# Simulation parameters
initial_balance = 10000  # Starting with $10,000
btc_balance = 0
usd_balance = initial_balance
trades = []
portfolio_values = []

line = 60000
sell_price = line * 1.025
buy_price = line * 0.975


for i in range(1, len(price_values)):
    current_price = price_values[i]
    current_date = dates[i]
    
    # Buy decision - buy when price is at or below buy_price
    if current_price <= buy_price and usd_balance > 0:
        # Calculate BTC amount to buy
        btc_to_buy = usd_balance / current_price
        
        # Execute trade
        usd_balance -= usd_balance
        btc_balance += btc_to_buy
        
        trades.append({
            'date': current_date,
            'type': 'BUY',
            'price': current_price,
            'amount': btc_to_buy,
            'value': usd_balance
        })

    # Sell decision - sell when price is at or above sell_price
    elif current_price >= sell_price and btc_balance > 0:
        # Calculate USD value
        usd_value = btc_balance * current_price
        
        # Execute trade
        usd_balance += usd_value
        btc_balance -= btc_balance
        
        trades.append({
            'date': current_date,
            'type': 'SELL',
            'price': current_price,
            'amount': btc_balance,
            'value': usd_value
        })
    
    # Calculate total portfolio value (USD + BTC value)
    portfolio_value = usd_balance + (btc_balance * current_price)
    portfolio_values.append(portfolio_value)

# Calculate performance metrics
final_portfolio_value = portfolio_values[-1]
total_return = ((final_portfolio_value - initial_balance) / initial_balance) * 100
buy_hold_return = ((price_values[-1] - price_values[0]) / price_values[0]) * 100

# Print a sample of trades
print("\nSample Trades:")
print("=============")
for trade in trades[:10]:  # Show first 10 trades
    print(f"{trade['date'].strftime('%Y-%m-%d')}: {trade['type']} {trade['amount']:.6f} BTC at ${trade['price']:,.2f} (Value: ${trade['value']:,.2f})")

# Visualize results
plt.figure(figsize=(12, 8))
plt.plot(dates[1:], portfolio_values, label='Portfolio Value')
plt.plot(dates, price_values, label='BTC Price', alpha=0.5)

# Add buy and sell markers
buy_dates = [trade['date'] for trade in trades if trade['type'] == 'BUY']
buy_prices = [trade['price'] for trade in trades if trade['type'] == 'BUY']
sell_dates = [trade['date'] for trade in trades if trade['type'] == 'SELL']
sell_prices = [trade['price'] for trade in trades if trade['type'] == 'SELL']

try:

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
print(f"Initial Balance: ${initial_balance:,.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Number of Trades: {len(trades)}")