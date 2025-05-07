import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm
from get_data import get_data

def buy(usd_balance, btc_balance, price, fee_percent=0.005, real=False):
    # Calculate fee amount
    fee_amount = usd_balance * fee_percent
    # Calculate how much BTC we can buy with our USD balance after fees
    btc_to_buy = (usd_balance - fee_amount) / price
    return 0, btc_balance + btc_to_buy

def sell(usd_balance, btc_balance, price, fee_percent=0.005, real=False):
    # Calculate how much USD we get from selling our BTC
    usd_from_sale = btc_balance * price
    # Calculate fee amount
    fee_amount = usd_from_sale * fee_percent
    # Return USD balance after fees
    return usd_balance + (usd_from_sale - fee_amount), 0

def simulate(usd_balance, btc_balance, start_date: datetime, end_date: datetime, buy_percent=0.98, sell_percent=1.02, floor=0.99, fee_percent=0.005, real=False, verbose=False):
    # Load and preprocess data more efficiently
    df = pd.read_csv('BTC_USD.csv', parse_dates=['datetime'])
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()
    
    # Pre-calculate moving averages and filter out NaN values
    df['buy_price'] = df['moving_average_60'] * buy_percent
    df['sell_price'] = df['moving_average_60'] * sell_percent
    
    # Remove rows with NaN values
    df = df.dropna(subset=['moving_average_60'])
    
    # Initialize arrays for tracking
    n = len(df)
    portfolio_values = np.zeros(n)
    btc_balances = np.zeros(n)
    usd_balances = np.zeros(n)
    total_fees = 0.0  # Track total fees paid
    
    # Set initial values
    usd_balances[0] = usd_balance
    btc_balances[0] = btc_balance
    portfolio_values[0] = usd_balance + (btc_balance * df.iloc[0]['close'])
    
    trades = []
    last_purchase_price = float('inf')
    
    # Vectorized price arrays
    prices = df['close'].values
    buy_prices = df['buy_price'].values
    sell_prices = df['sell_price'].values
    dates = df['datetime'].values
    
    # Main simulation loop
    for i in tqdm(range(1, n)):
        current_price = prices[i]
        current_date = dates[i]
        
        # Buy decision
        if current_price <= buy_prices[i] and usd_balances[i-1] > 0:
            usd_balances[i], btc_balances[i] = buy(usd_balances[i-1], btc_balances[i-1], current_price, fee_percent, real)
            last_purchase_price = current_price * floor
            fee_amount = usd_balances[i-1] * fee_percent
            total_fees += fee_amount
            trades.append({
                'datetime': current_date,
                'price': current_price,
                'usd_balance': usd_balances[i],
                'btc_balance': btc_balances[i],
                'trade': 'BUY',
                'ma': df.iloc[i]['moving_average_60'],
                'fee': fee_amount
            })
        # Sell decision
        elif current_price >= sell_prices[i] and current_price > last_purchase_price and btc_balances[i-1] > 0:
            usd_balances[i], btc_balances[i] = sell(usd_balances[i-1], btc_balances[i-1], current_price, fee_percent, real)
            fee_amount = btc_balances[i-1] * current_price * fee_percent
            total_fees += fee_amount
            trades.append({
                'datetime': current_date,
                'price': current_price,
                'usd_balance': usd_balances[i],
                'btc_balance': btc_balances[i],
                'trade': 'SELL',
                'ma': df.iloc[i]['moving_average_60'],
                'fee': fee_amount
            })
        else:
            usd_balances[i] = usd_balances[i-1]
            btc_balances[i] = btc_balances[i-1]
        
        # Update portfolio value
        portfolio_values[i] = usd_balances[i] + (btc_balances[i] * current_price)
    
    if verbose:
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.style.use('classic')
        
        # Set white background
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Plot the price line
        plt.plot(df['datetime'], df['close'], label='BTC Price', color='blue')
        
        # Plot moving averages
        plt.plot(df['datetime'], df['moving_average_60'], 
                label='60-minute MA', color='purple', linestyle='--')
        
        # Plot buy and sell points
        buy_points = [(t['datetime'], t['price']) for t in trades if t['trade'] == 'BUY']
        sell_points = [(t['datetime'], t['price']) for t in trades if t['trade'] == 'SELL']
        
        if buy_points:
            dates, prices = zip(*buy_points)
            plt.scatter(dates, prices, marker='^', color='green', s=100, label='Buy')
        
        if sell_points:
            dates, prices = zip(*sell_points)
            plt.scatter(dates, prices, marker='v', color='red', s=100, label='Sell')
        
        plt.title('BTC Price with Buy/Sell Points and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        
        # Improve grid appearance
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
    
    # Calculate final profit
    initial_value = usd_balance + (btc_balance * df.iloc[0]['close'])
    
    # Find last sell trade
    last_sell = next((trade for trade in reversed(trades) if trade['trade'] == 'SELL'), None)
    last_sell_value = last_sell['usd_balance'] if last_sell else portfolio_values[-1]
    last_sell_profit = last_sell_value - initial_value
    
    # Calculate final portfolio value profit
    final_portfolio_value = portfolio_values[-1]
    final_portfolio_profit = final_portfolio_value - initial_value
    
    # Calculate ROIs
    roi_with_trades = (final_portfolio_profit / initial_value) * 100
    roi_without_trades = ((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
    
    if verbose:
        print(f"Number of trades: {len(trades)}")
        print(f"Total fees paid: ${total_fees:.2f}")
        print(f"Profit based on last sell: ${last_sell_profit:.2f}")
        print(f"Profit based on final portfolio: ${final_portfolio_profit:.2f}")
        print(f"ROI with trades: {roi_with_trades:.2f}%")
        print(f"ROI without trades (just holding): {roi_without_trades:.2f}%")
        print(f"Final portfolio value: ${final_portfolio_value:.2f}")
        print(f"Final BTC balance: {btc_balances[-1]:.8f}")
        print(f"Final USD balance: ${usd_balances[-1]:.2f}")
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        plt.close()
    
    return trades, last_sell_value

if __name__ == '__main__':
    usd_balance = 100000
    btc_balance = 0
    initial_value = usd_balance

    current_time = datetime(2022, 1, 1)
    start_date = current_time - timedelta(days=2 * 365)
    end_date = current_time
    get_data(start_date, end_date)
    
    # Using 0.5% trading fee (0.005)
    best_buy_percent = 0
    best_sell_percent = 0
    best_floor = 0
    best_last_sell_value = 0
    for buy_percent in np.arange(0.98, 1.02, 0.01):
        for sell_percent in np.arange(0.98, 1.02, 0.01):
            for floor in np.arange(0.98, 1.02, 0.01):
                trades, last_sell_value = simulate(usd_balance, btc_balance, start_date, end_date, buy_percent, sell_percent, floor, 0.005)
                if last_sell_value > best_last_sell_value:
                    best_buy_percent = buy_percent
                    best_sell_percent = sell_percent
                    best_floor = floor
                    best_last_sell_value = last_sell_value
                print(f"Buy percent: {buy_percent}, Sell percent: {sell_percent}, Floor: {floor}, Last sell value: {last_sell_value}")
    print(f"Best buy percent: {best_buy_percent}, Best sell percent: {best_sell_percent}, Best floor: {best_floor}, Best last sell value: {best_last_sell_value}")


    # current_time = datetime.now()
    # start_date = current_time - timedelta(days=365)
    # end_date = current_time
    # get_data(start_date, end_date)

    trades, last_sell_value = simulate(usd_balance, btc_balance, start_date, end_date, 0.99, 1.01, 0.99, 0.005, verbose=True)
