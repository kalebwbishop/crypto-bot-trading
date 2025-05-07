import simulator
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide
from get_data import get_data
from datetime import datetime, timedelta

def buy(usd_balance, btc_balance, price, fee_percent=0.005):
    client = TradingClient()
    account = client.get_account()
    btc_to_buy = account.cash / price
    client.order_market_buy(symbol='BTCUSD', qty=btc_to_buy)
    return 0, btc_balance + btc_to_buy

def sell(usd_balance, btc_balance, price, fee_percent=0.005):
    client = TradingClient()
    account = client.get_account()
    usd_from_sale = btc_balance * price
    fee_amount = usd_from_sale * fee_percent
    client.order_market_sell(symbol='BTCUSD', qty=btc_balance)
    return usd_balance + (usd_from_sale - fee_amount), 0


if __name__ == "__main__":
    current_time = datetime.now()
    start_date = current_time - timedelta(days=365)
    end_date = current_time
    get_data(start_date, end_date)

    trades, last_sell_value = simulator.simulate(100000, 0, start_date, end_date, 0.98, 1.02, 0.99, 0.005)

    print(last_sell_value)
