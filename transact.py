import http.client
import json
import uuid
from jwt_generator import build_jwt

def get_account_balance(currency):
    conn = http.client.HTTPSConnection("api.coinbase.com")
    headers = {
        "Authorization": "Bearer " + build_jwt("GET", "api/v3/brokerage/accounts"),
    }
    conn.request("GET", "/api/v3/brokerage/accounts", headers=headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    
    # Find USD account
    for account in data.get('accounts', []):
        if account.get('currency') == currency:
            return float(account.get('available_balance', {}).get('value', '0'))
    return 0.0


def buy_btc():
    print("Buying BTC")
    try:
        # Get available USD balance
        usd_balance = get_account_balance('USD')

        usd_balance = usd_balance * 0.99


        if usd_balance <= 0:
            print("No USD balance available to buy BTC")
            return
        
        # Format the USD amount to have appropriate precision (2 decimal places)
        formatted_usd_balance = "{:.2f}".format(usd_balance)
        
        conn = http.client.HTTPSConnection("api.coinbase.com")
        client_order_id = str(uuid.uuid4())
        
        payload = json.dumps(
            {
                "product_id": "BTC-USD",
                "client_order_id": client_order_id,
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {
                        "quote_size": formatted_usd_balance,  # Changed from base_size to quote_size for USD amount
                    }
                },
            }
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + build_jwt("POST", "api/v3/brokerage/orders"),
        }
        conn.request("POST", "/api/v3/brokerage/orders", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")


def sell_btc():
    print("Selling BTC")
    try:
        # Get available BTC balance
        btc_balance = get_account_balance('BTC')

        btc_balance = btc_balance * 0.99

        if btc_balance <= 0:
            print("No BTC balance available to sell")
            return
        
        conn = http.client.HTTPSConnection("api.coinbase.com")
        client_order_id = str(uuid.uuid4())
        
        # Calculate the amount of BTC we can sell    
        payload = json.dumps(
            {
                "product_id": "BTC-USD",
                "client_order_id": client_order_id,
                "side": "SELL",
                "order_configuration": {
                    "market_market_ioc": {
                        "base_size": str(btc_balance),
                    }
                },
            }
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + build_jwt("POST", "api/v3/brokerage/orders"),
        }
        conn.request("POST", "/api/v3/brokerage/orders", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(data.decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # sell_btc()
    buy_btc()