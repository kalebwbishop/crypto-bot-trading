import requests
import csv
import time
import os
import logging
import pandas as pd
from datetime import datetime
from jwt_generator import build_jwt
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_data(symbol, start_epoch, granularity="ONE_MINUTE", rate_limit_delay=0.5):
    """
    Fetch historical candle data for a given symbol.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC-USD')
        start_epoch (int): Start time in epoch seconds
        granularity (str): Candle timeframe (default: 'ONE_MINUTE')
        rate_limit_delay (float): Delay between API calls in seconds (default: 0.5)
    
    Returns:
        tuple: (bool, pd.DataFrame) - Success status and DataFrame containing the data
    """
    data_file = f"{symbol.replace('-', '_').replace('^', '')}.csv"
    
    try:
        # Initialize or read existing file
        existing_data = {}
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            with open(data_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) > 0:
                        timestamp = int(row[0])
                        existing_data[timestamp] = row
                if existing_data:
                    # Find the most recent timestamp
                    start_epoch = max(existing_data.keys())
                    logger.info(f"Resuming from timestamp: {datetime.fromtimestamp(start_epoch)}")
        else:
            # Create new file with header
            with open(data_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["start", "open", "high", "low", "close", "volume"])

        end_epoch = int(time.time()) - 60
        jwt = build_jwt("GET", f"api/v3/brokerage/products/{symbol}/candles")
        
        while start_epoch < end_epoch:
            url = f"https://api.coinbase.com/api/v3/brokerage/products/{symbol}/candles"
            params = {
                "start": start_epoch,
                "granularity": granularity
            }
            
            try:
                response = session.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {jwt}"},
                    timeout=10
                )

                if response.status_code == 401:
                    # If unauthorized, rebuild JWT and retry
                    jwt = build_jwt("GET", f"api/v3/brokerage/products/{symbol}/candles")
                    continue
                
                response.raise_for_status()
                
                data = response.json().get("candles")
                if not data:
                    logger.info(f"No more data available for {symbol}")
                    break
                
                data.reverse()
                
                # Filter out duplicates and prepare new data
                new_data = []
                for d in data:
                    timestamp = int(d['start'])
                    if timestamp not in existing_data:
                        new_data.append([timestamp, d['open'], d['high'], d['low'], d['close'], d['volume']])
                        existing_data[timestamp] = [timestamp, d['open'], d['high'], d['low'], d['close'], d['volume']]
                
                # Only write if we have new data
                if new_data:
                    with open(data_file, "a") as f:
                        writer = csv.writer(f)
                        writer.writerows(new_data)
                    logger.info(f"Added {len(new_data)} new data points for {symbol}")
                
                new_start_epoch = int(data[-1]['start'])
                if new_start_epoch <= start_epoch:
                    logger.info(f"No new data available for {symbol}")
                    break
                
                start_epoch = new_start_epoch
                end_epoch = int(time.time()) - 60
                
                logger.info(f"Processed data for {symbol} up to {datetime.fromtimestamp(start_epoch)}")
                time.sleep(rate_limit_delay)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                return False, None
            except (KeyError, ValueError, IndexError) as e:
                logger.error(f"Error processing data for {symbol}: {str(e)}")
                return False, None
            
        # Read the CSV file into a pandas DataFrame
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            df = pd.read_csv(data_file)
            # Convert timestamp to datetime
            df['start'] = pd.to_datetime(df['start'], unit='s')
            return True, df
        else:
            return True, pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume"])
        
    except Exception as e:
        logger.error(f"Unexpected error for {symbol}: {str(e)}")
        return False, None

if __name__ == "__main__":
    one_year_ago = int(1682195523) - 3600 * 24 * 365
    success, df = get_data("BTC-USD", start_epoch=one_year_ago)
    if success:
        logger.info("Data collection completed successfully")
        print(f"Collected {len(df)} data points")
    else:
        logger.error("Data collection failed")
        