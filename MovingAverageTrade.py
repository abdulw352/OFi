import yfinance as yf
import pandas as pd
import time

# Load trading account credentials
api_key = 'YOUR_API_KEY'
secret_key = 'YOUR_SECRET_KEY'

# Initialize the trading environment
trading_env = TradingEnvironment(api_key, secret_key)

while True:
    # Fetch live stock data
    ticker = 'AAPL'
    data = yf.download(ticker, period='1d', interval='1m')

    # Generate signals
    signals = moving_average_crossover(data, short_window=20, long_window=50)

    # Check for new signals
    if signals['signal'].iloc[-1] == 1.0 and signals['positions'].iloc[-1] == 1.0:
        # Buy signal
        trading_env.buy_stock(ticker, 100)  # Buy 100 shares
        print(f"Bought {ticker}")

    elif signals['signal'].iloc[-1] == 0.0 and signals['positions'].iloc[-1] == -1.0:
        # Sell signal
        trading_env.sell_stock(ticker, 100)  # Sell 100 shares
        print(f"Sold {ticker}")

    # Wait for the next trading interval
    time.sleep(60)  # Wait for 1 minute
