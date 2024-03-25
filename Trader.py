from package_name import StockTradingAgent
from package_name.environment import StockTradingEnv
import yfinance as yf
from datetime import datetime

# Define trading environment
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 3, 1)
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB"]
env = StockTradingEnv(tickers, start_date, end_date)

# Fetch stock data
stock_data = yf.download(tickers, start_date, end_date)

# Initialize agent
agent = StockTradingAgent(model_name="google/mt5-small", tokenizer_name="google/mt5-small", env=env, data=stock_data)

# Fine-tune the model (optional)
agent.fine_tune(training_data)

# Trading loop
state = env.reset()
while True:
    current_data = env.get_current_data()
    market_analysis = agent.get_market_analysis(current_data)
    trade_decision = agent.get_trade_decision(market_analysis)

    if trade_decision.lower() in ["buy", "sell", "hold"]:
        state, reward, done, info = agent.execute_trade(trade_decision)
        print(f"Trade Decision: {trade_decision}")
        print(f"Reward: {reward}")
        print(f"Portfolio Value: {info['portfolio_value']:.2f}")
    else:
        print("Invalid trade decision. Skipping this step.")

    if done:
        break
