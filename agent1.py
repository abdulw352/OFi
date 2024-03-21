import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from finrl.meta.data_processors import FeatureEngineer
from finrl.meta.env_stock_trading import StockTradingEnv
import yfinance as yf
from datetime import datetime, timedelta

# Load fine-tuned model and tokenizer
model_name = './fine_tuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define stock trading agent
class StockTradingAgent:
    def __init__(self, model, tokenizer, env):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.feature_engineer = FeatureEngineer()

    def get_market_analysis(self, current_data):
        market_analysis = self.model.generate(
            **self.tokenizer(
                f"Based on the following data: {current_data}, please provide a detailed market analysis and stock trading recommendations.",
                return_tensors="pt",
            )
        )[0].text

        return market_analysis

    def get_trade_decision(self, market_analysis):
        trade_decision = self.model.generate(
            **self.tokenizer(
                f"Given the following market analysis: {market_analysis}, should we buy, sell, or hold stocks? Please provide a concise trade decision.",
                return_tensors="pt",
            )
        )[0].text

        return trade_decision

    def execute_trade(self, trade_decision):
        action = {"buy": 1, "sell": 2, "hold": 0}.get(trade_decision.lower(), 0)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

# Define trading environment
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 3, 1)
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB"]
env = StockTradingEnv(tickers, start_date, end_date)

# Initialize agent
agent = StockTradingAgent(model, tokenizer, env)

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

# Print final portfolio value
print(f"Final Portfolio Value: ${info['portfolio_value']:.2f}")
