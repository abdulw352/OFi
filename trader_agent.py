# Trader_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import load_model, fine_tune_model
from .data import preprocess_data

class StockTradingAgent:
    def __init__(self, model_name, tokenizer_name, env, data):
        self.model, self.tokenizer = load_model(model_name, tokenizer_name)
        self.env = env
        self.data = preprocess_data(data)

    def fine_tune(self, training_data):
        self.model, self.tokenizer = fine_tune_model(self.model, self.tokenizer, training_data)

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
