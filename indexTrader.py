import pandas as pd
import yfinance as yf
from datetime import datetime

# Define the sectors of interest
sectors = ['Technology', 'Energy', 'Healthcare', 'Financials', 'Consumer Discretionary']

# Load the S&P 500 index components and their weights
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500.columns = ['Ticker', 'Company', 'SEC Filings', 'GICS Sector', 'GICS Sub-Industry', 'Headquarters Location', 'Date First Added', 'CIK', 'Founded']

# Filter the companies by sector
sector_companies = {}
for sector in sectors:
    sector_companies[sector] = sp500[sp500['GICS Sector'] == sector].copy()

# Select the top 5 companies by weight for each sector
top_companies = {}
for sector, companies in sector_companies.items():
    companies = companies.sort_values(by='Weight', ascending=False)
    top_companies[sector] = companies.head(5)['Ticker'].tolist()

# Fetch historical stock data for the selected companies
start_date = datetime(2022, 1, 1)
end_date = datetime.now()
all_tickers = []
for sector, tickers in top_companies.items():
    all_tickers.extend(tickers)

stock_data = yf.download(all_tickers, start=start_date, end=end_date)

# Define the trading strategy
portfolio_value = 100000  # Initial portfolio value
portfolio = {}

for sector, tickers in top_companies.items():
    sector_allocation = portfolio_value / len(sectors)
    company_allocation = sector_allocation / len(tickers)
    for ticker in tickers:
        shares = company_allocation // stock_data[ticker]['Close'].iloc[-1]
        portfolio[ticker] = {
            'shares': shares,
            'allocation': shares * stock_data[ticker]['Close'].iloc[-1]
        }
        print(f"Invested ${portfolio[ticker]['allocation']:.2f} in {ticker}")

print(f"Total portfolio value: ${sum(holding['allocation'] for holding in portfolio.values()):.2f}")
