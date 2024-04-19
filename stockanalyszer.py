import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yfinance as yf
from datetime import datetime, timedelta

# Load the Mistral local model and tokenizer
model_path = "path/to/mistral-local-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Define a function to scrape news articles
def scrape_news_articles(company_name, max_articles=5):
    search_query = f"{company_name} stock news"
    search_url = f"https://www.google.com/search?q={search_query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    news_articles = []
    for result in soup.select(".g"):
        title = result.select_one(".LC20lb").text
        link = result.select_one(".yuRUbf a")["href"]
        snippet = result.select_one(".VwiC3b").text

        if "news.google.com" in link:
            article_response = requests.get(link)
            article_soup = BeautifulSoup(article_response.text, "html.parser")
            article_text = " ".join([p.text for p in article_soup.select(".BNeawe.AP7Wnd")])
            news_articles.append({"title": title, "text": article_text})

        if len(news_articles) >= max_articles:
            break

    return news_articles

# Define a function to generate stock analysis and trade recommendation
def generate_stock_analysis(company_name, news_articles):
    stock_info = yf.Ticker(company_name).info
    company_summary = stock_info["longBusinessSummary"]

    input_text = f"Company: {company_name}\n\nBusiness Summary: {company_summary}\n\nRecent News Articles:\n\n"
    for article in news_articles:
        input_text += f"Title: {article['title']}\n{article['text']}\n\n"

    input_text += "Please provide a detailed analysis of the company's stock, considering the business summary and recent news. Also, recommend whether to buy, sell, or hold the stock based on your analysis."

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=1024, num_beams=5, early_stopping=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

# Example usage
company_name = "AAPL"
news_articles = scrape_news_articles(company_name)
stock_analysis = generate_stock_analysis(company_name, news_articles)
print(stock_analysis)
