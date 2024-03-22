import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Define a function to scrape news articles
def scrape_news_articles(query, max_articles=10):
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    data = []
    for result in soup.select(".g"):
        title = result.select_one(".LC20lb").text
        link = result.select_one(".yuRUbf a")["href"]
        snippet = result.select_one(".VwiC3b").text

        if "news.google.com" in link:
            article_response = requests.get(link)
            article_soup = BeautifulSoup(article_response.text, "html.parser")
            article_text = " ".join([p.text for p in article_soup.select(".BNeawe.AP7Wnd")])
            data.append({"title": title, "link": link, "text": article_text})

        if len(data) >= max_articles:
            break

    return data

# Define a function to process the scraped data
def process_data(data):
    input_texts = []
    target_texts = []
    for item in data:
        input_text = f"Title: {item['title']}\n{item['text']}"
        target_text = f"Recommended action: <action>"
        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts

# Load pre-trained model and tokenizer
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Scrape news articles
query = "Apple stock news"
data = scrape_news_articles(query)

# Process the scraped data
input_texts, target_texts = process_data(data)

# Tokenize data
tokenized_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
tokenized_targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")
