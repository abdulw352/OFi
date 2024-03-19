import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load and preprocess financial data
stock_data = pd.read_csv('stock_data.csv') # with training data
news_data = pd.read_csv('news_data.csv') # training new data

# Combine and preprocess data
data = stock_data.merge(news_data, on='date', how='left')
data = data.dropna()
data['text'] = data['company_name'] + ' ' + data['news_headline'] + ' ' + data['news_body']

# Load pre-trained model and tokenizer
model_name = 'tiiuae/mistral-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize data
tokenized_data = tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./output',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end=True
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./fine_tuned_model')