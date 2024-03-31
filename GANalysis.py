import data_ingestion
import gan_module
import evaluation_module
import visualization_module

# Data Ingestion
stock_data, news_data = data_ingestion.fetch_data(stock_ticker, start_date, end_date)
processed_data = data_ingestion.preprocess_data(stock_data, news_data)

# GAN Module
generator = gan_module.Generator(processed_data)
discriminator = gan_module.Discriminator(processed_data)
gan = gan_module.GAN(generator, discriminator)
gan.train(epochs=100)
generated_strategies = gan.generate_strategies()

# Evaluation Module
backtesting_results = evaluation_module.backtest(generated_strategies, processed_data)
ranked_strategies = evaluation_module.rank_strategies(backtesting_results)
top_strategy = ranked_strategies[0]

# Visualization Module
visualization_module.plot_stock_data(processed_data)
visualization_module.plot_news_sentiment(processed_data)
visualization_module.plot_strategy(top_strategy)
visualization_module.plot_performance_metrics(backtesting_results, top_strategy)
