import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import yfinance as yf
import tweepy
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv



load_dotenv()




# Set up your Twitter API credentials
consumer_key = os.getenv("TWITTER_API_KEY")
consumer_secret = os.getenv("TWITTER_API_SECRET_KEY")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")


# The Twitter API might be better to follow along with, feel there is more control there!
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)

# api = tweepy.Client(auth)
client = tweepy.Client(bearer_token)


# Search Recent Tweets

# This endpoint/method returns Tweets from the last seven days

# Search for recent tweets containing "Tweepy"
response = client.search_recent_tweets("Tweepy")
print(response.meta)

tweets = response.data
for tweet in tweets:
    print(tweet.id)
    print(tweet.text)

# Search for recent tweets containing "stocks", with a maximum of 100 results
response = client.search_recent_tweets("stocks", max_results=100)
stock_tweets = response.data

# Create an instance of SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each tweet
sentiment_scores = []
for tweet in stock_tweets:
    sentiment_scores.append(sia.polarity_scores(tweet.text))

# Print the sentiment scores
for score in sentiment_scores:
    print(score)


# Define the stock symbol and date range
stock_symbol = "AAPL"
start_date = "2021-01-01"
end_date = "2021-12-31"

# Retrieve the historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Print the retrieved stock data
print(stock_data)

# Define your trading strategy logic
def trading_strategy(data):
    # Calculate the moving average
    data['MA'] = data['Close'].rolling(window=50).mean()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(data)):
        if data['Close'][i] > data['MA'][i] and position == 0:
            position = 1
            buy_price = data['Close'][i]
        elif data['Close'][i] < data['MA'][i] and position == 1:
            position = 0
            sell_price = data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Apply the trading strategy to the historical stock data
trading_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)



# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your sentiment-based trading strategy
def sentiment_based_strategy(sentiment_scores, stock_data):
    # Combine sentiment scores with stock data
    combined_data = stock_data.join(sentiment_scores)

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(combined_data)):
        # Check sentiment score and stock price conditions
        if combined_data['Sentiment'][i] > 0.5 and position == 0:
            position = 1
            buy_price = combined_data['Close'][i]
        elif combined_data['Sentiment'][i] < 0.5 and position == 1:
            position = 0
            sell_price = combined_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
sentiment_scores = [0.6, 0.8, 0.4, 0.2]
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
sentiment_based_strategy(sentiment_scores, stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)


# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Sentiment-Based Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your mean-reversion trading strategy
def mean_reversion_strategy(stock_data):
    # Calculate the mean price
    mean_price = stock_data['Close'].mean()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the price is above or below the mean
        if stock_data['Close'][i] > mean_price and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        elif stock_data['Close'][i] < mean_price and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
mean_reversion_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)



# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Mean-Reversion Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your trend-following trading strategy
def trend_following_strategy(stock_data):
    # Calculate the 50-day and 200-day moving averages
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the 50-day moving average is above or below the 200-day moving average
        if stock_data['MA50'][i] > stock_data['MA200'][i] and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        elif stock_data['MA50'][i] < stock_data['MA200'][i] and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
trend_following_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)



# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Trend-Following Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your breakout trading strategy
def breakout_strategy(stock_data, breakout_window=20, breakout_multiplier=1.5):
    # Calculate the breakout levels
    stock_data['RollingMax'] = stock_data['Close'].rolling(window=breakout_window).max()
    stock_data['BreakoutLevel'] = stock_data['RollingMax'] * breakout_multiplier

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the price breaks above the breakout level
        if stock_data['Close'][i] > stock_data['BreakoutLevel'][i] and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        # Check if the price falls below the breakout level
        elif stock_data['Close'][i] < stock_data['BreakoutLevel'][i] and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
breakout_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)



# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Breakout Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your mean reversion trading strategy
def mean_reversion_strategy(stock_data, mean_window=20, deviation_threshold=1):
    # Calculate the mean and standard deviation
    stock_data['RollingMean'] = stock_data['Close'].rolling(window=mean_window).mean()
    stock_data['RollingStd'] = stock_data['Close'].rolling(window=mean_window).std()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the price deviates below the mean level
        if stock_data['Close'][i] < stock_data['RollingMean'][i] - deviation_threshold * stock_data['RollingStd'][i] and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        # Check if the price reverts above the mean level
        elif stock_data['Close'][i] > stock_data['RollingMean'][i] and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
mean_reversion_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)


# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Mean Reversion Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your trend following trading strategy
def trend_following_strategy(stock_data, ma_short=50, ma_long=200):
    # Calculate the moving averages
    stock_data['MA_short'] = stock_data['Close'].rolling(window=ma_short).mean()
    stock_data['MA_long'] = stock_data['Close'].rolling(window=ma_long).mean()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the short-term moving average crosses above the long-term moving average
        if stock_data['MA_short'][i] > stock_data['MA_long'][i] and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        # Check if the short-term moving average crosses below the long-term moving average
        elif stock_data['MA_short'][i] < stock_data['MA_long'][i] and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
trend_following_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)


# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Trend Following Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your breakout trading strategy
def breakout_strategy(stock_data, lookback_period=20, breakout_multiplier=1.05):
    # Calculate the high and low prices for the lookback period
    stock_data['High_lookback'] = stock_data['High'].rolling(window=lookback_period).max()
    stock_data['Low_lookback'] = stock_data['Low'].rolling(window=lookback_period).min()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the current price breaks above the high of the lookback period
        if stock_data['Close'][i] > stock_data['High_lookback'][i] and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        # Check if the current price breaks below the low of the lookback period
        elif stock_data['Close'][i] < stock_data['Low_lookback'][i] and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
breakout_strategy(stock_data)

# Define a function to calculate performance metrics
def calculate_performance(buy_prices, sell_prices):
    # Calculate the number of trades
    num_trades = len(buy_prices)

    # Calculate the total profit
    total_profit = sum(sell_prices[i] - buy_prices[i] for i in range(num_trades))

    # Calculate the average profit per trade
    avg_profit_per_trade = total_profit / num_trades

    # Calculate the winning trades
    winning_trades = sum(sell_prices[i] > buy_prices[i] for i in range(num_trades))

    # Calculate the winning percentage
    winning_percentage = (winning_trades / num_trades) * 100

    # Print the performance metrics
    print(f"Number of trades: {num_trades}")
    print(f"Total profit: {total_profit}")
    print(f"Average profit per trade: {avg_profit_per_trade}")
    print(f"Winning trades: {winning_trades}")
    print(f"Winning percentage: {winning_percentage}%")

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
calculate_performance(buy_prices, sell_prices)


# Define a function to visualize the performance
def visualize_performance(buy_prices, sell_prices):
    # Calculate the cumulative profit
    cumulative_profit = [0]
    for i in range(1, len(buy_prices)):
        profit = sell_prices[i] - buy_prices[i]
        cumulative_profit.append(cumulative_profit[i-1] + profit)

    # Create the line plot
    plt.plot(range(len(buy_prices)), cumulative_profit)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Profit')
    plt.title('Performance of Breakout Trading Strategy')
    plt.show()

# Example usage
buy_prices = [100, 110, 120]
sell_prices = [105, 115, 130]
visualize_performance(buy_prices, sell_prices)

# Define your mean reversion trading strategy
def mean_reversion_strategy(stock_data, lookback_period=20, deviation_threshold=1):
    # Calculate the mean and standard deviation for the lookback period
    stock_data['Mean'] = stock_data['Close'].rolling(window=lookback_period).mean()
    stock_data['Std'] = stock_data['Close'].rolling(window=lookback_period).std()

    # Initialize variables
    position = 0  # 0: out of the market, 1: long position
    buy_price = 0

    # Apply the trading strategy
    for i in range(len(stock_data)):
        # Check if the current price is below the mean minus the deviation threshold
        if stock_data['Close'][i] < stock_data['Mean'][i] - (deviation_threshold * stock_data['Std'][i]) and position == 0:
            position = 1
            buy_price = stock_data['Close'][i]
        # Check if the current price is above the mean plus the deviation threshold
        elif stock_data['Close'][i] > stock_data['Mean'][i] + (deviation_threshold * stock_data['Std'][i]) and position == 1:
            position = 0
            sell_price = stock_data['Close'][i]
            profit = sell_price - buy_price
            print(f"Buy at {buy_price}, Sell at {sell_price}, Profit: {profit}")

# Example usage
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
mean_reversion_strategy(stock_data)