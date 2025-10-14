# filename: stock_prices_plot.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(tickers):
    # Get current date and start of this year for YTD range
    today = datetime.now()
    start_date = datetime(today.year, 1, 1)
    
    # Fetch stock data from Yahoo Finance API
    stock_data = yf.download(tickers=tickers, 
                              start=start_date.strftime('%Y-%m-%d'), 
                              end=today.strftime('%Y-%m-%d'))
    
    return stock_data['Close']

# Define tickers and fetch data
tickers = ['NVDA', 'TSLA']
stock_prices = get_stock_data(tickers)

# Plot the data using matplotlib
plt.figure(figsize=(14,7))
for ticker in tickers:
    plt.plot(stock_prices[ticker], label=ticker)
plt.title('Year-to-Date Stock Prices for NVDA and TSLA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()