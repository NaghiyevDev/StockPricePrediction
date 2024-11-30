# data_collection.py

import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date):
    """
    Downloads stock data from Yahoo Finance.
    
    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: DataFrame containing the stock data
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Example usage: download data for Apple (AAPL)
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-12-01"
    df = download_data(ticker, start_date, end_date)
    
    # Save to CSV
    df.to_csv(f"data/{ticker}_stock_data.csv")
    print(f"Data saved for {ticker} from {start_date} to {end_date}.")
