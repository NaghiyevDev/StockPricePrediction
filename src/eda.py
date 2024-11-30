# eda.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_stock_data(file_path):
    """
    Plots the closing stock price from the CSV data file.
    
    :param file_path: Path to the CSV file containing stock data
    """
    df = pd.read_csv(file_path)
    
    # Check if 'Date' is present as a column; if so, convert to datetime and set it as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Ensure the 'Close' column is numeric (handle errors gracefully if not)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Drop rows with NaN values in the 'Close' column (if any)
    df = df.dropna(subset=['Close'])
    
    # Plot the 'Close' price
    df['Close'].plot(title="Stock Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()

# Example usage: plot data for Apple (AAPL)
if __name__ == "__main__":
    file_path = "data/AAPL_stock_data.csv"
    plot_stock_data(file_path)
