# stock_price_prediction.py

import os
from src.data_collection import download_data
from src.eda import plot_stock_data
from src.data_preprocessing import preprocess_data
from src.lstm_model import build_lstm_model
from src.evaluate_model import evaluate_model

def main():
    # Step 1: Data Collection
    ticker = "AAPL"  # You can change the stock ticker here (e.g., 'GOOGL', 'MSFT')
    start_date = "2010-01-01"
    end_date = "2023-12-01"
    print(f"Downloading stock data for {ticker} from {start_date} to {end_date}...")
    
    # Download stock data and save it to CSV
    data = download_data(ticker, start_date, end_date)
    data.to_csv(f"data/{ticker}_stock_data.csv")
    print(f"Data saved to data/{ticker}_stock_data.csv.")

    # Step 2: Exploratory Data Analysis (EDA)
    print("Plotting stock data...")
    plot_stock_data(f"data/{ticker}_stock_data.csv")
    
    # Step 3: Data Preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(f"data/{ticker}_stock_data.csv")

    # Step 4: Model Building (LSTM)
    print("Building and training LSTM model...")
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save(f"{ticker}_lstm_model.h5")
    print(f"LSTM model saved as {ticker}_lstm_model.h5.")
    
    # Step 5: Model Evaluation
    print("Evaluating the LSTM model...")
    evaluate_model(f"{ticker}_lstm_model.h5", f"data/{ticker}_stock_data.csv")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    main()
