# evaluate_model.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.data_preprocessing import preprocess_data

def evaluate_model(model_path, file_path):
    """
    Evaluates the trained model by comparing predictions with actual stock prices.
    
    :param model_path: Path to the saved model
    :param file_path: Path to the CSV file containing stock data
    """
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Invert scaling
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred)

    # Plot the results
    plt.plot(y_test, label='True Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Example usage: evaluate the LSTM model for Apple (AAPL)
if __name__ == "__main__":
    model_path = "lstm_model.h5"
    file_path = "data/AAPL_stock_data.csv"
    evaluate_model(model_path, file_path)
