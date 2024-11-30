# lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model for stock price prediction.
    
    :param input_shape: Shape of the input data (time steps, features)
    :return: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))  # Output layer for predicting the stock price
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Example usage: build and train LSTM model for Apple (AAPL)
if __name__ == "__main__":
    from src.data_preprocessing import preprocess_data
    file_path = "data/AAPL_stock_data.csv"
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)

    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save("lstm_model.h5")
    print("LSTM model trained and saved.")
