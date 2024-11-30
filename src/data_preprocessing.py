# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    """
    Preprocesses the stock data by scaling and splitting it into training and test sets.
    
    :param file_path: Path to the CSV file containing stock data
    :return: Scaled training and test data
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Use the 'Close' column for prediction
    data = df[['Close']]

    # Scale the data to a range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for the model
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    window_size = 60  # Use 60 previous days for prediction
    X, y = create_sequences(scaled_data, window_size)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape for LSTM input (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test, scaler

# Example usage: preprocess data for Apple (AAPL)
if __name__ == "__main__":
    file_path = "data/AAPL_stock_data.csv"
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_path)
    print("Data preprocessed and ready for training.")
