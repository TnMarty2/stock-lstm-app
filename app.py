import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.layers import Dropout

# -------------------------------
# Function: Load and preprocess
# -------------------------------
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)[['Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def preprocess_data(data, lookback=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])  # only predict 'Close'
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_test_split(X, y, ratio=0.8):
    train_size = int(len(X) * ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# -------------------------------
# Function: Build models
# -------------------------------
def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == "Vanilla LSTM":
        model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(50, return_sequences=False), input_shape=input_shape))
    elif model_type == "Stacked LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------------
# Function: Evaluate models
# -------------------------------
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("📈 Multivariate Stock Prediction with LSTM")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
model_type = st.selectbox("Select Model Type", ["Vanilla LSTM", "Bidirectional LSTM", "Stacked LSTM"])

if st.button("Train and Predict"):
    df = load_data(ticker, start_date, end_date)
    st.write("📊 Raw Data", df.tail())
    
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = build_model(model_type, (X.shape[1], X.shape[2]))
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))), axis=1))[:, 0]
    y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, np.zeros_like(y_pred)), axis=1))[:, 0]

    rmse, mae, mape = evaluate_model(y_test_inv, y_pred_inv)
    
    st.write(f"📉 **Evaluation Metrics**\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- MAPE: {mape:.2f}%")

    st.line_chart(pd.DataFrame({
        "Actual": y_test_inv,
        "Predicted": y_pred_inv
    }))
