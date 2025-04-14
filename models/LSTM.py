import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


def lstm_forecast(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Split data into train (up to 2019) and test (2020 onwards)
    train_data = df[df.index <= '2019-12-31']['Close'].values
    test_data = df[df.index > '2019-12-31']['Close'].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

    # Prepare sequences for training
    def create_sequences(data, time_step=90):
        x, y = [], []
        for i in range(time_step, len(data)):
            x.append(data[i - time_step:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    # Create training sequences
    x_train, y_train = create_sequences(train_scaled)

    # Reshape x_train for LSTM input
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for prediction
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=200, batch_size=32, callbacks=[early_stop])

    # Last date from your dataset
    last_date = df.index.max()

    # Generate future date range for next 8 months (≈240 days)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=240, freq='D')

    # Prepare input sequence: last 60 days from the available data
    inputs = df['Close'].values
    inputs_scaled = scaler.transform(inputs.reshape(-1, 1))
    x_input = inputs_scaled[-90:].reshape(1, -1, 1)

    # Forecast for next 240 days
    forecast_values = []
    for _ in range(240):
        forecast = model.predict(x_input, verbose=0)
        forecast_values.append(forecast[0, 0])
        x_input = np.append(x_input[:, 1:, :], forecast.reshape(1, 1, 1), axis=1)

    # Inverse transform to get real prices
    forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))

    # Plot actual and forecasted prices
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Close'], label='Actual Close Price', color='blue')
    plt.plot(future_dates, forecast_values, label='Forecasted Close (Next 8 Months)', color='red')
    plt.title('Forecast for Next 8 Months After Last Known Date')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/lstm_img.png")
    plt.show()

    actual_values = test_data[:len(forecast_values)]

    # Slice predicted values to match length (just in case)
    predicted_values = forecast_values[:len(actual_values)]

    # Compute error metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    r2 = r2_score(actual_values, predicted_values)

    # Print them nicely
    print("\n--- Error Metrics for 8-Month Forecast ---")
    print(f"MAE  (Mean Absolute Error)      : {mae:.2f}")
    print(f"MSE  (Mean Squared Error)       : {mse:.2f}")
    print(f"RMSE (Root Mean Squared Error)  : {rmse:.2f}")
    print(f"MAPE (Mean Absolute % Error)    : {mape:.2f}%")
    print(f"R² Score (Coefficient of Determination): {r2:.4f}")

    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2_Score": round(r2, 4)
    }

    with open("../results/lstm_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics
