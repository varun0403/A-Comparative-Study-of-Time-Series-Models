import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def arima_forecast(df):
    df = df.drop(columns=['Turnover', 'Trades', 'Deliverable Volume', '%Deliverble'], axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.drop('Last', axis=1)

    # Split the data
    train_size = int(len(df) * 0.8)
    train = df['Close'][:train_size]
    test = df['Close'][train_size:]

    # Train ARIMA model
    p, d, q = 1, 1, 1
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # Forecast on test set
    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index  # Align index for comparison

    # Compute error metrics
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    r2 = r2_score(test, forecast)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2_Score": r2
    }

    # Forecast for the next 2 years (730 days)
    future_forecast = model_fit.get_forecast(steps=730)
    forecast_mean = future_forecast.predicted_mean

    # Reconstruct Close from differenced forecast
    last_real_close = df['Close'].iloc[-1]
    forecast_cumsum = forecast_mean.cumsum()
    forecast_close = last_real_close + forecast_cumsum

    # Build future date range
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=730, freq='D')
    forecast_series = pd.Series(forecast_close.values, index=future_dates)

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(df['Close'], label='Actual Close', color='blue')
    plt.plot(forecast_series, label='Forecast Close', color='red')
    plt.title('Forecast of Close Price for Next 2 Years')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/arima_img.png")
    plt.show()

    return metrics
