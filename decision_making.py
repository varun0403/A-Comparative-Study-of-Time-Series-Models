import pandas as pd
import sys
from statsmodels.tsa.stattools import adfuller
import json
from models.ARIMA import arima_forecast
from models.LSTM import lstm_forecast
from models.XGBOOST import xgboost_forecast


def is_stationary(series, alpha=0.05):
    result = adfuller(series)
    p_value = result[1]
    return p_value < alpha


csv_path = "data/HDFCBANK.csv"

try:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    close_series = df['Close']

    if is_stationary(close_series):
        print("Data is stationary. Using ARIMA model...")
        metrics = arima_forecast(df)
        output_file = "results/arima_forecast.png"
    else:
        print("Data is NOT stationary. Using LSTM and XGBoost...")
        metrics = {
            # "LSTM": lstm_forecast(df),
            "XGBoost": xgboost_forecast(df)
        }
        print(metrics)
        output_file = "results/lstm_xgboost_forecast.png"

        #Print for Java to pick up image
        print(f"Forecast Plot Saved As: {output_file}")
        print("--- Error Metrics for XGBoost Forecast ---")
        for model_name, model_metrics in metrics.items():
            print(f"{model_name}:")
            for metric, value in model_metrics.items():
                print(f"  {metric}: {value}")
            print()

    with open("output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

except Exception as e:
    print(f"Error reading CSV or processing data: {e}")
    sys.exit(1)
