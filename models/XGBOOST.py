import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import json


def xgboost_forecast(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Split data into train and test
    # Split data into train and test
    train_data = df[df.index <= pd.to_datetime('2020-06-30')]['Close'].values
    test_data = df[df.index > pd.to_datetime('2020-06-30')]['Close'].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

    # Prepare sequences for XGBoost
    def create_supervised_data(data, time_step=90):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i - time_step:i, 0])  # Features (previous 'time_step' values)
            y.append(data[i, 0])  # Target (next value)
        return np.array(X), np.array(y)

    # Create training data
    X_train, y_train = create_supervised_data(train_scaled)

    # Reshape X_train for XGBoost (it expects a 2D array of features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])

    # Prepare the DMatrix for XGBoost (efficient way to use XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train XGBoost model
    params = {
        'objective': 'reg:squarederror',  # Regression task
        'eval_metric': 'rmse',  # Evaluation metric
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
    }

    model = xgb.train(params, dtrain, num_boost_round=100)

    # Prepare test data (scale and create sequences)
    test_scaled = scaler.transform(test_data.reshape(-1, 1))
    X_test, y_test = create_supervised_data(test_scaled)

    # Reshape X_test for prediction
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Prepare DMatrix for test data
    dtest = xgb.DMatrix(X_test)

    # Make predictions
    predictions = model.predict(dtest)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Evaluate the model
    mae = mean_absolute_error(test_data[-len(predictions):], predictions)
    mse = mean_squared_error(test_data[-len(predictions):], predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data[-len(predictions):] - predictions) / test_data[-len(predictions):])) * 100
    r2 = r2_score(test_data[-len(predictions):], predictions)

    # Print error metrics
    print("\n--- Error Metrics for XGBoost Forecast ---")
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

    with open("../results/xgboost_metrics.json", "w") as f:
        json.dump(metrics, f)

    # Plot actual vs forecasted values
    plt.figure(figsize=(15, 6))
    # plt.plot(df.index[-len(test_data):], test_data, label='Actual Close Price', color='blue')
    plt.plot(df['Close'], color='blue')
    plt.plot(df.index[-len(predictions):], predictions, label='Forecasted Close Price', color='red')
    plt.title('XGBoost Forecast vs Actual Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig("results/xgboost_img.png")
    plt.grid(True)
    plt.show()

    return metrics
