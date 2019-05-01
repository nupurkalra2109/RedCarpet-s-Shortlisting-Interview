import sys
import nsepy as ns
import pandas as pd
import numpy as np
import datetime
from datetime import date
from sklearn.linear_model import LinearRegression

# GET STOCK DATA
def get_stock_data(stock, start_date, end_date):
    return ns.get_history(symbol=stock, start=start_date, end=end_date)

# GET FEATURES FROM STOCK DATA
def get_stock_features(stock_data):
    stock_day = [date.day for date in stock_data.index.values]
    stock_month = [date.month for date in stock_data.index.values]
    stock_prev_close = stock_data['Prev Close'].values.reshape(1, -1)
    stock_price = stock_data.Close.values
    return (stock_day, stock_month, stock_prev_close, stock_price)

# MAKE TRAINING DATA
def get_training_data(stock_features):
    stock_day, stock_month, stock_prev_close, stock_price = stock_features
    # stack all three features together
    X = np.dstack((stock_day, stock_month, stock_prev_close))[0]
    y = stock_price
    return (X, y)

# CREATE MODELS
def create_model():
    linear_model = LinearRegression()
    return linear_model


# TRAIN MODELS
def train_model(model, X, y):
    model.fit(X, y)

# PREDICTION 
def predict_price(model, X):
    # Stock Price Prediction for next 10 days
    pred_prev_close = X[-1][2]
    day = X[-1][0]
    month = X[-1][1]
    year = 2018
    predictions = {}
    for i in range(1, 11): 
        dt = date(year, int(month), int(day))
        dt += datetime.timedelta(days=i) 
        next_day_closing_price = model.predict([[dt.day, dt.month, pred_prev_close]])[0]
        pred_prev_close = next_day_closing_price
        predictions['{}-{}-{}'.format(dt.day, dt.month, dt.year)] = next_day_closing_price
    return predictions

if __name__ == '__main__':
    stock = sys.argv[1]
    print('Getting stock data for %s' % stock)
    start_date = date(2015, 1, 1)
    end_date = datetime.datetime.now().date()
    data = get_stock_data(stock, start_date, end_date)
    features = get_stock_features(data)
    X, y = get_training_data(features)
    model = create_model()
    train_model(model, X, y)
    print('Your predictions are being made....')
    predictions=predict_price(model, X)
    for date, price in predictions.items():
        print('Predicted closing price on {} is: {:.2f}'.format(date, price))
