from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

def preprocess_data(data):
    def posix_time(dt):
        return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)

    data = pd.read_csv('traffic_volume_data.csv')
    data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)

    last_n_hours = [1, 2, 3, 4, 5, 6]
    for n in last_n_hours:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)

    data = data.dropna().reset_index(drop=True)
    data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
    data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
    data['is_holiday'] = data['is_holiday'].astype(int)
    data['date_time'] = pd.to_datetime(data['date_time'])

    data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
    data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
    data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
    data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
    data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))

    data.to_csv("traffic_volume_data.csv", index=None)
    data = pd.read_csv("traffic_volume_data.csv")

    data = data.sample(9950, replace=True).reset_index(drop=True)
    label_columns = ['weather_type', 'weather_description']
    numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']
    features = numeric_columns + label_columns

    X = data[features]

    def unique(list1):
        ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
        return ans

    n1 = data['weather_type']
    n2 = data['weather_description']
    unique_n1 = unique(n1)
    unique_n2 = unique(n2)

    n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
    n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain',
                  'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

    # Data Preparation
    n11 = []
    n22 = []
    for i in range(9950):
        if n1[i] not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1[i])) + 1)
        if n2[i] not in n2features:
            n22.append(0)
        else:
            n22.append((n2features.index(n2[i])) + 1)

    # Apply label encoding to the dataframe
    data['weather_type'] = n11
    data['weather_description'] = n22

    # Separate numeric and label columns
    label_columns = ['weather_type', 'weather_description']
    numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']
    features = numeric_columns + label_columns
    target = ['traffic_volume']

    X = data[features]
    y = data[target]

    # Apply MinMax scaling to both numeric and label columns
    x_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y).flatten()

    # Train your machine learning model here
    model = MLPRegressor(random_state=1, max_iter=500).fit(X, y)

    return data, model, x_scaler

def scale_features(X, scaler):
    # Scale the features using the provided scaler
    X_scaled = scaler.transform(X)
    return X_scaled


def predict_traffic(model, features, x_scaler):
    # Make predictions using the trained model
    scaled_features = x_scaler.transform(features)
    predicted_traffic = model.predict(scaled_features)
    return predicted_traffic[0]


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    day = int(request.form['day'])
    time = request.form['time']
    temperature = int(request.form['temperature'])
    is_holiday = int(request.form['isholiday'] == 'yes')
    x0 = request.form['x0']
    x1 = request.form['x1']

    # Feature engineering for weather_type and weather_description
    weather_type_features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
    weather_description_features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain', 'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

    weather_type = weather_type_features.index(x0) + 1 if x0 in weather_type_features else 0
    weather_description = weather_description_features.index(x1) + 1 if x1 in weather_description_features else 0

    # Prepare the input features for prediction
    features = [[is_holiday, temperature, day, int(time[:2]), int(date[8:]), int(date[:4]), int(date[5:7]), weather_type, weather_description]]

    # Load the trained model and scaler
    data, model, x_scaler = preprocess_data(pd.read_csv("traffic_volume_data.csv"))
    X = data[['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month', 'weather_type', 'weather_description']]
    y = data['traffic_volume']

    # Scale the input features
    features_scaled = scale_features(features, x_scaler)

    # Make prediction
    predicted_traffic = predict_traffic(model, features_scaled, x_scaler)

    # Categorize the predicted traffic
    if predicted_traffic <= 30:
        category = "No traffic"
    elif 30 < predicted_traffic <= 40:
        category = "Busy or normal traffic"
    elif 50 < predicted_traffic <= 100:
        category = "Heavy traffic"
    else:
        category = "Worse case"

    # Pass the predicted_traffic and category values to the HTML template
    return render_template('output.html', statement=predicted_traffic, category=category)

@app.route('/output')
def output():
    predicted_traffic = request.args.get('predicted_traffic')
    return render_template('index.html', predicted_traffic=predicted_traffic)


if __name__ == '__main__':
    app.run(debug=True)
