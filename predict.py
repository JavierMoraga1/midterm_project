import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import argparse

from flask import Flask
from flask import request
from flask import jsonify

# Functions to calculate Haversine distance & Compass direction
def hDistance(lat1, lng1, lat2, lng2):
  lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
  R = 6371 #Earth radius
  lat = lat2 - lat1
  lng = lng2 - lng1
  d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
  h = 2 * R * np.arcsin(np.sqrt(d))
  return h

def cDirection(lat1, lng1, lat2, lng2):
  lng_delta_rad = np.radians(lng2 - lng1)
  lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
  y = np.sin(lng_delta_rad) * np.cos(lat2)
  x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
  return np.degrees(np.arctan2(y, x))

#Parameters
parser = argparse.ArgumentParser(description="Run /predict service",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-P","--port", help="port")
parser.add_argument("-M","--model", help="model file")
args = parser.parse_args()
config = vars(args)

model_file = 'XGB_log_model.bin'
if config["model"] != None:
  model_file = config["model"]
port = 9696
if config["port"] != None:
  port = config["port"]  

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('trip_duration')

@app.route('/predict', methods=['POST'])
def predict():
    trip = request.get_json(force=True)
    print(trip)

    df_request = pd.DataFrame.from_dict([trip])
    df_request.pickup_datetime = pd.to_datetime(df_request.pickup_datetime)
    df_request['month'] = df_request.pickup_datetime.dt.month.astype(str)
    df_request['weekday'] = df_request.pickup_datetime.dt.weekday.astype(str)
    df_request['hour'] = df_request.pickup_datetime.dt.hour.astype(str)
    df_request['hdistance'] = hDistance(df_request.pickup_latitude.values, df_request.pickup_longitude.values, df_request.dropoff_latitude.values, df_request.dropoff_longitude.values)
    df_request['cdirection'] = cDirection(df_request.pickup_latitude.values, df_request.pickup_longitude.values, df_request.dropoff_latitude.values, df_request.dropoff_longitude.values)

    features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
              'dropoff_longitude', 'dropoff_latitude', 'month',
              'weekday', 'hour', 'hdistance', 'cdirection']

    X = df_request[features].values
    dX = xgb.DMatrix(X, feature_names=features)
    y_pred = np.exp(model.predict(dX)) + 1

    print(y_pred)
    result = {
        'trip_duration': int(y_pred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)