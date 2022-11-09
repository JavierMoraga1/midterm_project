import numpy as np
import pandas as pd
import zipfile
import pickle
import xgboost as xgb
import argparse

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

features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'month',
            'weekday', 'hour', 'hdistance', 'cdirection']

#Parameters
parser = argparse.ArgumentParser(description="Generate test predictions in a .csv file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-M","--model", help="model file")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

#model_file = 'model_002_10_4000_log2.bin'
model_file = 'XGB_log_model.bin'
if config["model"] != None:
  model_file = config["model"]
output_file = 'predictions.csv'
if config["output"] != None:
  output_file = config["output"]

# Loading and preparing the dataset
print ('Loading and preparing the dataset...')

zf = zipfile.ZipFile('test.zip')
df_test = pd.read_csv(zf.open('test.csv'))

df_test.pickup_datetime = pd.to_datetime(df_test.pickup_datetime)
df_test['month'] = df_test.pickup_datetime.dt.month.astype(str)
df_test['weekday'] = df_test.pickup_datetime.dt.weekday.astype(str)
df_test['hour'] = df_test.pickup_datetime.dt.hour.astype(str)
df_test['hdistance'] = hDistance(df_test.pickup_latitude.values, df_test.pickup_longitude.values, df_test.dropoff_latitude.values, df_test.dropoff_longitude.values)
df_test['cdirection'] = cDirection(df_test.pickup_latitude.values, df_test.pickup_longitude.values, df_test.dropoff_latitude.values, df_test.dropoff_longitude.values)

# Doing the predictions
print("Doing the predictions with model %s..." % model_file)
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

X = df_test[features].values
dX = xgb.DMatrix(X, feature_names=features)
y_pred = np.exp(model.predict(dX)) + 1

df_test['trip_duration'] = y_pred.round(0).astype(int)

df_test[['id','trip_duration']].to_csv(output_file,index=False)
print("File %s generated" % output_file)
