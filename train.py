import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import zipfile
import xgboost as xgb
import pickle
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
parser.add_argument("-E","--eta", help="eta")
parser.add_argument("-D","--maxdepth", help="maximum depth")
parser.add_argument("-R","--nrounds", help="number of rounds")
parser.add_argument("-S","--nsplits", help="number of splits in cross validation")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

eta = 0.04
if config["eta"] != None:
  eta = float(config["eta"])
max_depth = 9
if config["maxdepth"] != None:
  max_depth = int(config["maxdepth"])
xgb_num_rounds = 200
if config["nrounds"] != None:
  xgb_num_rounds = int(config["nrounds"])
n_splits = 5
if config["nsplits"] != None:
  n_splits = int(config["nsplits"])
output_file = 'XGB_log_model.bin'
if config["output"] != None:
  output_file = config["output"]

xgb_params = {'eta': eta, 'max_depth': max_depth, 'min_child_weight': 1, 'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1, 'verbosity': 1}

# Loading and preparing the dataset
print ('Loading and preparing the dataset...')
print('Doing validation with ETA=%.2f' % eta)

zf = zipfile.ZipFile('train.zip')
df = pd.read_csv(zf.open('train.csv'))

# Calculation of new derived variables with potentially important information for the model
df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
df['month'] = df.pickup_datetime.dt.month.astype(str)
df['weekday'] = df.pickup_datetime.dt.weekday.astype(str)
df['hour'] = df.pickup_datetime.dt.hour.astype(str)
df['hdistance'] = hDistance(df.pickup_latitude.values, df.pickup_longitude.values, df.dropoff_latitude.values, df.dropoff_longitude.values)
df['cdirection'] = cDirection(df.pickup_latitude.values, df.pickup_longitude.values, df.dropoff_latitude.values, df.dropoff_longitude.values)

# Split train,val,test and extracts target
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Functions for Training and Validation
def train(df_train, y_train):
  X_train = df_train[features].values
  y_train = np.log1p(y_train)

  dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
  model = xgb.train(xgb_params, dtrain, num_boost_round=xgb_num_rounds)

  return model

def predict(df_pred, model):
  X_pred = df_pred[features].values
  dpred = xgb.DMatrix(X_pred, feature_names=features)
  y_pred = np.exp(model.predict(dpred)) + 1

  return y_pred

def rmsle(y, y_pred):
    error = np.log1p(y_pred) - np.log1p(y)
    msle = (error ** 2).mean()
    return np.sqrt(msle)

# Validating
print('Doing validation with %s splits (ETA=%.2f Max_Depth=%s Num_Boost_Rounds=%s)...' % (n_splits, eta, max_depth, xgb_num_rounds))
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.trip_duration.values
    y_val = df_val.trip_duration.values

    model = train(df_train, y_train)
    y_pred = predict(df_val, model)

    RMSLE = rmsle(y_val, y_pred)
    scores.append(RMSLE)

    print(f'RMSLE on fold {fold} is {RMSLE}')
    fold = fold + 1

print('Validation results:')
print('RMSE: %.3f +- %.3f' % (np.mean(scores), np.std(scores)))

# Training the final model
print('Training the final model...')
model = train(df_full_train, df_full_train.trip_duration.values)
y_pred = predict(df_test, model)

y_test = df_test.trip_duration.values
RMSLE = rmsle(y_test, y_pred)

print(f'RMSLE={RMSLE}')

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print(f'The model is saved to {output_file}')
