import requests
import argparse
 
parser = argparse.ArgumentParser(description="Try predict service",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-H","--host", help="host of the service")
args = parser.parse_args()
config = vars(args)

if config["host"] == None:
  host = 'localhost:9696'
else:
  host = config["host"]
# host = 'churn-serving-env.eba-vupegvrr.eu-west-3.elasticbeanstalk.com'

url = f'http://{host}/predict'

#trip = {"vendor_id":1,"pickup_datetime":1454709496000,"passenger_count":2,"pickup_longitude":-73.8706970215,"pickup_latitude":40.7738456726,"dropoff_longitude":-73.9078292847,"dropoff_latitude":40.7532691956}
trip = {"vendor_id":1,"pickup_datetime":1467331198000,"passenger_count":1,"pickup_longitude":-73.9881286621,"pickup_latitude":40.7320289612,"dropoff_longitude":-73.9901733398,"dropoff_latitude":40.7566795349}
print('Trip: ', trip)

response = requests.post(url, json=trip).json()
print('Prediction: ', response)