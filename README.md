# New York City Taxi Trip Duration

## Problem description
___
For this midterm project I've choosen a dataset from a Kaggle competitition ([**New York City Taxi Trip Duration**](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview)), with a subset of data specifically selected and cleaned from those originally published by the NYC Taxi and Limousine Commission (TLC).

The goal is **predict the duration (secs) of a taxi trip from the data known at its beginning** (vendor, initial al final position, datetime, passenger_count)

To get a score in the competition, you must first built and train a model using a training dataset and then use another testing dataset to calculate and submit a set of predictions

### Used Datasets
___

Description and dowload: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data
(also, a copy of the files is available in the repository)

- train.csv: Used to train the model
- test.csv: Used to predict and submit the results


### Variables of the training dataset (according to Kaggle's description)
___

- **id** - a unique identifier for each trip
- **vendor_id** - a code indicating the provider associated with the trip record
- **pickup_datetime** - date and time when the meter was engaged
- **dropoff_datetime** - date and time when the meter was disengaged
- **passenger_count** - the number of passengers in the vehicle (driver entered value)
- **pickup_longitude** - the longitude where the meter was engaged
- **pickup_latitude** - the latitude where the meter was engaged
- **dropoff_longitude** - the longitude where the meter was disengaged
- **dropoff_latitude** - the latitude where the meter was disengaged
- **store_and_fwd_flag** - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
- **trip_duration** - duration of the trip in seconds (Target variable)

### Constraints
___

I've respected the main constraints of the competition. That means:
1. Using only the variables of the test set (i.e. the known data at the beginning of a trip). So, I didn't use `dropoff_datetime` or `store_and_fwd_flag`
2. The evaluation metric is RMSLE (Root Mean Square Logarithmic Error). RMSLE is better metric than RMSE if there are data very far from the mean

### Additional dificulties
___
1. **The dataset has a very large size (1458644 records, 196MB)**, which sometimes has forced me to reduce its size to be able to train and evaluate the different models on my computer with a reasonable performance
2. **The dataset has a good number of "outliers"** (trip_duration values very far from the mean) which make the result of the metric much worse (despite using RMSLE)

## Approach and solutions
___
- For this regression problem I've trained the different types of algorithms seen in the course so far and I've evaluated them using RMSLE
- I've selectioned ___XGBoost___ as the best model and tuned its parameters. Given the dispersion of the data and the type of metric, the model has to use logarithms
- Since the dataset is very large, the model metrics seem to keep improving (very slightly) with a large number of iterations, but the size of the model also increases a lot and its perfomance degrades. For the competition I have used my best model without overfitting (4000 iterations) but for a deployment I think a lightweight model with a lot less iterations (e.g. 200) would probably work better
- Although the Kaggle's competition ended 5 years ago, I have submitted my results for evaluation. The RMSLE score (0.38944) is in the 33th percentile
- ___Future improvement___: I think a good improvement would be to try to classify each trip to separate the main population and the outliers and apply a different model for each case

## Repo description
___
- `notebook.ipynb` -> Notebook for EDA, data cleaning, model testing and other preparatory steps
- `train.py` -> Script with the entire process to train the model from the initial loading of the dataset to the generation of the `.bin` file
- `predict.py` -> Creates an application that uses port 9696 and the endpoint `/predict` (`POST` method) to:
  - Receive data request in json format
  - Validate and adapt the data to the model
  - Predict and return the result in json format
- `predict_test.py` -> Sends a test request and prints the result
- `generate_predictions.py` -> Script to generate the results to be sent to Kaggle. It creates the (`predictions.csv`) file
- `Pipfile`, `Pipfile.lock` -> For dependency management using Pipenv
- `Dockerfile` -> Required for build a docker image of the app with the 
predict service
- `XGB_log_model.bin` -> Model file used by predict.py. Another can be generated using train.py

## Dependencies Management
___
In order to manage the dependencies of the project you could:
- Install Pipenv => `pip install pipenv`
- Use it to install the dependencies => `pipenv install`

## Train the model
___
1. Enter the virtual environment => `pipenv shell`
2. Run script => `train.py`
or
1. Run directly inside the virtual environment => `pipenv run train.py`

Optional parameters:
- `-E --eta` (default 0.05)
- `-D --maxdepth` (default 10)
- `-R --nrounds` (default 200)
- `-S --nsplits` (default 5)
- `-O --output` (default `XGB_log_model.bin`)
## Run the app locally (on port 9696)
___
1. Enter the virtual environment => `pipenv shell`
2. Run script => `predict.py`
or
1. Run directly inside the ve => `pipenv run predict.py`.

This use a development server. You can also run locally the app using a deployment server as `gunicorn` (Linux) o `waitress` (Windows), if it is installed in your environment
- Linux -> `gunicorn --bind=0.0.0.0:9696 predict:app`
- Windows -> `waitress-serve --listen=*:9696 predict:app`
## Using Docker
___
You can also build a Docker image using the provided `Dockerfile` file =>
`docker build . -t tripduration`

To run this image locally => `docker run -it --rm -p 9696:9696 tripduration`
## Deploying to AWS
___
Assuming you have a EWS account you can deploy the image to AWS Elastic Beanstalk with the next steps:
- Install CLI into your ve => `pipenv install awsebcli –dev`
- `pipenv shell`
- Init EB configuration => `eb init -p docker -r your-zone tripduration` and authenticate with your credentials
- Create and deploy the app => `eb create tripduration-env`
## Using the service
___
- For a simple test request you can run the test script => `python predict_test.py` (needs `requests`. To install => `pip install requests`)
Optional parameter =>` -H --host` (`localhost:9696` by default)
Until the end of the project review period the application will remain deployed on AWS Elastic Beanstalk and accessible on tripduration-env.eba-3sk54gy3.eu-west-3.elasticbeanstalk.com
=> `python predict_test.py --host tripduration-env.eba-3sk54gy3.eu-west-3.elasticbeanstalk.com`

- Or use curl, Postman or another similar tool for the request.

## Generating a file to submit the predictions to Kaggle
___
For generate a .csv file with the predictions => `pipenv run generate_predictions`. Optional parameters:
- `-M --model` (default `XGB_log_model.bin`)
- `-O --output` (default `predictions.csv`)












