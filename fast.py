import pandas as pd
# $WIPE_BEGIN
from tensorflow.keras.models import load_model


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from taxifare.ml_logic.registry import load_model
# from taxifare.ml_logic.preprocessor import preprocess_features
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model  = load_model('Cloud_model.h5')

print("Model loaded successfully")

merged_data = pd.read_csv("new_preprocces.csv" ,index_col=0)
train = merged_data[800:950]
test = merged_data[800:950]
test_norm = merged_data[600:750]
test_start = merged_data[730:880]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
X_test_norm = scaler.transform(test_norm)
X_test_start = scaler.transform(test_start)
scaler_filename = "scaler_data"
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], 1, X_test_norm.shape[1])
X_test_start = X_test_start.reshape(X_test_start.shape[0], 1, X_test_start.shape[1])
print("Test data shape:", X_test.shape)
X_pred = app.state.model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.05
scored['waring'] = scored['Loss_mae'] < scored['Threshold']
print (scored.head())


# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
# app.state.model = load_model()
# $WIPE_END

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2

@app.get("/predict")
def predict(test_name: str):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # $CHA_BEGIN

    # 💡 Optional trick instead of writing each column name manually:
    # locals() gets us all of our arguments back as a dictionary
    # https://docs.python.org/3/library/functions.html#locals
    train = merged_data[800:950]
    test = merged_data[800:950]
    test_norm = merged_data[600:750]
    test_start = merged_data[730:880]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_test_norm = scaler.transform(test_norm)
    X_test_start = scaler.transform(test_start)
    scaler_filename = "scaler_data"
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], 1, X_test_norm.shape[1])
    X_test_start = X_test_start.reshape(X_test_start.shape[0], 1, X_test_start.shape[1])
    if test_name == "failer_test":
        print ("fail")
    elif test_name == "norm_test":
        X_test=X_test_norm
    elif test_name == "between_test":
        X_test=X_test_start



    X_pred = app.state.model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index

    scored = pd.DataFrame(index=test.index)
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    scored['Threshold'] = 0.05
    scored['waring'] = scored['Loss_mae'] > scored['Threshold']

    print (scored.head())
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    res = scored.head().to_dict()
    warnings = np.array(scored['waring'])
    num_true = np.sum(warnings)
    threshold = 0.6 * warnings.size
    result=bool( num_true > threshold)

    gg = X_test.reshape(X_test.shape[0], X_test.shape[2])

    res =  {"warning":result , "vib" :list (gg[:,0]) }
    print (res)
    return dict(results=res )
    # $CHA_END


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END

@app.get("/test_list")
def root():
    res = ["failer_test" , "norm_test","between_test"]
    return dict(results=res)
    # $CHA_END
