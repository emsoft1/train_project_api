
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
try:
    model = load_model('Cloud_model.h5')

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
    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns=test.columns)
    X_pred.index = test.index

    scored = pd.DataFrame(index=test.index)
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
    scored['Threshold'] = 0.05
    scored['waring'] = scored['Loss_mae'] < scored['Threshold']
    print (scored.head())

except Exception as e:
    print("Error loading model:", e)
