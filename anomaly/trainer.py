import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def data_loader(timeseries: pd.DataFrame, column: str):
    data = timeseries.tail(365)
    data = data[column].to_numpy()

    diff_values = np.diff(data)
    diff_values = np.where(diff_values == 0, 0.001, diff_values)

    return diff_values.reshape(-1, 1)

########################################################################################
"""
from sklearn.preprocessing import MinMaxScaler
import joblib

def scale_values(array: np.array):
    scaler = MinMaxScaler()
    scaler.fit(array.reshape(-1, 1))
    scaled_array = scaler.transform(array.reshape(-1, 1))

    scaler_filename = "weather_scaler_weights_1.pkl"
    joblib.dump(scaler, scaler_filename)

    return scaled_array
"""

########################################################################################
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential


def model_train(train:np.array,column:str, epochs:int=500, batch_size:int=128, shuffle:bool=True, verbose:int=0, monitor:str='val_loss', patience:int=5, mode:str='min',optimizer:str='adam',loss:str='mse'):

    AutoEncoder = Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(train), 1)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model = AutoEncoder

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience,mode=mode)

    model.compile(optimizer=optimizer, loss=loss)

    train = train.reshape((-1,364,1))

    print("Training model... ",column)
    history = model.fit(train, train, epochs=epochs, batch_size=batch_size, validation_data=(train, train), shuffle=shuffle, verbose=verbose, callbacks=[early_stopping])
    print(column," Model trained!")

    reconstructions = model.predict(train)
    train_loss = tf.keras.losses.mae(reconstructions,train)

    threshold = np.mean(train_loss) + 3*np.std(train_loss)

    model.save(column+"weather_anomaly_detector.h5")
    
    return threshold


def run(link: str, column: str):
    while True:
        data = data_loader(pd.read_csv(link), column)
    
        thresh = model_train(data, column)

        existing_data = {}
        try:
            with open("threshold_values.json", "r") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        existing_data[column + "threshold"] = thresh

        with open("threshold_values.json", "w") as f:
            json.dump(existing_data, f)
        
        time.sleep(60*2)

