import numpy as np
import pandas as pd

import tensorflow as tf
from time import time
import joblib
import json
import datetime

import trainer
import inputs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pre_data_pipe(timeseries: pd.DataFrame,column: str ):
    data = timeseries.tail(5)
    data = data[column].to_numpy()

    value = inputs.get_input(column)

    last_value = data[-1]

    data = np.append(data,value)

    diff_values = np.diff(data)
    diff_values = np.where(diff_values == 0, 0.001, diff_values)
    

    return diff_values, last_value


#################################################################################
from datetime import datetime

def save_json(column: str, value: float):
    existing_data = {}
    try:
        with open("weather_values.json", "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    current_date = datetime.now().strftime("%Y-%m-%d")
    existing_data["date"] = current_date
    existing_data[column] = value

    with open("weather_values.json", "w") as f:
        json.dump(existing_data, f)


#################################################################################

def analyser(link: str,column: str, model_obj,threshold:float):
    print('Analyser is running...')
    timeseries = pd.read_csv(link)
    values,last_value = pre_data_pipe(timeseries,column)
    

    values = values.reshape(-1, 1)

    reconstructions_a = model_obj.predict(values)
    train_loss_a = tf.keras.losses.mae(reconstructions_a.reshape(-1, 1), values)

    preds_a = tf.math.less(train_loss_a, threshold)

    result = preds_a[-1].numpy()

    if result == False:
        print('Anomaly detected! in ',column)
        save_json(column,float(last_value)+round(float(reconstructions_a[-1]),1))
    
    else:
        print('No anomaly detected! in ',column)
        save_json(column,float(last_value)+float(values[-1]))

###################################################################################
import time



def run(link: str,column: str ,threshold: float = None, timer: datetime = None):
    
    while True:
        #Load model
        model_path = column+'weather_anomaly_detector.h5'
        loaded_model = tf.keras.models.load_model(model_path)
        
        with open("threshold_values.json", "r") as f:
                shared_dict = json.load(f)
        threshold = shared_dict.get(column+"threshold", 0.000000)
        
        analyser(link,column,loaded_model, threshold)

        time.sleep(30)

