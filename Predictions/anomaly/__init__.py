import time
import threading
import os

from .trainer import*
from .detector import*
import json
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def is_shared_value_empty():
    return not os.path.isfile("../anomaly/threshold_values.json") or os.path.getsize("../anomaly/threshold_values.json") == 0

def run_trainer(link: str, column: str):
    trainer.run(link, column)

def run_detector(link: str, column: str):
    detector.run(link, column)

def main():

    dataset_info = [
        {'link': '../Dataset/seattle-weather.csv', 'column': 'temp_max'},
        {'link': '../Dataset/seattle-weather.csv', 'column': 'temp_min'},
        {'link': '../Dataset/seattle-weather.csv', 'column': 'wind'},
        {'link': '../Dataset/seattle-weather.csv', 'column': 'precipitation'}
    ]

    threads = []

    for info in dataset_info:
        link = info['link']
        column = info['column']

        if is_shared_value_empty():
            t = threading.Thread(target=run_trainer, args=(link, column,))
            t.daemon = True
            t.start()
            threads.append(t)
        else:
            update_thread = threading.Thread(target=run_trainer, args=(link, column,))
            update_thread.daemon = True
            update_thread.start()
            threads.append(update_thread)


        detector_thread = threading.Thread(target=run_detector, args=(link, column,))
        detector_thread.daemon = True
        detector_thread.start()
        threads.append(detector_thread)

    for t in threads:
        t.join()
