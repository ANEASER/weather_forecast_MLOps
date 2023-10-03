import json
import csv
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def save_daily_values(json_filename, csv_filename):
    try:
        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)

        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # If the CSV file is empty, write the header
            if csv_file.tell() == 0:
                header = ["date", "precipitation", "temp_max", "temp_min", "wind"]
                csv_writer.writerow(header)

            # Write the data in the specified order
            csv_writer.writerow([data["date"],
                                 data["precipitation"],
                                 data["temp_max"],
                                 data["temp_min"],
                                 data["wind"]])
    
    except FileNotFoundError:
        print(f"Error: File '{json_filename}' not found.")


json_filename = "../anomaly/weather_values.json"
csv_filename = "../Dataset/weather_data.csv"




def data_preprocessing(link:str,column:str):
       data = pd.read_csv(link)
       df = data[['date', column]].tail(375)
       df['date'] = pd.to_datetime(df['date'])
       df.set_index('date', inplace=True)
       df.index.freq = 'D'

       df[column+'_trend'] = df[column].rolling(window=3).mean()
       mean = df[column+'_trend'].mean()
       df[column+'_trend'].fillna(mean, inplace=True)

       return df


def forecast_data(df: pd.DataFrame, column: str):
    sarima = sm.tsa.SARIMAX(df[column + '_trend'], order=(1, 3, 7), seasonal_order=(1, 1, 1, 30))
    sarima_result = sarima.fit()
    prediction_result = sarima_result.get_forecast(7)
    forecast = round(prediction_result.predicted_mean,2)
    
    # Create a dictionary with date as keys and column name as a value
    forecast_dict = {date.strftime('%Y-%m-%d'): value for date, value in zip(forecast.index, forecast)}
    
    existing_data = {}
    
    try:
        with open('output.json', 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if os.path.exists('output.json'):
        os.remove('output.json')

    # Update the existing data with the new forecast
    existing_data[column] = forecast_dict

    # Append the updated data to the file
    with open('output.json', 'a') as f:
        json.dump(existing_data, f)

    
    


def forecast_results(link:str,column:str):
    dataframe = data_preprocessing(link,column)
    forecast_data(dataframe, column)


def run():

    #save_daily_values(json_filename, csv_filename)

    max_forecast = forecast_results(csv_filename, "temp_max")
    min_forecast = forecast_results(csv_filename, "temp_min")
    precipitation_forecast = forecast_results(csv_filename, "precipitation")
    wind_forecast = forecast_results(csv_filename, "wind")
