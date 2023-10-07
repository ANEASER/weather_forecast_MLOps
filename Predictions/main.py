import predictor
import anomaly
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score


def get_predictions(probabilities):
   df_probabilities= pd.DataFrame()
   df_probabilities['probabilities'] = probabilities.tolist()
   df_probabilities['second_value'] = round(df_probabilities['probabilities'].apply(lambda x: x[1]*100),2)
   return df_probabilities['second_value']


def classify(test:pd.DataFrame, train:pd.DataFrame):
   X = train[['precipitation','temp_max','temp_min', 'wind']]
   Y = train['weather']

   test = test[['precipitation','temp_max','temp_min', 'wind']]

   X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
   logistic = LogisticRegression()
   model = logistic.fit(X_train, Y_train)
   predictions = model.predict(X_test) 
   acc = accuracy_score(Y_test, predictions)

   if acc > 0.9:
      with open('model.pkl', 'wb') as f:
         pickle.dump(model, f)
      predictions = model.predict(test)
      probabilities = model.predict_proba(test)
      probabilities = get_predictions(probabilities)
      return predictions, probabilities
   else:
      with open('model.pkl', 'rb') as f:
         model = pickle.load(f)
      predictions = model.predict(test)
      probabilities = model.predict_proba(test)
      probabilities = get_predictions(probabilities)
      return predictions, probabilities


def main():
   try:
      anomaly.main()
      print('Anomaly detection completed!')
      predictor.run()
   except Exception as e:
      print(f"An error occurred in anomaly.main(): {str(e)}")
   
   df_test = pd.read_json('output.json')
   df_train = pd.read_csv('../Dataset/weather_data.csv')
   df_test.reset_index(inplace=True)
   df_test.rename(columns={'index': 'date'}, inplace=True)

   predictions, probabilities = classify(df_test, df_train)
   df_test['weather'] = predictions
   df_test['Chance of rain'] = probabilities.tolist()
   
   df_test.to_csv('weather_forecasts.csv', index=False)

   os.remove('output.json')

main()