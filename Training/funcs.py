import pandas as pd
import requests
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer    
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
from darts import TimeSeries as ts
from darts.models import Prophet


TIME_COL = 'Time'
COUNTRY_COL = "Country_name"
PRED_COUNT = 5

def load_DF(url):
  data = StringIO(requests.get(url).text)
  return pd.read_csv(data)

def standardize(df,col_names):
  df.reset_index(inplace=True, drop=True)
  features = df[col_names]
  scaler = StandardScaler().fit(features.values)
  features = scaler.transform(features.values)
  df[col_names] = features
  return df

def oneHotEncode(df, discrete_columns):
  for var in discrete_columns:
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(df[[var]]).toarray())
    enc_df.columns = enc.get_feature_names([var])
    df = df.join(enc_df)
    df = df.drop([var], axis=1)
  return df

def print_analysis(y_pred, y_test):
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('R2 score:', np.sqrt(metrics.r2_score(y_pred,y_test)))


def compare_results(y_pred, y_test):
  compare_df = pd.DataFrame({'Actual': y_test.to_numpy().flatten(), 'Predicted': y_pred.flatten()})
  compare_df['error'] = ((compare_df['Actual']-compare_df['Predicted'])/compare_df['Actual']).abs()
  compare_df.sort_values(by=['error'], ascending = False, inplace = True)
  display(compare_df.head(60))

# predicts a specific feature (col) for a country given the df and the name of the country col
def predict_col(df, time_col, country_col, country, col):
  df = df.loc[df[country_col] == country, [TIME_COL, col]]
  df[TIME_COL] = pd.to_datetime(df[TIME_COL], format = '%Y')
  series = ts.from_dataframe(df, time_col = time_col)
  model = Prophet()
  model.fit(series)
  forecast = model.predict(PRED_COUNT)
  return ts.pd_dataframe(forecast)

def predict(df, time_col, country_col, country, cols):
  merged_df = predict_col(df, time_col, country_col, country, cols[0])
  cols.pop(0)
  for feature in cols:
    merged_df = pd.concat([merged_df, predict_col(df, time_col, country_col, country, feature)], axis=1)
  return merged_df
