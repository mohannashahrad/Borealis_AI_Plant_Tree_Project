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