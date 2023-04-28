# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
api_token = {"username":"david10188","key":"3c6ba0ceb741ea66c0d22d10e0079cfc"}
import json
import zipfile
import os

if not os.path.exists("/root/.kaggle"):
    os.makedirs("/root/.kaggle")

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)

!chmod 600 /root/.kaggle/kaggle.json

if not os.path.exists('/kaggle'):
    os.makedirs('/kaggle')
os.chdir('/kaggle')
!kaggle competitions download -c new-york-city-taxi-fare-prediction
!ls /kaggle 

# %%
import numpy as np
import pandas as pd
from zipfile import ZipFile
df_train = pd.read_csv(ZipFile('/kaggle/new-york-city-taxi-fare-prediction.zip').open('train.csv'),nrows=500000)
df_test = pd.read_csv(ZipFile('/kaggle/new-york-city-taxi-fare-prediction.zip').open('test.csv'))
train_Y = np.array(df_train['fare_amount'])
ids = df_test['key']

# %%
df_train = df_train.drop(['key', 'fare_amount'] , axis=1)
df_test = df_test.drop(['key'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()
df['distance_2D'] = ((df['dropoff_longitude'] - df['pickup_longitude'])**2 + (df['dropoff_latitude'] - df['pickup_latitude'])**2)**0.5
df = df.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime'], axis=1)
df.head()

# %%
df['passenger_count'] = df['passenger_count'].fillna(-1)
df['distance_2D'] = df['distance_2D'].fillna(-1)

# %%
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor()
reg.fit(train_X, train_Y)
pred = reg.predict(test_X)

# %%
data_path = '/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/期末考/data/'
pred = np.array(pred)
sub = pd.DataFrame({'key': ids, 'fare_amount': pred})
sub.to_csv(data_path + 'baseline/taxi_baseline_1111_finalexam.csv', index=False) 


