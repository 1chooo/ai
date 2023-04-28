# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
import numpy as np
import pandas as pd

data_path = '/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/期末考/data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket','Cabin'] , axis=1)
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket','Cabin'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

# %%
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
MMEncoder = MinMaxScaler()
LEncoder = LabelEncoder()
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].fillna('None')
        df[c] = LEncoder.fit_transform(df[c]) 
    else:
        df[c] = df[c].fillna(-1)
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))

df.head()

# %%
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(train_X, train_Y)
pred = clf.predict(test_X)

# %%
pred = np.array(pred)
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
sub.to_csv(data_path + '/data/baseline/Titanic_baseline_1111_finalexam.csv', index=False) 

# %%



