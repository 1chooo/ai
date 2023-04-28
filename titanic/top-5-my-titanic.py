# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

# %%
df_train

# %%
df_train.info()

# %%
fig, ax=plt.subplots(1,figsize=(8,6))
sns.boxplot(x='Survived',y='Age', data=df_train)
ax.set_ylim(0,100)
plt.title("Survived vs Age")
plt.show()

# %%
fig, ax=plt.subplots(1,figsize=(8,6))
sns.countplot(x='Survived' ,hue='Sex', data=df_train)
ax.set_ylim(0,500)
plt.title("Survived vs Sex")
plt.show()

# %%
fig, ax=plt.subplots(1,figsize=(8,6))
sns.countplot(x='Survived' ,hue='Pclass', data=df_train)
ax.set_ylim(0,400)
plt.title("Survived vs Pclass")
plt.show()

# %%
fig, ax=plt.subplots(1,figsize=(8,6))
sns.countplot(x='Survived' ,hue='Embarked', data=df_train)
ax.set_ylim(0,500)
plt.title("Survived vs Embarked")
plt.show()

# %%
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket','Cabin'] , axis=1)
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket','Cabin'] , axis=1)

combine = [df_train, df_test]

for df in combine:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# %%
df_test.head()

# %%
df_train = df_train.drop(['SibSp', 'Parch'] , axis=1)
df_test = df_test.drop(['SibSp', 'Parch',] , axis=1)

# %%
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
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

clf = GradientBoostingClassifier()
clf.fit(train_X, train_Y)
pred = clf.predict(test_X)
pred = np.array(pred)

# %%
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
sub.to_csv('submission.csv', index=False) 

# %%



