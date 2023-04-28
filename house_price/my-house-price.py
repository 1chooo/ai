# %% [markdown]
# # Final Exam - House Prices
# 
# Course: AP4063
# 
# Student Number: 109601003
# 
# Name: 林群賀
# 
# #### Baseline: 0.1333

# %% [markdown]
# ## Import the package

# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile

# %% [markdown]
# ## Import the data sets.
# 

# %%
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# df_train = pd.read_csv(train_data_path)
# df_test = pd.read_csv(test_data_path)

# %%
y = df_train['SalePrice']
train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)

# %%
df_train.head()

# %%
df_train.info()

# %%
df_train.describe()

# %% [markdown]
# #### Check the missing values

# %%
sns.set_style("whitegrid")
missing = df_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

# %%
import scipy.stats as stats

plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)

# %%
df_train = df_train.drop(['FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC'] , axis=1)
df_test = df_test.drop(['FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC'] , axis=1)

df_train = df_train.fillna(df_train.mean()) 

# %%
df_train

# %% [markdown]
# #### If dropping a significant number of features, the presicion will get down to 0.14313

# %%
# df_train = pd.get_dummies(df_train)
# df_test = pd.get_dummies(df_test)

# %%
# df_train

# %%
# #Drop features that are correlated to each other

# covarianceMatrix = df_train.corr()
# listOfFeatures = [i for i in covarianceMatrix]
# setOfDroppedFeatures = set() 
# for i in range(len(listOfFeatures)) :
#     for j in range(i+1,len(listOfFeatures)): #Avoid repetitions 
#         feature1=listOfFeatures[i]
#         feature2=listOfFeatures[j]
#         if abs(covarianceMatrix[feature1][feature2]) > 0.8: #If the correlation between the features is > 0.8
#             setOfDroppedFeatures.add(feature1) #Add one of them to the set
# #I tried different values of threshold and 0.8 was the one that gave the best results

# df_train = df_train.drop(setOfDroppedFeatures, axis=1)
# # df_test = df_train.drop(setOfDroppedFeatures, axis=1)

# %%
# #Drop features that are not correlated with output

# nonCorrelatedWithOutput = [column for column in df_train if abs(y.corr(y)) < 0.045]
# #I tried different values of threshold and 0.045 was the one that gave the best results

# df_train = df_train.drop(nonCorrelatedWithOutput, axis=1)

# %%
# df_train

# %%
plt.plot(df_train['LotArea'], y, 'bo')
plt.axvline(x=75_000, color='r')
plt.ylabel('SalePrice')
plt.xlabel('LotArea')
plt.title('SalePrice in function of LotArea')

plt.show()

# %%
df = pd.concat([df_train,df_test])

# %%
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].fillna('None')
        df[c] = LEncoder.fit_transform(df[c]) 
    else:
        df[c] = df[c].fillna(-1)
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))

df.head()

# %% [markdown]
# ## Prediction
# 
# #### With the Gradient Boosting Machine Model

# %%
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor()
reg.fit(train_X, train_Y)
pred = reg.predict(test_X)

# %% [markdown]
# ## Output the Results

# %%
pred = np.expm1(pred)
sub = pd.DataFrame({'Id': ids, 'SalePrice': pred})
sub.to_csv('submission.csv', index=False) 
# sub.to_csv(data_path + 'house_1111_finalexam.csv', index=False) 

# %% [markdown]
# ## My Result
# 
# #### The screenshot
# 
# ![the result](./data/my_result/house_1111_finalexam.png)

# %% [markdown]
# ## What I have tried?
# 
# 1. 
#    I first examined the data, but found that there were a lot of variables. 
#    
#    Additionally, I discovered that there were many pieces of data that could not be used. 
#    
#    As a result, I referred to the approach used in this "#1 House Prices Solution [top 1%]" and replaced the data. However, when I continued to implement it, I found that the accuracy decreased. 
#    
#    Also, I did not understand the author's approach very well, so I decided not to use it.

# %% [markdown]
# ## Reference
# 
# [#1 House Prices Solution [top 1%]](https://www.kaggle.com/code/jesucristo/1-house-prices-solution-top-1#Submission)
# 
# [House Prices Complete Solution with Guidance](https://www.kaggle.com/code/marweni/house-prices-complete-solution-with-guidance)


