# %% [markdown]
# # 一、機器學習評估指標選定
# ## [教學目標]
# 學習 sklearn 中，各種評估指標的使用與意義

# %% [markdown]
# ## [範例重點]
# 注意觀察各指標的數值範圍，以及輸入函數中的資料格式

# %% [markdown]
# ### import 需要的套件

# %%
from sklearn import metrics, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np

# %% [markdown]
# ## 回歸問題
# 常見的評估指標有
# - MAE
# - MSE
# - R-square

# %% [markdown]
# 我們隨機生成(X, y)資料，然後使用線性回歸模型做預測，再使用 MAE, MSE, R-square 評估

# %%
X, y = datasets.make_regression(n_features=1, random_state=42, noise=100) # 生成資料
model = LinearRegression() # 建立回歸模型
model.fit(X, y) # 將資料放進模型訓練
prediction = model.predict(X) # 進行預測
mae = metrics.mean_absolute_error(prediction, y) # 使用 MAE 評估
mse = metrics.mean_squared_error(prediction, y) # 使用 MSE 評估
r2 = metrics.r2_score(prediction, y) # 使用 r-square 評估
print("MAE: ", mae)
print("MSE: ", mse)
print("R-square: ", r2)

# %%
plt.scatter(X,y)
plt.show()

# %%
plt.scatter(X, prediction)
plt.show()

# %% [markdown]
# ## 分類問題
# 常見的評估指標有
# - AUC
# - F1-Score (Precision, Recall)

# %%
cancer = datasets.load_breast_cancer() # 我們使用 sklearn 內含的乳癌資料集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=50, random_state=0)

# %%
print(y_test) # 測試集中的 label

# %%
print(X_train)

# %%
y_pred = np.random.random((50,)) # 我們先隨機生成 50 筆預測值，範圍都在 0~1 之間，代表機率值

# %%
print(y_pred)

# %% [markdown]
# ### AUC

# %%
auc = metrics.roc_auc_score(y_test, y_pred) # 使用 roc_auc_score 來評估。 **這邊特別注意 y_pred 必須要放機率值進去!**
print("AUC: ", auc) # 得到結果約 0.5，與亂猜的結果相近，因為我們的預測值是用隨機生成的

# %% [markdown]
# ### F1-Score

# %%
threshold = 0.5 
y_pred_binarized = np.where(y_pred>threshold, 1, 0) # 使用 np.where 函數, 將 y_pred > 0.5 的值變為 1，小於 0.5 的為 0
f1 = metrics.f1_score(y_test, y_pred_binarized) # 使用 F1-Score 評估
precision = metrics.precision_score(y_test, y_pred_binarized) # 使用 Precision 評估
recall  = metrics.recall_score(y_test, y_pred_binarized) # 使用 recall 評估
print("F1-Score: ", f1) 
print("Precision: ", precision)
print("Recall: ", recall)

# %% [markdown]
# ## [本節重點]
# 了解 F1-score 的公式意義，並試著理解程式碼

# %% [markdown]
# ## 練習
# 請參考 F1-score 的公式與[原始碼](https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py#L620)，試著寫出 F2-Score 的計算函數
