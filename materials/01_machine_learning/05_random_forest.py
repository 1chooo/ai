# %% [markdown]
# # 五、隨機森林
# ## [範例重點]
# 了解隨機森林的建模方法及其中超參數的意義

# %% [markdown]
# ### import 需要的套件

# %%
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# %%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=10000, max_depth=10)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

# %%
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
#0.9736842105263158
#0.9736842105263158

# %%
print(iris.feature_names)

# %%
print("Feature importance: ", clf.feature_importances_)
#[0.06292474 0.02576675 0.40464774 0.50666078]
#[0.10037212 0.03238696 0.43257994 0.43466098]

# %% [markdown]
# ## 練習
# 
# 1. 試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較