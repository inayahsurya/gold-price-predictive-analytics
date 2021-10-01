# -*- coding: utf-8 -*-
"""proyek predictive analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18h15eA_5PTR1SCVjA_28QxQZOei0JiYp
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

"""# Data loading

Dataset diambil dari [Kaggle Gold Price Prediction Dataset](https://www.kaggle.com/sid321axn/gold-price-prediction-dataset). Dataset tersebut dikumpulkan dari 18 November 2011 hingga 1 Januari 2019 yang berisi 1718 baris dan 80 kolom.

Target class dari dataset ini adalah Adjusted Close yang merupakan harga Close yang disesuaikan dengan faktor seperti dividen, pemecahan saham, dan penawaran saham baru untuk menentukan nilai.
"""

csv = "FINAL_USO.csv"
gold_df = pd.read_csv(csv)
X = gold_df.copy()
y = X.pop('Adj Close')
date = X.pop('Date')
X.pop('Close')
X.head()

gold_df['Date'] = pd.to_datetime(gold_df['Date'], format = '%Y-%m-%d')

# Adjusted Close with Time Series
sns.set(rc={"figure.figsize":(20, 4)})
daily_change = sns.lineplot(x="Date", y="Adj Close", 
                            data=gold_df).set(title="Adjusted Close Emas Harian")

"""# Exploratory Data Analysis

Cek jumlah kolom dan baris dari dataset
"""

row, col = X.shape
print('{} rows, {} features'.format(row, col))

"""## Deskripsi variabel

Cek informasi karakteristik dari tiap kolom
"""

X.info()

"""Cek bagaimana distribusi variabel dalam dataset"""

X.describe()

"""## Menganani missing value

Cek apakah terdapat missing value di dalam data. Karena data sudah cukup bersih, maka tidak perlu melakukan proses handling missing value
"""

# check null value
print('Null value:', X.isnull().any().sum())

"""## Univariate analysis

Analisa histogram dari tiap fitur
"""

X.hist(bins=50, figsize=(20,30))
plt.show()

"""## Multivariate analysis

analisa hubungan korelasi antar fitur pada data. koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah.

sehingga dapat disimpulkan terdapat beberapa fitur redundant (tidak berguna) pada dataset
"""

plt.figure(figsize=(25,20))
correlation_matrix = gold_df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix pada Fitur", size=20)

X.corrwith(y).plot.bar(
        figsize = (20, 10), title = "Correlation with Adj Close", fontsize = 20,
        rot = 90, grid = True)

"""# Data Preparation

## Feature selection

Analisa fitur apa saja yang mempengaruhi hasil target dengan mutual_info_regression
"""

# create mutual info score

def mutual_info_scores(X, y):
  mi_scores = mutual_info_regression(X, y)
  mi_scores = pd.Series(mi_scores, name="Mutual Info Score", index=X.columns)
  mi_scores = mi_scores.sort_values(ascending=True)

  return mi_scores

def plot_mi_scores(scores):
  score = scores.sort_values(ascending=True)
  width = np.arange(len(scores))
  ticks = list(scores.index)
  plt.figure(figsize=(20,30))
  plt.barh(width, scores)
  plt.yticks(width, ticks)
  plt.title("Mutual Information Scores")

mi_scores = mutual_info_scores(X, y)
plot_mi_scores(mi_scores)

"""setelah diurutkan, maka akan tampak fitur-fitur yang paling berpengaruh dengan target. kemudian ambil 10 fitur dengan mutual info score tertinggi untuk dijadikan data."""

mi_scores = mi_scores.sort_values(ascending=False)
selected_features = mi_scores.index[:10].to_list()
X = X[selected_features]
X.head()

"""## Train test split

membagi data menjadi data training dan testing dengan rasio 80%:20%
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
print(f'Total sample in whole dataset: {len(X)}')
print(f'Total sample in train dataset: {len(X_train)}')
print(f'Total sample in test dataset: {len(X_test)}')

"""## Standarisasi

Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

StandardScaler melakukan proses standarisasi fitur dengan z-score, yaitu mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0.
"""

# z-score standardization
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train, columns=X.columns)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_train.head()

"""## Dimension reduction dengan PCA"""

# dimension reduction
pca = PCA().fit(X_train)
X_train_pca = pca.transform(X_train)
X_train_pca = pd.DataFrame(X_train_pca, columns=selected_features)

X_test_pca = pca.transform(X_test)
X_test_pca = pd.DataFrame(X_test_pca, columns=selected_features)

X_train_pca.head()

"""# Model Development

## Linear regression
"""

# linear regression

linreg_model = LinearRegression()
linreg_model.fit(X_train_pca, y_train)
linreg_pred = linreg_model.predict(X_test_pca)

linreg_df = pd.DataFrame({'y_true': y_test,
                          'y_pred': linreg_pred})

linreg_mse = mean_squared_error(y_test, linreg_pred)
linreg_mae = mean_absolute_error(y_test, linreg_pred)

linreg_df.head()

"""## Random forest regression"""

rf_model = RandomForestRegressor(n_estimators=50, max_depth=16,
                                 random_state=55, n_jobs=-1)
rf_model.fit(X_train_pca, y_train)
rf_pred = rf_model.predict(X_test_pca)

rf_df = pd.DataFrame({'y_true': y_test,
                          'y_pred': rf_pred})

rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_df.head()

"""## Gradient booster"""

xgb_model = XGBRegressor()
xgb_model.fit(X_train_pca, y_train)
xgb_pred = xgb_model.predict(X_test_pca)

xgb_df = pd.DataFrame({'y_true': y_test,
                          'y_pred': xgb_pred})

xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)

xgb_df.head()

"""# Model Evaluation

mengevaluasi model dengan Mean Squared Error (MSE) dan Mean Absolute Error (MAE) dari hasil prediksi data testing.

Dapat disimpulkan bahwa model Linear Regression mendapatkan nilai error paling kecil dari MSE dan MAE nya.
"""

metrics_df = pd.DataFrame({'MSE': [linreg_mse, rf_mse, xgb_mse],
                           'MAE': [linreg_mae, rf_mae, xgb_mae]},
                          index=['Linear Regression', 'Random Forest', 'Gradient Booster'])
metrics_df

"""melakukan plotting hasil prediksi dan hasil aktual. dapat disimpulkan bahwa model bekerja cukup baik"""

data = pd.DataFrame({'Linear Regression Prediction': linreg_pred,
                     'Actual': y_test})
data.plot()