import joblib
import pandas as pd
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from HousePriceProject.helpers import *

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# pandas_options
set_pandas_options()

# import dataset

train = pd.read_csv("HousePriceProject/dataset/train.csv")
test = pd.read_csv("HousePriceProject/dataset/test.csv")

train.head()
test.head()

##################################################################################
# EDA
##################################################################################
# 1. Genel Bakış
check_df(train, 10)

# değişkenlerin yakalanması

cat_cols, num_cols, cat_but_car = grab_col_names(train, cat_th=17, car_th=26)

# cat_cols_test, num_cols_test, cat_but_car_test, = grab_col_names(test, cat_th=17, car_th=26)

# 2. Kategorik Değişken Analizi
for col in cat_cols:
    cat_summary(train, col, plot=True)

# 3. Nümerik Değişken Analizi
for col in num_cols:
    num_summary(train, col)

# 4. Hedef Değişken Analizi

for col in cat_cols:
    target_summary_with_cat(train,target="SalePrice",categorical_col=col)
# Missingleri doldurmak için kullanılabilecek değişkenler: Neighborhood, MSZoning, MSSubClass

# Bağımlı değişken sola çarpık görünüyor
train["SalePrice"].hist(bins= 100)
plt.show()

# logaritmik dönüşüm ile çarpıklığı törpülüyoruz.
np.log1p(train['SalePrice']).hist(bins=100)
plt.show()

# 5. Korelasyon analizi
high_correlated_cols(train, corr_th=0.9)

##################################################################################
# Data Preprocessing
##################################################################################
### 1. Outliers

for col in num_cols:
    print(f"{col}:", {check_outlier(train, col)})

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(train, col)

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(test, col)

# İşlemler işe yaradı mı ?
for col in num_cols:
    print(f"{col}:", {check_outlier(train, col)})

for col in num_cols:
    if col != "SalePrice":
        print(f"{col}:", {check_outlier(train, col)})

### 2. Missing Values
na_columns_train = missing_values_table(train, na_name=True)
na_columns_test = missing_values_table(test, na_name=True)

# missing olmanın hedef değişken üzerinde bir anlamı var mı? Var. Olmamayı temsil ediyor
missing_vs_target(train, "SalePrice", na_columns_train)
# missing_vs_target(test, "SalePrice", na_columns_test)

# train ve test içerisindeki null değerlere missing ataması yapıyoruz çünkü anlam ifade ediyorlar.
for col in cat_cols:
    train[col].fillna("missing", inplace=True) # kategorik değişkenlerdeki missing değerler anlam ifade ediyor.

for col in cat_cols:
    test[col].fillna("missing", inplace=True)


# geriye sadece nümerik değerler kalacak
na_columns_train = missing_values_table(train, na_name=True)
na_columns_test = missing_values_table(test, na_name=True)

# nümerik null değerleri test ve trainde ortalama ile dolduruyoruz.
for col in num_cols:
    train[col].fillna(train.groupby("Neighborhood")[col].transform("mean"), inplace=True)

for col in num_cols:
    if col != 'SalePrice':
        test[col].fillna(test.groupby("Neighborhood")[col].transform("mean"), inplace=True)

na_columns_train = missing_values_table(train, na_name= True) # null değer yok
na_columns_test = missing_values_table(test, na_name= True)

### 3. Rare Encoder
rare_analyser(train, "SalePrice", cat_cols)

rare_encoder(train, rare_perc= 0.2, cat_cols= cat_cols)
rare_encoder(test, rare_perc= 0.2, cat_cols= cat_cols)

useless_cols = [col for col in cat_cols if train[col].nunique() == 1 or
                    (train[col].nunique() == 2 and (train[col].value_counts() / len(train) <= 0.01).any(
                        axis=None))]

train.head()
test.head()

##################################################################################
# Feature Engineering (Train)
##################################################################################

# Age
train["NEW_age"] = 2022 - train["YearBuilt"].astype(int)
train.loc[(train["NEW_age"] >= 0) & (train["NEW_age"] < 5), "NEW_status"] = "new"
train.loc[(train["NEW_age"] >= 5) & (train["NEW_age"] < 15), "NEW_status"] = "mid"
train.loc[(train["NEW_age"] >= 15) & (train["NEW_age"] < 25) , "NEW_status"] = "mid-old"
train.loc[train["NEW_age"] >= 25, "NEW_status"] = "old"

# OverAll Skor
train["NEW_OverallSkor"] = train["OverallQual"] * train["OverallCond"]

# Bahçe Alanı
train['TotalPorchSF'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])

# Parsel Büyüklüğü
train.loc[(train["LotArea"] >= 0) & (train["LotArea"] < 5000), "NEW_AreaSize"] = "Small"
train.loc[(train["LotArea"] >= 5000) & (train["LotArea"] < 10000), "NEW_AreaSize"] = "Medium"
train.loc[(train["LotArea"] >= 10000) , "NEW_AreaSize"] = "Big"

# Condition(Metroya yakınlık vs.) Skor
train["New_CondScore"] = train["Condition1"].apply(lambda x: 1 if x != "Norm" else 0 ) + train["Condition2"].apply(lambda x: 1 if x != "Norm" else 0)

# Dış Cephe Skoru
train["NEW_ExterScore"] = train["ExterQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else 1)))) + \
train["ExterCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else 1))))

# Basement Skoru
train["NEW_BsmtScore"] = train["BsmtQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
train["BsmtCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
train["BsmtExposure"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))

# Garage Skoru
train["NEW_GarageYear"] = 2022 - train["GarageYrBlt"]
train["NEW_GarageScore"] = train["GarageQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
train["GarageCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
train["GarageFinish"].apply(lambda x: 3 if x == "Fin" else (2 if x == "RFn" else (1 if x == "Unf" else 0))) + \
train["PavedDrive"].apply(lambda x: 2 if x == "Y" else (1 if x == "P" else 0)) / train["NEW_GarageYear"].astype(int)

# Fireplace Skoru
train["NEW_FirePlaceScore"] = train["FireplaceQu"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))\
*train["Fireplaces"]

# Mutfak Skoru
train["NEW_KitchenScore"] = train["KitchenQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))\
*train["KitchenAbvGr"]

# Banyo Sayısı
train["NEW_NumberOfBath"] = train["BsmtFullBath"] + train["BsmtHalfBath"]* 1/2 +train["FullBath"] + train["HalfBath"]*1/2

# Evin Alanı
train["HouseArea"] = train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF']

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler(feature_range= (0,10))
# Evin Skoru
train[["NEW_KitchenScore","NEW_FirePlaceScore","NEW_GarageScore","NEW_BsmtScore","NEW_ExterScore"]] = minmax.fit_transform(train[["NEW_KitchenScore","NEW_FirePlaceScore","NEW_GarageScore","NEW_BsmtScore","NEW_ExterScore"]])

train["NEW_HouseScore"] = train["NEW_KitchenScore"] + train["NEW_GarageScore"] + train["NEW_BsmtScore"] + train["NEW_FirePlaceScore"] + train["NEW_ExterScore"] *\
train["Functional"].apply(lambda x: 1.5 if x == "Typ" else (1.25 if x == "Min1" or "Min2" else (1 if x == "Mod" else (0.75 if x == "Maj1" or "Maj2" else 0.5 ))))

##################################################################################
# Feature Engineering (Test)
##################################################################################

# Age
test["NEW_age"] = 2022 - test["YearBuilt"].astype(int)
test.loc[(test["NEW_age"] >= 0) & (test["NEW_age"] < 5), "NEW_status"] = "new"
test.loc[(test["NEW_age"] >= 5) & (test["NEW_age"] < 15), "NEW_status"] = "mid"
test.loc[(test["NEW_age"] >= 15) & (test["NEW_age"] < 25) , "NEW_status"] = "mid-old"
test.loc[test["NEW_age"] >= 25, "NEW_status"] = "old"

# OverAll Skor
test["NEW_OverallSkor"] = test["OverallQual"] * test["OverallCond"]

# Bahçe Alanı
test['TotalPorchSF'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])

# Parsel Büyüklüğü
test.loc[(test["LotArea"] >= 0) & (test["LotArea"] < 5000), "NEW_AreaSize"] = "Small"
test.loc[(test["LotArea"] >= 5000) & (test["LotArea"] < 10000), "NEW_AreaSize"] = "Medium"
test.loc[(test["LotArea"] >= 10000) , "NEW_AreaSize"] = "Big"

# Condition(Metroya yakınlık vs.) Skor
test["New_CondScore"] = test["Condition1"].apply(lambda x: 1 if x != "Norm" else 0 ) + test["Condition2"].apply(lambda x: 1 if x != "Norm" else 0)

# Dış Cephe Skoru
test["NEW_ExterScore"] = test["ExterQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else 1)))) + \
test["ExterCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else 1))))

# Basement Skoru
test["NEW_BsmtScore"] = test["BsmtQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
test["BsmtCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
test["BsmtExposure"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))

# Garage Skoru
test["NEW_GarageYear"] = 2022 - test["GarageYrBlt"]
test["NEW_GarageScore"] = test["GarageQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
test["GarageCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
test["GarageFinish"].apply(lambda x: 3 if x == "Fin" else (2 if x == "RFn" else (1 if x == "Unf" else 0))) + \
test["PavedDrive"].apply(lambda x: 2 if x == "Y" else (1 if x == "P" else 0)) / test["NEW_GarageYear"].astype(int)

# Fireplace Skoru
test["NEW_FirePlaceScore"] = test["FireplaceQu"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))\
*test["Fireplaces"]

# Mutfak Skoru
test["NEW_KitchenScore"] = test["KitchenQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))\
*test["KitchenAbvGr"]

# Banyo Sayısı

test["NEW_NumberOfBath"] = test["BsmtFullBath"].apply(lambda x: 0 if x == "missing" else x) + test["BsmtHalfBath"].apply(lambda x: 0 if x == "missing" else x).astype(int) * 1/2 +test["FullBath"] + test["HalfBath"]*1/2

# Evin Alanı
test["HouseArea"] = test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF']

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler(feature_range= (0,10))
# Evin Skoru
test[["NEW_KitchenScore","NEW_FirePlaceScore","NEW_GarageScore","NEW_BsmtScore","NEW_ExterScore"]] = minmax.fit_transform(test[["NEW_KitchenScore","NEW_FirePlaceScore","NEW_GarageScore","NEW_BsmtScore","NEW_ExterScore"]])

test["NEW_HouseScore"] = test["NEW_KitchenScore"] + test["NEW_GarageScore"] + test["NEW_BsmtScore"] + test["NEW_FirePlaceScore"] + test["NEW_ExterScore"] *\
test["Functional"].apply(lambda x: 1.5 if x == "Typ" else (1.25 if x == "Min1" or "Min2" else (1 if x == "Mod" else (0.75 if x == "Maj1" or "Maj2" else 0.5 ))))


##################################################################################
# Encoding
##################################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(train, cat_th=17, car_th=26)

train = one_hot_encoder(train, cat_cols, drop_first=True)
test = one_hot_encoder(test, cat_cols, drop_first=True)

useless_cols_new = [col for col in cat_cols if
                        (train[col].value_counts() / len(train) <= 0.01).any(axis=None)]

for col in useless_cols_new:
    train.drop(col, axis=1, inplace=True)

for col in useless_cols_new:
    test.drop(col, axis=1, inplace=True)

check_df(train)
check_df(test)
##################################################################################
# Base Models
##################################################################################
# y = np.log1p(train['SalePrice'])
y = train['SalePrice']
X = train.drop(["Id", "SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          #("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

y.mean()
y.std()

# Seçilen modeller: Catboost, GBM, Ridge, LightGBM
##################################################################################
# Hyper Tuning
##################################################################################
catboost_params = {"iterations": [400, 500, 600],
                   "learning_rate": [0.01, 0.1],
                   "depth": [4, 5, 6, 7]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 4, 5],
              "n_estimators": [1600, 1800, 2000],
              "subsample": [0.2, 0.3, 0.4],
              "loss": ['huber'],
              "max_features": ['sqrt']}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [1300, 1500, 1700],
                   "colsample_bytree": [0.2, 0.3, 0.4]}

ridge_params = {"alpha": np.linspace(0, 0.02, 11)}

# 1. Lgbm Model Tuning
lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y,
                                        cv=5,
                                        scoring="neg_mean_squared_error")))

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lightgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

new_y = np.expm1(y_pred)
new_y_test = np.expm1(y_test)

np.sqrt(mean_squared_error(new_y_test, new_y))

##################################################################################
# Submission
##################################################################################

new_predict = pd.DataFrame()
new_predict["Id"] = test["Id"].astype(int)
y_pred_sub = np.expm1(lgbm_tuned.predict(test.drop("Id", axis=1)))
new_predict['SalePrice'] = y_pred_sub
new_predict.to_csv('saleprice.csv', index=False)