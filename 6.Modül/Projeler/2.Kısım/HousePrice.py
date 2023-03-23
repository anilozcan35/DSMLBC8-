# Kütüphane import işlemleri ve ayarlamalar
import pandas as pd
from eda import *
import missingno as msno
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)


######### Veri setinin yüklenmesi
df_ = pd.read_csv("6.Modül/Projeler/2.Kısım/HousePrice/train.csv")
df = df_.copy()
df.head()

######### Veri ön işleme
df.shape
df.isnull().sum()
df.info()
df.describe([0.01,0.05, 0.10, .25, .50, .75, .90, .95, .99]).T

# 1. Genel Bakış

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int64", "int32", "float32", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]  # INTEGER VE FLOAT OLANLAR GELECEK
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th= 16, car_th= 25)

check_df(df, 5) # Outlierları 5  ve 95 çeyrekliklerine göre baskılayabiliriz.

# 2. Outlier Kontrolü


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# 3. Missing Values
# null barındıran kolonlar
na_columns = missing_values_table(df, na_name= True)

# missing olmanın hedef değişken üzerinde bir anlamı var mı?
missing_vs_target(df, "SalePrice",na_columns)

# kategorik değişkenlere atama.
for col in cat_cols:
    df[col].fillna("missing", inplace= True) # kategorik değişkenlerdeki missing değerler anlam ifade ediyor.

na_columns = missing_values_table(df, na_name= True)


msno.bar(df)  # TAM OLAN GÖZLEMLERİN SAYISINI VERİYOR
plt.show()

msno.matrix(df)  # SİYAH BEYAZ YAPILI MATRİX BİRİ BOŞSA DİĞERİ DE BOŞ MU. GÖRSELLEŞTİRME
plt.show()

msno.heatmap(df)  # EKSİKLİKLER ÜZERİNE BİR HEATMAP
plt.show()  #

# sayısal değişkenlere atama. Mahalle kırılımında yapılacak.
# df.groupby("Neighborhood")["MasVnrArea"].mean()

for col in num_cols:
    df[col].fillna(df.groupby("Neighborhood")[col].transform("mean"), inplace=True)

na_columns = missing_values_table(df, na_name= True) # null değer yok

check_df(df, 5)
########### RARE ANALYSER #########

rare_analyser(df, "SalePrice", cat_cols)

new_df = rare_encoder(df, 0.01, cat_cols)

########### EDA ##########

check_df(new_df)

for col in new_df.columns:
    if new_df[col].max() == 0:
        new_df = new_df.drop(columns = col )

check_df(new_df)

# kategorik değişken analizi
cat_cols, num_cols, cat_but_car = grab_col_names(new_df, cat_th= 16, car_th= 25)

new_df.head()
num_cols = [col for col in num_cols if col != "Id"]

feat_df = new_df.copy()

for col in cat_cols:
    cat_summary(new_df, col, plot= True)

for col in num_cols:
    num_summary(new_df, col, plot= True)

# Hedef değişken analizi
for col in cat_cols:
    target_summary_with_cat(new_df ,"SalePrice", col)

# Korelasyon analizi
high_correlated_cols(new_df, plot= True) # Korele kolon yok

############ Label Encoder ##########
# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in new_df.columns if new_df[col].dtype not in ['int64', "int32","float32", 'float64']
               and new_df[col].nunique() == 2] # 2 SINIFI OLAN KATEGORİK DEĞİŞKENLERİ DEĞİŞTİRİYORUM.

for col in binary_cols:
    new_df = label_encoder(new_df, col)

ohe_cols = [col for col in new_df.columns if 25 >= new_df[col].nunique() > 2]

new_df = one_hot_encoder(new_df, ohe_cols,drop_first= True)

new_df["Neighborhood"].nunique()
new_df.head()

################# Model #################

y = new_df["SalePrice"]
X = new_df.drop(["SalePrice", "Id"], axis=1)

X.head()
X.shape

catboost_model = CatBoostRegressor(random_state=17, verbose=False) # ÇIKTI ÇOK ÇİRKİNMİŞ VERBOSE= FALSE
catboost_model.get_params()

rmse = np.mean(np.sqrt(-cross_val_score(catboost_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

# 25858.752926006426

####### Feature Engineering #######
import datetime
feat_df.head()
feat_df["NEW_age"] = 2022 - feat_df["YearBuilt"]
feat_df.loc[(feat_df["NEW_age"] >= 0) & (feat_df["NEW_age"] < 5), "NEW_status"] = "new"
feat_df.loc[(feat_df["NEW_age"] >= 5) & (feat_df["NEW_age"] < 15), "NEW_status"] = "mid"
feat_df.loc[(feat_df["NEW_age"] >= 15) & (feat_df["NEW_age"] < 25) , "NEW_status"] = "mid-old"
feat_df.loc[feat_df["NEW_age"] >= 25 , "NEW_status"] = "old"

feat_df["LotArea"].describe().T

feat_df.loc[(feat_df["LotArea"] >= 0) & (feat_df["LotArea"] < 5000), "NEW_AreaSize"] = "Small"
feat_df.loc[(feat_df["LotArea"] >= 5000) & (feat_df["LotArea"] < 10000), "NEW_AreaSize"] = "Medium"
feat_df.loc[(feat_df["LotArea"] >= 10000) , "NEW_AreaSize"] = "Big"

feat_df["OpenPorchSF"].describe().T

feat_df["NEW_ExterScore"] = feat_df["ExterQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else 1)))) + \
feat_df["ExterCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else 1))))


feat_df["NEW_BsmtScore"] = feat_df["BsmtQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
feat_df["BsmtCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
feat_df["BsmtExposure"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))

feat_df["NEW_GarageYear"] = 2022 - feat_df["GarageYrBlt"]
feat_df["NEW_GarageScore"] = feat_df["GarageQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
feat_df["GarageCond"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0))))) + \
feat_df["GarageFinish"].apply(lambda x: 3 if x == "Fin" else (2 if x == "RFn" else (1 if x == "Unf" else 0))) + \
feat_df["PavedDrive"].apply(lambda x: 2 if x == "Y" else (1 if x == "P" else 0)) + feat_df["GarageCars"] / feat_df["NEW_GarageYear"]

feat_df["NEW_FirePlaceScore"] = feat_df["FireplaceQu"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))\
*feat_df["Fireplaces"]

feat_df["NEW_KitchenScore"] = feat_df["KitchenQual"].apply(lambda x: 5 if x == "Ex" else (4 if x == "Gd" else (3 if x == "TA" else (2 if x == "Fa" else (1 if x == "Po" else 0)))))

from sklearn.preprocessing import MinMaxScaler, StandardScaler

minmax = MinMaxScaler(feature_range= (0,10))

feat_df[["NEW_KitchenScore","NEW_FirePlaceScore","NEW_GarageScore","NEW_BsmtScore","NEW_ExterScore"]] = minmax.fit_transform(feat_df[["NEW_KitchenScore","NEW_FirePlaceScore","NEW_GarageScore","NEW_BsmtScore","NEW_ExterScore"]])

feat_df["NEW_HouseScore"] = feat_df["NEW_KitchenScore"] + feat_df["NEW_GarageScore"] + feat_df["NEW_BsmtScore"] + feat_df["NEW_FirePlaceScore"] + feat_df["NEW_ExterScore"] *\
feat_df["Functional"].apply(lambda x: 1.5 if x == "Typ" else (1.25 if x == "Min1" or "Min2" else (1 if x == "Mod" else (0.75 if x == "Maj1" or "Maj2" else 0.5 ))))

#### LabelEncoder ####

y = feat_df["SalePrice"]
X = feat_df.drop(["SalePrice","Id"], axis = 1)

binary_cols = [col for col in feat_df.columns if feat_df[col].dtype not in ['int64', "int32","float32", 'float64']
               and feat_df[col].nunique() == 2] # 2 SINIFI OLAN KATEGORİK DEĞİŞKENLERİ DEĞİŞTİRİYORUM.


for col in binary_cols:
    X = label_encoder(feat_df, col)

ohe_cols = [col for col in cat_cols if 25 >= feat_df[col].nunique() > 2]

X = one_hot_encoder(X, ohe_cols,drop_first= True)
X = pd.get_dummies(X, columns=[ "NEW_status", "NEW_AreaSize"], drop_first=False)
X.head()

# cat_cols, num_cols, cat_but_car = grab_col_names(feat_df, cat_th= 16, car_th= 25)

####### Model ##########

from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier

lightgbm = LGBMRegressor()

catboost_model.get_params()

rmse = np.mean(np.sqrt(-cross_val_score(lightgbm,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

# 11710.171993123175
# 12885 aldım 2. seferde ?

lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lightgbm, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lightgbm.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

rmse_final = np.mean(np.sqrt(-cross_val_score(lgbm_final,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

# 11620
# 12910 ???????????