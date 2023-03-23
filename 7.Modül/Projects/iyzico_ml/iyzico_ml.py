import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

df_ = pd.read_csv("7.Modül/Projects/iyzico_ml/iyzico_data.csv", index_col= "index")
df = df_.copy()
df.head()
df.info()

# transaction date'i datetime'a çeviriyoruz.
df["transaction_date"] = df["transaction_date"].apply(pd.to_datetime)

# Adım2:  Veri setinin başlangıc ve bitiş tarihleri nedir?
df["transaction_date"].max()
df["transaction_date"].min()

# Adım3: Her üye iş yerindeki toplam işlem sayısı kaçtır ?
df.groupby("merchant_id").agg("Total_Transaction").sum()

# Adım4: Her üye iş yerindeki toplam ödeme miktarı kaçtır?
df.groupby("merchant_id").agg("Total_Paid").sum()

# Adım5: Her üye iş yerininin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.
df.groupby("merchant_id").agg("Total_Transaction").sum().plot()
plt.show()

# Feature Engineering

def create_date_features(df): # DATE İÇERİSİNDEN DEĞİŞİK DEĞİŞKENLER ÜRETECEK.
    df['month'] = df.transaction_date.dt.month # AY DEĞİŞKENİ ALMAK İÇİN
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df["is_wknd"] = df.transaction_date.dt.weekday // 4
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

########### lag / shifted features
# Random Noise
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# sort_values
df.sort_values(["merchant_id","transaction_date"], axis= 0, inplace= True)
df

# lag
df.groupby("merchant_id")["Total_Transaction"].head()

df.groupby("merchant_id")["Total_Transaction"].transform(lambda x: x.shift(1))

def lag_features(lags, dataframe):
    for lag in lags:
        dataframe["lag_" + str(lag)] = dataframe.groupby("merchant_id")["Total_Transaction"].transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

lag_list = [91, 98, 182, 364, 728]
df = lag_features(lag_list, df ) # BU KADAR GÜNLÜK LAGLARI DEĞİŞKEN OLARAK EKLEYECEĞİZ.
df

# rolling mean / haraketli ortalama

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")["Total_Transaction"]. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [180 , 365])
df

# exponantially/
def ewm_features(dataframe, alphas, lags): # GECİKME VE ALFA DEĞERLERİNE GÖRE AĞIRLIKLI ÜSTEL ORTALAMAYI
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")["Total_Transaction"].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alpha_list = [ 0.7 , 0.75, 0.8, 0.85, 0.9, 0.95]

df = ewm_features(df, alpha_list,lag_list)
df

# one hot encoding
df = pd.get_dummies(df, columns=['merchant_id', 'day_of_week', 'month'])
df

# custom cost functions

def smape(preds, target): # SMAPE İN FONKSİYON HALİ
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data): #
    labels = train_data.get_label() # ZAMAN SERİSİ İÇİNDEKİ BAĞIMLI DEĞİŞKEN GERÇEK DEĞİŞKEN YANİ
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# veri setini train ve validation olarak bölelim

train = df[:6500]
test = df[6500:]

cols = [col for col in train.columns if col not in ["transaction_date", "year", "Total_Transaction"]]

Y_train = train["Total_Transaction"]
X_train = train[cols]

Y_val = test["Total_Transaction"]
X_val = test[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# Lightgbm

import lightgbm as lgb

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000, # iterasyon sayısı
              'early_stopping_rounds': 200, # AŞIRI ÖĞRENMENİN ÖNÜNE GEÇMEK İÇİN
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols) # lightgbm apisi

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape, # HATA FONKSIYONUNU CUSTOM OLARAK VERDİK.
                  verbose_eval=100)