#####################################################
# Demand Forecasting
#####################################################

# Store Item Demand Forecasting Challenge
# https://www.kaggle.com/c/demand-forecasting-kernels-only
# !pip install lightgbm
# conda install lightgbm

# FARKLI MAĞAZALARDAKİ FARKLI ÜRÜNLERİN 3 AYLIK SATIŞ TAHMİNİ
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

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

train = pd.read_csv('7.Modül/time_series/datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('7.Modül/time_series/datasets/demand_forecasting/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv('7.Modül/time_series/datasets/demand_forecasting/sample_submission.csv') # KAGGLEIN ÜSTEDİĞİ FORMAT SUBMISSON

df = pd.concat([train, test], sort=False)


#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max() # MIN MAX TARIH

check_df(df)

df[["store"]].nunique() # 10 MAĞAZA

df[["item"]].nunique() # 50 ÜRÜN

df.groupby(["store"])["item"].nunique() # MAĞAZA ÜRÜN KIRILIMINDA EŞSİZ ÜRÜN SAYISI

df.groupby(["store", "item"]).agg({"sales": ["sum"]}) # HANGİ MAĞAZADA HANGİ ÜRÜNDEN NE KADAR SATILMIŞ

df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]}) # MAĞAZA ÜRÜN KIRILIMINDA SATIŞ İSTATİSTİKLERİ

df.head()




#####################################################
# FEATURE ENGINEERING
#####################################################
# MAKİNE ÖĞRENMESİ KAPSAMINDA TREND, MEVSİMSELLİK LEVEL GİBİ DEĞİŞKENLERİ BİR ŞEKİLDE BURADA DEĞİŞKEN OLARAK ÜRETMEMİZ LAZIM.
df.head()

def create_date_features(df): # DATE İÇERİSİNDEN DEĞİŞİK DEĞİŞKENLER ÜRETECEK.
    df['month'] = df.date.dt.month # AY DEĞİŞKENİ ALMAK İÇİN
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]}) # MAĞAZA ÜRÜN AY KIRILIMINDA BETİMSEL İSTATİSİTKSEL



########################
# Random Noise
########################
# LAG FEATURELAR SALES ÜZERİNDEN ÜRETİLECEK OVERFİTTİNGİ ENGELLEMEK İÇİN GÜRÜLTÜ EKLİYORUZ.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag/Shifted Features
########################
# GECİKME FEATURELARI GEÇMİŞ DÖNEM SATIŞLARINA İLİŞKİN FEATURELAR ÜRETMEK
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True) # MAĞAZA ÜRÜN VE TARİHE GÖRE SIRALIYORUZ

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10], # 1 GECİKMEYİ AL
              "lag2": df["sales"].shift(2).values[0:10], # 2 GECİKMEYİ AL
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})
# LAGLAMININ MANTIĞI OLUŞMASI ESNASINDA EN ÖNEMLİ DEĞERİN ONDAN ÖNCE OLAN DEĞER OLMASINDAN KAYNAKLIYOR. LAGLANAN DEĞİŞKENLERİ GÜRÜLTÜ EKLEYEREK BİRAZ BOZUCAK Kİ OVERFİT OLMASIN.

df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1)) # YUKARIDAKİ KODU LAGLIYORUZ.

def lag_features(dataframe, lags): # LAGLAMAK İÇİN KULLANILACAK DEĞİŞKEN
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]) # BU KADAR GÜNLÜK LAGLARI DEĞİŞKEN OLARAK EKLEYECEĞİZ.
# TAHMİN EDECEĞİMİZ ZAMANA İDLİMİ 3 AY PERİYODUNDA OLDUĞU İÇİN DEĞİŞKENLER DE 3 AY BANDINDA OLUŞTURULDU
check_df(df)
df
########################
# Rolling Mean Features
########################
# HAREKETLİ ORTALAMAYI DEĞİŞKEN OLARAK OLUŞTUAMYA ÇALIŞICAZ
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],# ROLLING METHODU HAREKETLİ ORTALAMA İÇİN KULLANILIR
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})
# BURDA YARINNIN HAREKETLİ ORTALAMASI İÇİN BUGÜN VE YARININ ORTALMAASI ALINIYOR AMA BU SORUNLU BİR DURUM. 1 TANE LAG ATARSAK BUGÜN VE DÜN ÜZERİNDEN AĞIRLIK ORTALAMA ALIP YARINI TAHMİN EDERİZ.

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})
 # LAG ATARAK HAREKETLİ ORTALAMA ALIRSAK EĞER İŞLEM ANLAMLI OLACAKTIR.

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe



df = roll_mean_features(df, [365, 546]) # 1 YIL VE 1.5 YILLIK AĞIRLIKLI ORTALAMASI

########################
# Exponentially Weighted Mean Features
########################
# ÜSTEL AĞIRLIKLI ORTALAMAYI TÜRETİCEZ
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10], # ALFA DEĞERİ 99'KEN EN YAKIN DEĞERE YAKIN OLACAK ÇIKAN SONUÇ
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

def ewm_features(dataframe, alphas, lags): # GECİKME VE ALFA DEĞERLERİNE GÖRE AĞIRLIKLI ÜSTEL ORTALAMAYI
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5] #ALFALAR
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728] # GECİKME SETİ

df = ewm_features(df, alphas, lags)
check_df(df)
df.head()
########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month']) # VAR OLAN KATEGORİK DEĞİŞKENLERİ ONE HOT ENCODERDAN GEÇİRİCEZ


check_df(df)


########################
# Converting sales to log(1+sales)
########################
# ZAMAN SERİSİ AĞAÇ VE REGRESYONSA TRAİN SÜRESİNİ AZALTMAK İÇİN STANDARDİZE ETMEK İÇİN LOGARİTMASI ALINABİLİR.
df['sales'] = np.log1p(df["sales"].values) # LOG1P 1 VARSA HATA ALMAMAK İÇİN

check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE) TAHMİN EDİLEN DEĞERLERLE GERÇEK DEĞERLER ARASINDAKİ FARKA DAİR HATA METRİĞİ NE KADAR KÜÇÜK O KADAR İYİ.

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




########################
# Time-Based Validation Sets
########################

train
test

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :] # KAGGLEIN BENDEN ISTEDIĞINA EN YAKIN SENARYOYU VALIDAYSON SETİ YAPIYORUM.
val.head()
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]] # BAĞIMSIZ DEĞİŞKENLERDE DATE DEĞİŞKENLERİ İFADE EDEN DEĞİŞKENLERİ ÇIKARTTIK ZATEN

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
# VALIDASYON SETINI BIZ KENDIMIZ OLUŞTURDUK KAGGLE TEST SETİNE BENZEMESİ AÇISINDAN

########################
# LightGBM ile Zaman Serisi Modeli
########################

# !pip install lightgbm
# conda install lightgbm


# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000, # iterasyon sayısı
              'early_stopping_rounds': 200, # AŞIRI ÖĞRENMENİN ÖNÜNE GEÇMEK İÇİN
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

# BURDAKI LIGHTGBM SKITLEARN İÇERİSİNDEKİ LIGHTGBM DEĞİL
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols) # lightgbm apisi

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape, # HATA FONKSIYONUNU CUSTOM OLARAK VERDİK.
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration) # BAĞIMSIZ DEĞİŞKENLERE SORUP SALES DEĞİŞKENLERİNİ TAHMİN EDİYORUZ
# PREDICTION'U BEST ITERASYON İLE YAP DİYOR OLABİLİR Mİ ?

smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True) # DEĞİŞKENLERİN ÖNEMİ BURDAKİ GAIN FEATURE KULLANILDIĞINDA HATADAKİ DÜŞÜŞÜ TEMSİL EDER.SPLIT DE KAÇ KERE BÖLME İŞLEMİNDE KULLANILMIŞ


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################
# TİMES BASED VALİDATİONDA HESABA KATILMAYAN DATALAR KULLANILARAK MODEL TEKRAR KURULACAK.

train = df.loc[~df.sales.isna()] # DOLU GÖZLEMLER
Y_train = train['sales'] # BAĞIMLI DEĞİŞKEN
X_train = train[cols] # BAĞIMSIZ DEĞİŞKENLER


test = df.loc[df.sales.isna()] # SALES NA OLANLAR TEST DATASI
X_test = test[cols] #

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration} # EARLY STOPPING ROUNS GİRİLMİYOR EN İYİ İTERASYONU BİLİYORUZ ÇÜNKÜ

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols) # BAĞIMLI DEĞİŞKNELERİN ADINI AYRICA BELİRTMEK GEREKİR

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration) # MODEL NESNESİNDEKİ EN İYİ İTERASYON SAYISINI BURAYA SET EDİCEZ

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration) # TAHMİN ETTİĞİMİZ KISIM MODELDEKİ İTERASYON SAYISINI EK OLARAK GİRDİK. NEDEN ANLAMADIM??

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]] # TEST DF TEN ALMAMIZ GEREKN DEĞİŞKENLERİ ALDIK
submission_df['sales'] = np.expm1(test_preds) # LOGARİTMASI ALINMIŞTI TERSİNİ ALICAZ

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)



