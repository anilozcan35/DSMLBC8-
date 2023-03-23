#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno  # MISSING VALUES BÖLÜMÜNDE KULLANMAK İÇİN
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor  # ÇOK DEĞİŞKENLİ AYKIRI DEĞİŞKENLERİ YAKALAMA YÖNTEMİ
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler  # STANDARTLAŞTIRMA DÖNÜŞTÜRME

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_application_train():  # BÜYÜK DATADA İŞLEMLERİ GÖRMEK İÇİN KULLANICAĞIMIZ DATASET.
    data = pd.read_csv("5.Modül/feature_engineering/datasets/application_train.csv")
    return data


df = load_application_train()
df.head()


def load():
    data = pd.read_csv("5.Modül/feature_engineering/datasets/titanic.csv")
    return data


df = load()
df.head()

#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])  # SAYISAL DEĞİŞKENDEKİ AYKIRI DEĞERLERİ GÖRMEK İÇİN
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################
# ÇEYREK DEĞERLERİNİ HESAPLICAZ Kİ IQR HESAPLAYABİLELİM.
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr  # ALT LİMİT
low = q1 - 1.5 * iqr  # ÜST LİMİT

df[(df["Age"] < low) | (df["Age"] > up)]  # OUTLIERLAR

df[(df["Age"] < low) | (df["Age"] > up)].index  # OUTLIERLARIN INDEXLERI

###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)  # AXİS= NONE HEPSİNE BAK
df[(df["Age"] < low)].any(axis=None)  # 0'DAN KÜÇÜK YAŞ OLMADIĞI İÇİN FALSE DÖNECEK


# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):  # ÇEYREKLİKLERE GÖRE TIRAŞLAMAK İÇİN OLUŞTURDUĞUMUZ FONKSİYON
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()  # GELEN DEĞERLERE GÖRE OUTLIERLARI BULMAK

df[(df["Fare"] < low) | (df["Fare"] > up)].index  # INDEXLERİ YAKALAMAK İÇİN


def check_outlier(dataframe, col_name):  # AYKIRI DEĞER VAR MI YOK MU FONKSİYONU
    low_limit, up_limit = outlier_thresholds(dataframe,
                                             col_name)  # LOW VE HIGH LIMITI HESAPLAMAK ICIN OUTLIER THRESHOLDS FONKSYIONUNU KULLANIYORUZ.
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "Fare")

###################
# grab_col_names
###################

dff = load_application_train()
dff.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):  # SCRİPT BAZINDA DEĞİŞKENLERİ AYIRMAK İÇİN KULLANILAN FONKSYİON
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]  # NUMERİK GÖRÜNEN AMA KATEGORİK FONKSİYONLAR
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]  # ÖLÇÜLEMEZ KATEGORİKLER
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # KATEGORİK DEĞİŞKENLERİN SON HALİ

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


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if
            col not in "PassengerId"]  # PASSANGERID NUMERİK GÖRÜNSE DE NÜMERİK ANLAMDA BİR İFADE ETMEZ.BURASI BİR EXCEPTİON !!!

for col in num_cols:  # OUTLIER VAR MI YOK MU
    print(col, check_outlier(df, col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if
            col not in "SK_ID_CURR"]  # ID DEĞİŞKENLERİ NUMERİK GÖRÜNSE DE NÜMERİK BİR ANLAM TAŞIMAZ.

for col in num_cols:  # AYKIRI DEĞER VAR MI YOK MU
    print(col, check_outlier(dff, col))


###################
# Aykırı Değerlerin Kendilerine Erişmek
###################
# AYKIRI DEĞERLERİN İNDEXİNE ERİŞTİK. ŞİMDİ KENDİLERİNE ERİŞMEK İÇİN YAZILAN FONKSİYON.
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Age")

grab_outliers(df, "Age", True)

age_index = grab_outliers(df, "Age", True)  # İNDEXLERİ TUTMAK İÇİN

outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df, "Fare")  # FARE DEĞİŞKENİNE GÖRE ALT VE ÜST LİMİTLER
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape  # AYKIRILAR SİLİNDİKTEN SONRA NE KALACAK


def remove_outlier(dataframe, col_name):  # DEĞİŞKENDEKİ OUTLİERLARI SİLMEK İÇİN
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:  # OUTLİERLARI SİLMEK İÇİN
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]  # BELİRLİ BİR DEĞERDEN BÜYÜK YA DA KÜÇÜK FARE DEĞİŞKENİNİ GETİR

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up  # BASKILADIĞIMIZ KISIM

df.loc[(df["Fare"] < low), "Fare"] = low


def replace_with_thresholds(dataframe, variable):  # SCRİPT BAZINDA OUTLIERLARLARI REPLACE EDEN FONKSIYON
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:  # AYKIRI DEĞER VAR MI SOR ?
    print(col, check_outlier(df, col))

for col in num_cols:  # BASKILA
    replace_with_thresholds(df, col)

for col in num_cols:  # TEKRAR OUTLIER VAR MI SOR?
    print(col, check_outlier(df, col))

###################
# Recap
###################

df = load()
outlier_thresholds(df, "Age")  # AYKIRI DEĞER SAPTAMA
check_outlier(df, "Age")  # OUTLIER VAR MI YOK MU ?
grab_outliers(df, "Age", index=True)  # BASKILAMA

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")

#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

# 17, 3 # 17 YAŞINDAKİ BİRİNİN 3 KEZ EVLENMİŞ OLMASI OUTLIER OLABİLİR
# 100 TANE DEĞİŞKEN VAR. NASIL 2 BOYUTA İNDİRGEYEBİLİRİM ? PCA İLE

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape
for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (
            df["depth"] > up))].shape  # 25'E 75 OUTLIERLARI BASKILASAYDIK. VERİYE KENDİ ELİMİZLE GÜRÜLTÜ EKLİCEKTİK
# TEK BAŞINA ÇOK OUTLIER VAR PEKI LOF İLE DE ÖYLE Mİ ?

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)  # LOCAL OUTLIER FACTORU DATASETE UYGULUYORUM.

df_scores = clf.negative_outlier_factor_  # SKORLARI TUTTURAN BÖLÜM BURASI
df_scores[0:5]  # SKORLAR EKSİ GELDİ BUNLARI POZİTİF ALMAK İÇİN AŞAĞIDAKİ GİBİ YAPILIR
# df_scores = -df_scores
np.sort(df_scores)[0:5]  # -1'E YAKIN OLMASI INLIER -10'A YAKIN OLMASI OUTLIER OLMASI ANLAMINA GELEBİLİR

scores = pd.DataFrame(np.sort(df_scores))  # EŞİK DEĞERLERE GÖRE BİR GRAFİK OLUŞTURUYORUM
scores.plot(stacked=True, xlim=[0, 50], style='.-')  # EN SON SERT DEĞİŞİKLİK YAŞANMIŞ NOKTA -5
plt.show()

th = np.sort(df_scores)[3]  # BU GÖZLEMDEN DAHA KÜÇÜK SKORA SAHİP OLANLARIN OUTLIER OLMASINA KARAR VERİYORUZ

df[df_scores < th]

df[df_scores < th].shape  # LOF İLE OUTLIER SAYISI BİNLERCEDEN 3 TANEYE DÜŞTÜ

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T  # ÇOK DEĞİŞKENLİ ETKİYE BURDAN BAKABİLİRSİN.

df[df_scores < th].index  # OUTLIERLARIN YAKALANMASI

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)  # GÖZLEMLERİ SİLİYORUZ.

# EĞER GÖZLEM SAYISI BİR MİKTAR FAZLAYSA BASKILAMADIK. GÖZLEM SAYISI AZSA LOF'TAN SONRA SİLİNİR.
# EĞER AĞAÇ YÖNTEMLERİYLE ÇALIŞIYORSAK BUNLARA HİÇ DOKUNMAMAYI TERCİH EDERİZ.

#############################################
# Missing Values (Eksik Değerler)
#############################################

# MİSSİNG VALUESLER'IN RASTSAL OLARAK OLUŞMASI GEREKİR SİLMEK İÇİN.
# BİR EKSİKLİĞİN SEBEBİ BAŞKA BİR DEĞİŞKEN İSE. SİLMEK DOĞRU OLMAYACAKTIR.

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()  # DOLU OLAN SAYILAR

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)  # EKSİKLİĞİN VERİSETİNDEKİ ORANA GÖRE

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]  # EKSİK DEĞERE SAHİP OLAN DEĞİŞKENLER


def missing_values_table(dataframe, na_name=False):  # EKSİK DEĞERLERİ ORTAYA ÇIKARTAN FONKSYİON
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)

#############################################
# Eksik Değer Problemini Çözme
#############################################
# AĞAÇLAR YÖNTEMLERİ KULLANILIYORSA. MİSSİNG VALUESLER VE OUTLIERSLAR'IN ETKISI YOK'A YAKINDIR.
# EĞER REGRESYON PROBLEMİYSE BAĞIMLI DEĞİŞKEN DE SAYISALSA. ORADA AYKIRILIK VARSA BAĞIMLI DEĞİŞKENİN BULUNMA SÜRESİ UZAYABİLİR.
# DOĞRUSAL VE GRADIENT DECENT TEMELLİ YÖNTEMLERDE BU TEKNİKLER ÇOK DAHA HASSAS.

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape  # BİR GÖZLEM BİLE VARSA SİL

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

df["Age"].fillna(df["Age"].mean()).isnull().sum()  # ORTALAMAYI ATAMAK
df["Age"].fillna(df["Age"].median()).isnull().sum()  # MEDYANI ATAMAK
df["Age"].fillna(0).isnull().sum()  # SABİT İLE DEĞİŞTİRMEK

# df.apply(lambda x: x.fillna(x.mean()), axis=0) # SİSTEMATİK OLARAK ATAMAK İÇİN

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()  # OBJECT DEĞİLSE ORTALAMAYLA DOLDUR.

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)  # SAYISAL DEĞİŞKENDEKİ NULL DEĞERLERDEN KURTULDUK.

df["Embarked"].fillna(
    df["Embarked"].mode()[0]).isnull().sum()  # KATEGORİK DEĞİŞKENLERDEKİ EN MANTIKLI DOLDURMA MOD İLE DOLDURMAKTIR.

df["Embarked"].fillna("missing")  # KATEGORİKTEKİ EKSİĞE MİSSİNG OLARAK DA ATAYABİLİRİZ.

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,
         axis=0).isnull().sum()  # KATEGORİK DEĞİŞKENİ MODU İLE FİLL ETMEK.
# KARDİNAL DEĞİŞKENLERİ DE ARADAN AYIKLIYORUZ.

###################
# Kategorik Değişken Kırılımında Değer Atama
###################
# SAYISAL DEĞİŞKENLERE KATEGORİK DEĞİŞKEN KIRILIMINDA DEĞER ATAMAK.

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform(
    "mean")).isnull().sum()  # KATEGORİK DEĞİŞKEN KIRILIMINDA SAYISAL DOLDURMA.!!!!!!! TRANSFORM!!!!!!

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()
# ONE HOT DÖNÜŞÜMÜ YAPILMALI
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols],
                     drop_first=True)  # ONE HOT DÖNÜŞÜMÜNÜ YANİ MALE 0 FEMALE 1 # CATCOLS + NUMCOLS VERSEM DAHİ SADECE KATEGORİKLERE İŞLEM YAPAR GET_DUMMIES

dff.head()

# değişkenlerin standartlatırılması GEREKLİ
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması.
from sklearn.impute import \
    KNNImputer  # TAHMİNE DAYALI BİR ŞEKİLDE EKSİKLERİ DOLDURUCAZ. KNN BANA ARKADAŞINI SÖYLE SANA KİM OLDUĞUNU SÖLİM.

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff),
                   columns=dff.columns)  # DEĞERLERİ ESKİ DEĞERLERİNE ÇEVİRMEK İÇİN STARNDARLAŞTIRMAYI GERİ ALIYORUM.

df["age_imputed_knn"] = dff[["Age"]]  # YENİ KOLONU ESKİ KOLONA ATIYORUM.

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]  # NEREYE NE ATAMIŞIZ
df.loc[df["Age"].isnull()]

###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma


#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df)  # TAM OLAN GÖZLEMLERİN SAYISINI VERİYOR
plt.show()

msno.matrix(df)  # SİYAH BEYAZ YAPILI MATRİX BİRİ BOŞSA DİĞERİ DE BOŞ MU. GÖRSELLEŞTİRME
plt.show()

msno.heatmap(df)  # EKSİKLİKLER ÜZERİNE BİR HEATMAP
plt.show()  # NULITY KORELASYONLAR ANLAMLI GÖRÜNMÜYOR.

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)  # IS NULL ISE 1 DEĞİLSE 0 YAZ

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)  # 3 DEĞİŞKENİN NULL OLUP OLMAMA DURUMLARINDA SURVIVED UZERINDEKİ ORTALAMARI
# KABİN DEĞİŞKENİNDEKİ FARK ANLAMLI OLABİLİR(Kİ ANLAMLI)


###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)

#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################
# YENİDEN KODLAMAK MALE --> 1 FEMALE --> 0
#############################################
# Label Encoding & Binary Encoding
#############################################

# ORDINAL KATEGORİK DEĞİŞKENLER LABEL ENCODER'DAN GEÇİRİLEBİLİR.
# KATEGORİK DEĞİŞKENLER LABEL ENCODER'DAN GEÇİRİLMEMELİDİR.
# BINARY ENCODIN -- > 1 , 0

df = load()
df.head()
df["Sex"].head()

# NEDEN ENCODE EDİYORUZUN CEVABI MAKİNE ÖĞRENME ALGORİTMALARINA SOKMAK İÇİN ALGORİTMALARIN BİZDEN BEKLEDİĞİ STANDART FORMATA UYGUN HALE GETİRMEK.

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]  # ENCODER OBJESİNİ SEX DEĞİŞKENİNE ÖNCE FİT ET SONRA TRANFORM.
le.inverse_transform([0, 1])  # HANGİSİNE 0 HANGİSİNE 1 VERDİĞİMİZİ UNUTTUYSAK DATASETİ ESKİ HALİNE GETİRMEK İÇİN


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()

# ELİMDE YÜZLERCE DEĞİŞKEN VARSA

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]  # EŞSİZ SINIF SAYISI 2 OLANLARI SEÇİCEZ. SİSTEMATİK OLARAK / NUNIQUE EKSİK DĞEERİ SINIF OLARAK GÖRMEZ

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]  # INT32 INT64 VSS
               and df[col].nunique() == 2]  # 122 DEĞİŞKENDEN BINARY OLAN KOLONLARI ALMAK İÇİN / KATEGORİ SINIF SAYISI 2

df[binary_cols].head()

df.head()
df["FLAG_DOCUMENT_10"][0].dtypes

for col in binary_cols:
    label_encoder(df,
                  col)  # EKSIK DEĞERLERİ DE DOLDURMUŞ. EKSİK DEĞERLERE 2 DEĞERİNİ VERDİĞİNİN BİLİNCİNDE SAHİP OLMAK LAZIM.

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()  # DEĞİŞKEN VAR
len(df["Embarked"].unique())  # BURDA NA DEĞERLER DE GELDİ VE UNIQUE OLARAK GELDİ.

#############################################
# One-Hot Encoding
#############################################

df = load()
df.head()
df["Embarked"].value_counts()  # S VE C VE Q ARASINDA SINIFLAR ARASI FARK YOK. -NOMINAL-

pd.get_dummies(df, columns=["Embarked"]).head()

pd.get_dummies(df, columns=["Embarked"],
               drop_first=True).head()  # DUMMY DEĞİŞKEN TUZAĞINA DÜŞMEMEK İÇİN EMBARKEDC DÜŞÜYOR

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()  # EKSİK DEĞERLER İÇİN SINIF OLUŞTURUR.

pd.get_dummies(df, columns=["Sex", "Embarked"],
               drop_first=True).head()  # 2 SINIFLI KATEGORİK DEĞİŞKENLERDE BİNARY ENCODE EDİLİR. LABEL ENCODERA GEREK KALMADAN.


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]  # ONE HOT ENCODE YAPACAĞIM KOLONLAR SEX VE BAĞIMLI DEĞİŞKEN SURVIVED'I DÖNÜŞTÜRMEDİM.

one_hot_encoder(df, ohe_cols).head()

df.head()

#############################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()  # ACADEMIC DEGREEYI ENCODE ETMEK

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):  # DEĞİŞKENİN İSMİ VE İLGİLİ DEĞİŞKENİN SINIFLARININ DAĞILIMI
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# VERİ SETİNDE ONLARCA KATEGORİK DEĞİŞKEN VAR. ONE HOTTAN GEÇİRİCEZ AMA GEREKLİ GEREKSİZ BİR ÇOK DEĞİŞKEN OLUŞACAK.
# GÖZLEM SAYISI ÇOK AZ OLAN GÖZLEMLERİN ENCODE EDİLİP EDİLMEMESİ.

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()  # SONDAKİ 4'LÜ RARE DURUMDA

df.groupby("NAME_INCOME_TYPE")[
    "TARGET"].mean()  # BUSINESS İLE UNEMPLOYED BIRBIRINDEN FAKRLI TARGET AÇISINDAN DA BENZER OLMALARINI BEKLERDİK.


# BAĞIMLI VE KATEGORİK DEĞİŞKENİ GÖNDERDİĞİMİZDE DEĞİŞKENİN SINIFLARINI SAYISINI, ORANINI VE TARGET İLE ORTALAMSINI VERİR
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", cat_cols)


#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]  # RARE COLUMNS

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)  # BU ORANIN ALTINDA KALAN KATEGORİK DEĞİŞKEN SINIFLARINI BİR ARAYA GETİRİCEK

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################


###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################
# -1 ile +1 ARASINDA DEĞERLER ALICAKTI.
df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

# NET Bİ LİTERATÜR YOKTUR. PROBLEMDEN PROBLEME DEĞİŞİR. VAR OLAN DEĞİŞKEN ÜZERİNDEN YENİ DEĞİŞKEN TÜRETMEK.

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')  # DOLU DEĞERLERE 1 BOŞ DEĞERLERE 0

df.groupby("NEW_CABIN_BOOL").agg(
    {"Survived": "mean"})  # KABİN NUMARASI OLANLARIN HAYATTA KALMA ORANI OLMAYANLARDAN YÜKSEK ÇIKTI
# PEKİ BU FARK ANLAMLI BİR FARKLILIK MI

from statsmodels.stats.proportion import proportions_ztest  # ORAN TESTİ

# BAŞARI SAYISI / GÖZLEM SAYISI

# HO: P0 = P1
# H1 P0 != P1
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# PVALUE DEĞERİ 0.05 TEN KÜÇÜK H0 RED EDİLİR. YANİ ARALARINDA ANLAMLI BİR FARK VARDIR.
# ANCAK !!!!! ÇOK DEĞİŞKENLİ ETKİYİ BİLMİYORUZ. YİNE DE BU DEĞİŞKEN ANLAMLI OLABİLİR DİYEREK DEVAM EDİYORUZ.

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"  # TEK Mİ DEĞİŞKENİ ÜRETİYORUZ
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})  # GÖRÜNEN O Kİ YALNIZ OLANLARIN HAYATTA KALMA ORANI DAHA KÜÇÜK.

# ORAN TESTİ YAPIYORUZ.
# HO: P0 = P1
# H1: P0 != P1
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # H0 RED EDİLİR. YANİ ANLAMLI BİR DEĞİŞİKLİK VAR.

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()  # İSİMLERDEKİ HARF SAYISINI ÇEKMEK İÇİN

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))  # KELİMELERİ SAYDIRMAK İÇİN

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(
    lambda x: len([x for x in x.split() if x.startswith("Dr")]))  # ADININ İÇİNDE DOKTOR GEÇEN

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})  # VE DOKTORLARIN HAYATTA KALMA OLASILIĞI

###################
# Regex ile Değişken Türetmek
###################

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # ÖNÜNDE BOŞLUK SONUNDA NOKTA OLAN PATTERNİ ÇIKART

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]}) # ÜNVAN DEĞİŞKENİNİ ÜRETTİK. BAĞIMLI VE YAŞ DEĞİŞKENLERİNE BAKIYORUZ

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("5.Modül/feature_engineering/datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d") # OBJECT TO DATE DEĞİŞİMİ

# year
dff['year'] = dff['Timestamp'].dt.year # YIL DEĞİŞKENİ TÜRETİYORUM

# month
dff['month'] = dff['Timestamp'].dt.month # AY DEĞİŞKENİ TÜRETİYORUM

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year # YILLARIN FARKINI

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

# day name
dff['day_name'] = dff['Timestamp'].dt.day_name() # İLGİLİ GÜNLERİN İSİMLERİNİ ÇEKEBİLİRİM.

dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
df = load()
df.head()

# YAPILAN İNTERACTİON BİR ŞEY İFADE ETMELİ

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"] # YAŞ İLE SINIFIN ÇARPILMASI REFAH DÜZEYİNDE BİR KARŞILIK DEĞER ÜRETİR.

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 #df AİLE BOYUTU ADINDA YENİ BİR DEĞİŞKEN OLUŞTURMUŞ OLURUZ.

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale' # CİNSİYETİ GENÇ OLAN ERKEKLERE BİR DEĞER ATMAA

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean() # YENİ BİR KATEGORİK DEĞİŞKEN OLUŞTURARAK. KADINLAR İLE ERKEKLER ARASINDAKİ FARKI BİR TIK DAHA HASSASLAŞTIRDIK.

#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.groupby("Sex").agg({"Survived": "mean"})

df.columns = [col.upper() for col in df.columns] # DEĞİŞKEN İSİMLERİNİ BÜYÜLT - TEK TİP HALİNE GETİRDİM.

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################
#OLUŞTURDUĞUMUZ DEĞİŞKENLER

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int') # KABİN DEĞERİ NAN MI DOLU MU
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col] # BU NUMERIK BIR KOLON DEĞİL ANLAM İFADE ETMİYOR

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df, na_name= True) # MISSING VALUESLERI VEREN FONKSYION

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"] # NAME ÜZERİNDEN ÜNVANLARI OLUŞTURDUK
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median")) # AGE İLE OLUŞTURULMUŞ DEĞİŞKENLER DE BOŞ GELİYORDU ONLARI DA DÜZENLİCEZ

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0) # KATEGORİK DEĞİŞKENLERİ MODE'U İLE DOLDURUYORUM.

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2] # 2 SINIFI OLAN KATEGORİK DEĞİŞKENLERİ DEĞİŞTİRİYORUM.

df.head()

for col in binary_cols:
    df = label_encoder(df, col)

#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols) # FREKANSLARINA VE ORANLARINA TARGET_MEAN'INI GETİREN ORTALAMA

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2] # 2 DEN FAZLA UNİQUE 10DAN AZ UNİQUE SINIFI OLAN KATEGORİK DEĞİŞKENLERİ DEÖNÜŞTÜRÜYORUZ.

df = one_hot_encoder(df, ohe_cols, drop_first=True) # DROP FİRST ÖN TANIMLI DEĞERİ YANİ ONE HOT ENCODİNG DUMMT TUZAĞINA DÜŞMEDİK.

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)] # OLUŞTURDUĞUMUZ ONE HOT DÖNÜŞÜMÜNDE ARTIK RARE OLMUŞ VAR MI
                                                                            # MESELA FAMİLY_SİZE_11 1 DEĞERİ 800 0 DEĞERİ 7 İDİ.

# df.drop(useless_cols, axis=1, inplace=True) # ONE HOTTAN SONRA RARE OLANLARI SİLMEK İÇİN

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler() # STANDARTLAŞTIRMAK İÇİN
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17) # DATAYI TEST VE TRAİN OLARAK BÖLÜYORUM

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) # TAHMİNLERİMİZİN SKORU

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# BAZI AĞAÇ YÖNTEMLERİNDE MİSSİNG VALUELERİ DROP ETMEDEN DE KULLANABİLECEĞİZ.

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
