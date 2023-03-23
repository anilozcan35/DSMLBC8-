import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler  # STANDARTLAŞTIRMA DÖNÜŞTÜRME


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("5.Modül/Projects/datasets/diabetes.csv")
df.head()
df.describe().T

# Numerik kolonları yakalıyorum
num_cols = [col for col in df.columns if df[col].dtypes != "O"]

num_cols = [col for col in num_cols if col != "Outcome"]
# kategorik değişkenleri yakalıyorum.
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
# numerik olup kategorik olanlar
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes != "O"]

def cat_summary(dataframe, col_name): # DEĞİŞKENİN İSMİ VE İLGİLİ DEĞİŞKENİN SINIFLARININ DAĞILIMI
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        sns.boxplot(x= dataframe[numerical_col], data=df)
        plt.show(block = True)

import seaborn as sns

for col in num_cols:
    num_summary(df, col, plot= True)

for col in cat_cols:
    cat_summary(df, col)

#Hedef değişken analizi
def num_target_summary(dataframe, cols):
    for col in cols:
        print(dataframe.groupby("Outcome")[col].mean())
        print("###########################")

num_target_summary(df, num_cols)

# Aykırı gözlem analizi

def outlier_thresholds(dataframe, variable): # OUTLIER HESAPLAMAK İÇİN
    quartile1 = dataframe[variable].quantile(0.01) # EŞİKLER NORMALDE 0.25 ÜZERİNDEN YAPILIRDI NORMALDE PROBLEM BAZINDA UCUNDAN TIRAŞLAMAK İÇİN BÖYLE YAPIYORUZ.
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON

# outlier sınırları ile değerleri değiştiren fonksiyon
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):  # AYKIRI DEĞER VAR MI YOK MU FONKSİYONU
    low_limit, up_limit = outlier_thresholds(dataframe,
                                             col_name)  # LOW VE HIGH LIMITI HESAPLAMAK ICIN OUTLIER THRESHOLDS FONKSYIONUNU KULLANIYORUZ.
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# outlier var mı yok mu
for col in num_cols:
    print(col, check_outlier(df, col))

# missing values
df.isnull().sum().any()

# korelasyon analizi
cor_matrix = df.corr()
round(cor_matrix, 2)

# Eksik ve aykırı gözlemlerin düzenlenmesi
import numpy as np

df.columns = [col.upper() for col in df.columns]

df.loc[df["BLOODPRESSURE"] == 0,"BLOODPRESSURE"] = np.NaN
df.loc[df["SKINTHICKNESS"] == 0,"SKINTHICKNESS"] = np.NaN
df.loc[df["GLUCOSE"] == 0,"GLUCOSE"] = np.NaN
df.loc[df["INSULIN"] == 0,"INSULIN"] = np.NaN
df.loc[df["BMI"] == 0,"BMI"] = np.NaN


df.isnull().sum()
# nan olanları ortalama ile dolduruyorum
dff = df.apply(lambda x: x.fillna(x.mean()), axis =0)

# Feature Extraction
df["AGE"].describe().T

dff.loc[(dff["AGE"] > 20) & (dff["AGE"] <= 30) ,"AGE_CAT"] = "young"
dff.loc[(dff["AGE"] >30) & (dff["AGE"] <=50), "AGE_CAT"] = "mature"
dff.loc[(dff["AGE"] >50), "AGE_CAT"] = "elder"

#FATNESS
dff.loc[(dff["BMI"] <= 18.5), "FATNESS"] = "thin"
dff.loc[(dff["BMI"] > 18.5) & (dff["BMI"] <= 24.9), "FATNESS"] = "normal"
dff.loc[(dff["BMI"] > 24.9) & (dff["BMI"] <= 30), "FATNESS"] = "fat"
dff.loc[(dff["BMI"] > 30) ,"FATNESS"] = "obese"


dff.head()
def groupby_with_fatness(dataframe, col):
    return dataframe.groupby("FATNESS")[col].mean()

for col in num_cols:
    print(col, ":", groupby_with_fatness(dff, col.upper()))
    print("##########################################")

#label encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

num_cols = [col for col in dff.columns if dff[col].dtypes != "O"]

num_cols = [col for col in num_cols if col != "OUTCOME"]

bin_columns = [col for col in dff.columns if dff[col].dtypes != "O" and dff[col].nunique() == 2]
cat_cols = [col.upper() for col in dff.columns if dff[col].dtypes == "O"]

#one hot encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

dff = one_hot_encoder(dff, ohe_cols)

# Numerik değerlerin standartlaştırılması
scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])

dff.head()

# Modelleme

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y = dff["OUTCOME"]
X = dff.drop("OUTCOME", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) #

# 0.7532467532467533