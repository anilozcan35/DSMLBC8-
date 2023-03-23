import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("5.Modül/Projects/datasets/Telco-Customer-Churn.csv")
df.head()

df.columns = [col.lower() for col in df.columns]

# nümerik olduğunu düşündüğüm bir kolonu düzenliyorum
df.loc[df["totalcharges"] == " ", "totalcharges"] = 0
df["totalcharges"] = df["totalcharges"].astype("float")

num_cols = [col for col in df.columns if df[col].dtypes != "O"]

# kategorik değişkenleri yakalıyorum.

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
num_but_cat = [col for col in df.columns if df[col].dtypes != "O" and df[col].nunique() < 10]
cat_but_car = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() > 20]
num_cols = [col for col in num_cols if col not in num_but_cat]

cat_cols = num_but_cat + cat_cols
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# cat summary, num summary

df.describe().T

def cat_summary(dataframe, col_name, plot=False):  # DEĞİŞKENİN İSMİ VE İLGİLİ DEĞİŞKENİN SINIFLARININ DAĞILIMI
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        sns.boxplot(x= dataframe[numerical_col], data=df)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col, plot= True)

# aykırı gözlem analizi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):  # ÇEYREKLİKLERE GÖRE TIRAŞLAMAK İÇİN OLUŞTURDUĞUMUZ FONKSİYON
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):  # AYKIRI DEĞER VAR MI YOK MU FONKSİYONU
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)  # LOW VE HIGH LIMITI HESAPLAMAK ICIN OUTLIER THRESHOLDS FONKSYIONUNU KULLANIYORUZ.
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Outlier kontorlü yapıyoruz.
for col in num_cols:
    print(col,":",check_outlier(df, col))

# Outlier yok gibi görünüyor.

def replace_with_thresholds(dataframe, variable):  # SCRİPT BAZINDA OUTLIERLARLARI REPLACE EDEN FONKSIYON
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# missing values
df.isnull().any().sum()

# korelasyon analizi

df.corr().unstack().sort_values(ascending=False).drop_duplicates() # acaba yüzde 80likler drop edilmeli mi

# feature engineering
df.head()

df["tenure"].describe().T

df.drop("customerclass", axis=1, inplace = True)
df.loc[(df["tenure"] >= 0) & (df["tenure"] < 12), "customerclass" ] = "new"
df.loc[(df["tenure"] >= 12) & (df["tenure"] < 24), "customerclass" ] = "mature"
df.loc[(df["tenure"] >= 24), "customerclass" ] = "old"

df["online"] = df["onlinesecurity"].astype(bool) + df["onlinebackup"].astype(bool)
df["online"] = df["online"].astype("int")

df["streaming"] = df["streamingtv"].astype(bool) + df["streamingmovies"].astype(bool)
df["streaming"] = df["streaming"].astype("int")

# Encoding işlemleri
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[
                   col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()


# one hot encode

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_columns = [col for col in df.columns if df[col].dtypes == "O" and 10 > df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_columns, drop_first= True)
df.head()

df["customerclass"].dtypes

# nümerik değişkenler için standartlaştırma yapılıyor.
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

# Modelleme

y = df["churn"]
X = df.drop(["customerid", "churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17) # DATAYI TEST VE TRAİN OLARAK BÖLÜYORUM

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7903 accuracy score