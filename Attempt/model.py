import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler  # STANDARTLAŞTIRMA DÖNÜŞTÜRME


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_csv("Attempt/Test.csv - test.csv")
df = df_.copy()

df[df["order_id_new"] != df["order_try_id_new"]].shape
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().any()
df.isnull().sum()

# device token column
df["device_token"].notnull().any()
df.drop(columns = "device_token", axis=1, inplace=True)

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
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

############ Handling with missing values #################
#metered_price
dropindex = df[(df["metered_price"].isnull())].index
df.drop(index=dropindex, axis= 0, inplace= True)

#fraud_score
df["fraud_score"].describe().T

df[df["fraud_score"].isnull()].index
df["fraud_score"] = df["fraud_score"].fillna(df["fraud_score"].mean())

#change_reason_pricing
df["change_reason_pricing"]

df.loc[df["change_reason_pricing"].isnull(), "change_reason_pricing"] = "missing"

# OUTLIERS
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):  # ÇEYREKLİKLERE GÖRE TIRAŞLAMAK İÇİN OLUŞTURDUĞUMUZ FONKSİYON
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):  # AYKIRI DEĞER VAR MI YOK MU FONKSİYONU
    low_limit, up_limit = outlier_thresholds(dataframe,
                                             col_name)  # LOW VE HIGH LIMITI HESAPLAMAK ICIN OUTLIER THRESHOLDS FONKSYIONUNU KULLANIYORUZ.
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):  # SCRİPT BAZINDA OUTLIERLARLARI REPLACE EDEN FONKSIYON
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def grab_col_names(dataframe, cat_th=10, car_th=20): # SCRİPT BAZINDA DEĞİŞKENLERİ AYIRMAK İÇİN KULLANILAN FONKSYİON
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
                   dataframe[col].dtypes != "O"] # NUMERİK GÖRÜNEN AMA KATEGORİK FONKSİYONLAR
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] # ÖLÇÜLEMEZ KATEGORİKLER
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # KATEGORİK DEĞİŞKENLERİN SON HALİ

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] # INTEGER VE FLOAT OLANLAR GELECEK
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:  # AYKIRI DEĞER VAR MI SOR ?
    print(col, check_outlier(df, col))

for col in num_cols:  # BASKILA
    replace_with_thresholds(df, col)

for col in num_cols:  # TEKRAR OUTLIER VAR MI SOR?
    print(col, check_outlier(df, col))

# df["b_state"].nunique() = 1
df.drop(columns = "b_state",axis = 1, inplace= True)
df.drop(columns = "order_try_state",axis = 1, inplace= True)
df.drop(columns= cat_but_car, axis=1, inplace=True)

df["prediction_price_type"].nunique()

# Label encoder and one hot encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64","int32", "float64","float32"] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 20 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

# StandartScaler
scaler = StandardScaler() # STANDARTLAŞTIRMAK İÇİN
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

df = one_hot_encoder(df, ohe_cols).head()
df.dropna(inplace=True)
# Model

y = df["upfront_price"]
X = df.drop(["order_id_new","order_try_id_new", "upfront_price"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17) # DATAYI TEST VE TRAİN OLARAK BÖLÜYORUM

from sklearn.metrics import r2_score
lineer_regresyon = LinearRegression()
model = lineer_regresyon.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_score(y_pred, y_test)