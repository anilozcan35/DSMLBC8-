import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler
import warnings
warnings.simplefilter(action="ignore")


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

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 50)

df_ = pd.read_csv("6.Modül/Projeler/hitters.csv")
df = df_.copy()

df.head()
df.shape
df.info()
df.isnull().sum().sum()

liste = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

df.describe(liste).T

df.loc[df["NewLeague"] != df["League"]]

############### Preprocessing ###################
###### Missing Values

def missing_values_table(dataframe, na_name=False):  # EKSİK DEĞERLERİ ORTAYA ÇIKARTAN FONKSYİON
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# KNN ile missing valuesleri dolduruyoruz.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
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
                   columns=dff.columns)
df["Salary_tahmin"] = dff["Salary"]

df.loc[:, ["Salary", "Salary_tahmin"]]
# df.loc[df["Salary"].isnull(), ["Salary", "Salary_tahmin"]]
# df.loc[df["Salary"].isnull(), ["Salary"]] = df.loc[df["Salary"].isnull(), ["Salary_tahmin"]]
# df.loc[df["Salary"].isnull()]
df.drop(columns="Salary", axis= 1, inplace=True)
df.head()


####### Outliers

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df,col)) # No outlier.

df.head()

# Local Outlier Factor
df.columns = [col.lower() for col in df.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(df)

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])  # LOCAL OUTLIER FACTORU DATASETE UYGULUYORUM.

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

# df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)  # GÖZLEMLERİ SİLİYORUZ.

######## Keşifçi veri analizi
# Kategorik değişken analizi
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

for col in cat_cols:
    cat_summary(df,col)

# Sayısal değişken analizi
import matplotlib.pyplot as plt
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

# Hedef değişken analizi
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.columns = [col.lower() for col in df.columns]

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "salary_tahmin", col)



# Korelasyon analizi
df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Yüksek korelasyonlu kolonların silinmesi # Bu işlemi feature engineering sonrası yapacağım.
def high_correlated_cols(dataframe, plot=False, corr_th=0.75):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

######## Feature Engineering

df.head()
df.columns = [col.lower() for col in df.columns]

# 1. yıllık total
df["absolute_yield"] = df["walks"] / (df["errors"] + df["walks"])

# 2. Bu yıl yapılan değerli vuruş sayısı oranı
df["prop_hmrun"] = df["hmrun"] + 1 / (df["chmrun"] + 1)

# 3. net skor katkısı
df["net_score_contribution"] = df["assists"] + df["runs"]

# 4. Kariyerine oranla bu sezon kazandırdığı skor
df["score_totalscore_prop"] = df["runs"] / df["cruns"]

# 5. faydaların kariyerindeki totalliklere oranı
df["yield_totalyields_years"] = df.loc[:, df.columns.str.startswith("c")].sum(axis=1) / (df.loc[:, [col[1:] for col in df.columns[df.columns.str.startswith("c")]]].sum(axis = 1) / df["years"])

# 6. toplam hatalr metriği
df["total_error_metric"] = df["walks"] - df["errors"]

# 7. skorerlik
df["scorer"] = pd.qcut(df["runs"], 3, labels=["scorer", "normal","non-scorer"])


# Yüksek korelasyoynlu kolonların temizlenmesi.
high_correlated_cols(df, plot=True)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1, inplace = True)

############# Encoder
# Label Encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].nunique() == 2 ]

for col in binary_cols:
    df = label_encoder(df, col)

# df.years.value_counts()

# one-hot encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ["scorer"], drop_first=True)

# Standart
cat_cols, num_cols, cat_but_car = grab_col_names(df)
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
# df[num_cols] = ss.inverse_transform(df[num_cols])
df.head()

########## Model

y = df["salary_tahmin"]
X = df.drop("salary_tahmin", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cat_cols, num_cols, cat_but_car = grab_col_names(X_train)
ss = StandardScaler()
X_train[num_cols] = ss.fit_transform(X_train[num_cols])
# X_train[num_cols] = ss.inverse_transform(X_train[num_cols])
X_train.head()

ss = StandardScaler()
y_train = ss.fit_transform(pd.DataFrame(y_train))
y_train = ss.inverse_transform(y_train)
y_train = pd.DataFrame(y_train)
y_train.head()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

X_train.isnull().sum()
reg_model = LinearRegression().fit(X_train, y_train)

# train hata oranı
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_pred, y_train))


reg_model.score(X_train, y_train)

# test hata oranı
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred, y_test))
reg_model.score(X_test, y_test)

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))