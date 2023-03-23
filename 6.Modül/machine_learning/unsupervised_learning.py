################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer # eblow visiualize
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

df = pd.read_csv("6.Modül/machine_learning/datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df) # UZAKLIK TEMELLİ VE GRADİENT DECENT TEMELLİ ALGORİTMALARDA STANDARTLAŞTIRMA YAPILIR.
df[0:5]

kmeans = KMeans(n_clusters=4, random_state=17).fit(df) # K -MEANS MODELİNİ KURUYORUZ
kmeans.get_params() # N-CLUSTERS, MAX_ITERATION,

kmeans.n_clusters
kmeans.cluster_centers_ # CLUSTERLARIN MERKEZLERİ
kmeans.labels_ # KÜME ETİKETLERİ
kmeans.inertia_ # SSE, SSR KARŞILIĞI ( SUM OF SQUARE ERROR )

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_) # HER BİR K SAYISINA GÖRE INERTIA(YANI SSE/SSR/SSD) DEĞERINI SSD LİSTESİNE EKLİYORUZ

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi") # KARAR VERMEK İÇİN DİRSEKLENMENİN OLDUĞU NOKTA SEÇİLMELİ
plt.show()

kmeans = KMeans() # ELBOW YÖNTEMİ İLE OPTİMUM K SAYISI BULUNABİLİR
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_ # K YI KAÇ SEÇMELİYİM ELBOWA GÖRE

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df) # ELBOWDAN GELEN K DEĞERİNİ SET EDİYORUM.

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5] # SONUÇLARI GÖREMİYORUM HANGİ EYALET HANGİ CLUSTERDA BİLEMİYORUM. ÇÜNKÜ ARRAY'E DÖNDÜ

clusters_kmeans = kmeans.labels_

df = pd.read_csv("6.Modül/machine_learning/datasets/USArrests.csv", index_col=0) # DF TEKRAR TANIMLANIYOR

df["cluster"] = clusters_kmeans # AZ ÖNCE TUTMUŞ OLDUĞUM LABEL'I DF'E TAŞIYORUM.

df.head()

df["cluster"] = df["cluster"] + 1 # CLUSTERDA 0 VARDI ONU DÜZELTTİK

df[df["cluster"]==5]

df.groupby("cluster").agg(["count","mean","median"]) # SINIFLARA GÖRE AGGREGE EDİYORUM
# ALGORİTMANIN VERDİĞİNE BAKIP GEÇMİYORUM. ÇIKTIYI KENDİM DE KONTROL EDİYORUM. KÜMELERİ BAKIP MANTIKSIZ MI BAKIYORUM.

df.to_csv("clusters.csv")


################################
# Hierarchical Clustering
################################
# BMÖLÜMEYİCİ VE BİRLEŞTİRİCİ CLUSTER YÖNTEMLERİ

df = pd.read_csv("6.Modül/machine_learning/datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average") # LINKAGE BİRLEŞTİRİCİ(AGGLOMERATIVE ) BİR CLUSTER YÖNTEMİDİR.

# DENDOGRAM METHODU İLE KÜMELEME YAPISINA BAKIYORUZ.
plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

# DAHA SADE BİR CLUSTER İÇİN
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

# HİYERARŞİK CLUSTERIN ACANTAJI BİZE GENELDEN BAKMA FIRSATI TANIR.

################################
# Kume Sayısını Belirlemek
################################


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering # BİRLEŞTİRİCİ CLUSTERING METHODU

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("6.Modül/machine_learning/datasets/USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1 # HİYERARŞİK KÜMELEME YÖNTENLERİNİN CLUSTERLARI

df["kmeans_cluster_no"] = clusters_kmeans # KMEANSDEN GELEN DEĞERLERİ DE ALIYORUZ DF' İN İÇİNE
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1

# ÖDEV !!!
# HİYERARŞİK KÜMELEMEDE 2 OLUP KMEANSTE 2 OLANLAR KİMLER BUNLARIN DIŞINDA OLANLAR NEDEN DIŞINDA.

################################
# Principal Component Analysis
# TEMEL BİLEŞEN ANALİZİ BOYUT İNDİRGEME
################################

df = pd.read_csv("6.Modül/machine_learning/datasets/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col] # UNSUPERVISED LEARNING GEREGI İŞLEMLER SADECE SAYISAL DEĞİŞKENLERLE İLGİLENİYORUM

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df) # TEMEL BİLEŞEN ANALİZİ(BOYUT İNDİRGEME)

pca.explained_variance_ratio_ # BİLEŞENLERİN AÇIKLADIĞI VARYANS ORANI. VARYANS = BİLGİ DİR. 1. BİLEŞENİN AÇIKLADIĞI VARYANS ORANI GİBİ GİBİ
np.cumsum(pca.explained_variance_ratio_) # MESELA 3 DEĞİŞKEN KULLANARAK DEĞİŞKENLİĞİN YÜZDE 82 SİNİ AÇIKALYABİLİYORUM.


################################
# Optimum Bileşen Sayısı
################################

# ELBOW YÖNTEMİNDEKİ GİBİ DİRSEK YAPMIŞ NOKTADAN DEĞİŞKEN İNDİRHEME YAPILABİLİR.
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_ # BİLEŞENLER SIRASIYLA NE KADAR VARYANS AÇIKLIYOR.
np.cumsum(pca.explained_variance_ratio_) # KÜMÜLATİF OLARAK NE KADAR VARYANS AÇIKLIYORLAR.


################################
# BONUS: Principal Component Regression
################################

df = pd.read_csv("6.Modül/machine_learning/datasets/hitters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head() # AZ ÖNCEKİ PCA FİT'İN DATAFRAME'E DÖNMÜŞ HALİ

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1) # PCA İLE KATEGORİK DF BİRLEŞTİRİLİYOR.
final_df.head() # GÜNÜN SONUNDA 16 DEĞİŞKENDEN 3 DEĞİŞKENE DÜŞTÜK VE KATEGORİK DEĞİŞKNELERİ DE EKLEYEREK REGRESYON YAPABİLİRİZ(MESELA)

# ŞİMDİ BU İNDİRGENMİŞ DF İLE MODELLEME YAPIYORUZ. REGRESYON VE CART İLE RMSLERİ BULUCAZ.
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
y.mean()


cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))


################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("6.Modül/machine_learning/datasets/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

# ÇOK FAZLA DEĞİŞKENLİ VERİYİ 2 BOYUTA İNDİRGEYEN FONKSİYON
def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1) # BAĞIMLI DEĞİŞKEN İLE CONCAT EDEREK DIŞARI ÇIKARTIYOR
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")




















