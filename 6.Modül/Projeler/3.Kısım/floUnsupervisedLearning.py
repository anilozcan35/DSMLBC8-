# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

# import işlemleri
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
from datetime import date

# pandas ayarlamaları
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# datasetin okunması

df_ = pd.read_csv("6.Modül/Projeler/3.Kısım/flo_data_20k.csv")
df = df_.copy()

df.head()
df.shape
df.info()
df["master_id"].nunique()

# kullanılacak değişkenlerin oluşturulması
today = date.today()
today = pd.to_datetime(today)
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df["order_num_totaL_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["Tenure"] = (today - df["first_order_date"]).dt.days
df["Recency"] = (today - df["last_order_date"]).dt.days


# master_id index olarak düzenleniyor
df.index = df["master_id"]
df.drop(["master_id"],inplace=True, axis= 1)

# K- means

sc = MinMaxScaler((0,1))

num_cols = [col for col in df.columns if df[col].dtypes in ["int32", "int64","float32","float64"]]
df[num_cols] = sc.fit_transform(df[num_cols])
df = df[num_cols]
df_h = df
df.head()

# Optimum küme sayısı kaç

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

elbow.elbow_value_

# Müşterileri segmente etmek
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df) # ELBOWDAN GELEN K DEĞERİNİ SET EDİYORUM.

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5] # SONUÇLARI GÖREMİYORUM HANGİ EYALET HANGİ CLUSTERDA BİLEMİYORUM. ÇÜNKÜ ARRAY'E DÖNDÜ

clusters_kmeans = kmeans.labels_

df = pd.read_csv("6.Modül/Projeler/3.Kısım/flo_data_20k.csv", index_col=0) # DF TEKRAR TANIMLANIYOR

df["cluster"] = clusters_kmeans # AZ ÖNCE TUTMUŞ OLDUĞUM LABEL'I DF'E TAŞIYORUM.

df.head()

df["cluster"] = df["cluster"] + 1 # CLUSTERDA 0 VARDI ONU DÜZELTTİK

df[df["cluster"]==5]

df.groupby("cluster").agg(["count","mean","median"]) # SINIFLARA GÖRE AGGREGE EDİYORUM
# ALGORİTMANIN VERDİĞİNE BAKIP GEÇMİYORUM. ÇIKTIYI KENDİM DE KONTROL EDİYORUM. KÜMELERİ BAKIP MANTIKSIZ MI BAKIYORUM.

df.to_csv("clusters.csv")

# Hierarchical Clustring

hc_average = linkage(df_h, "average") # LINKAGE BİRLEŞTİRİCİ(AGGLOMERATIVE ) BİR CLUSTER YÖNTEMİDİR.

# # DENDOGRAM METHODU İLE KÜMELEME YAPISINA BAKIYORUZ.
# plt.figure(figsize=(10, 5))
# plt.title("Hiyerarşik Kümeleme Dendogramı")
# plt.xlabel("Gözlem Birimleri")
# plt.ylabel("Uzaklıklar")
# dendrogram(hc_average,
#            leaf_font_size=10)
# plt.show()

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

# Küme sayısını belirlemek için

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

# Final model

from sklearn.cluster import AgglomerativeClustering # BİRLEŞTİRİCİ CLUSTERING METHODU

cluster = AgglomerativeClustering(n_clusters=10, linkage="average")

clusters = cluster.fit_predict(df_h)

df = pd.read_csv("6.Modül/Projeler/3.Kısım/flo_data_20k.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1 # HİYERARŞİK KÜMELEME YÖNTENLERİNİN CLUSTERLARI

df.hi_cluster_no.nunique()

df.groupby("hi_cluster_no").agg(["mean","median"])