import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("1.Modül/Ödevler/kuralTabanliSiniflandirmaİlePotansiyelMüşteriGetirisiHesaplama/persona.csv")

df.head()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
df.head()
df.info()
df["SOURCE"].unique()
df["SOURCE"].value_counts()
df["PRICE"].nunique()
df["PRICE"].value_counts()

df["COUNTRY"].value_counts()

# ülkelere göre satışlar toplamı
df.groupby("COUNTRY").agg("PRICE").sum()

# Sourcelara göre satışlar toplamı
df["SOURCE"].value_counts()

# Ülkelere göre price ortalamaları
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Sourcelara göre price ortalamaları
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Country Source kırılımında price ortalamaları
df.groupby(["COUNTRY", "SOURCE"]).mean()

# Countr Source Sex Age kırılımında price ortalamaları
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)

# Çıktıyı agg_df olarak kaydet
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)

# Price dışındakileri index olarak atama
# df.index = df.loc[: , ~df.columns.str.contains("PRICE")]
agg_df = agg_df.reset_index()

# Age'i kategorik yapalım
agg_df["AGE"].describe()

labels = ["{0}_{1}".format(agg_df["AGE"].quantile(i / 4).astype(int), agg_df["AGE"].quantile((i + 1) / 4).astype(int))
          for i in np.arange(0, 4)]

agg_df["AGE_CUT"] = pd.qcut(agg_df["AGE"], 4, labels=labels)

# Seviye Tabanlı Müşterilerin Tanımlanması

agg_df["customers_level_based"] = agg_df["COUNTRY"].str.upper() + \
                                  "_" + \
                                  agg_df["SOURCE"].str.upper() + \
                                  "_" + \
                                  agg_df["SEX"].str.upper() + \
                                  '_' + \
                                  agg_df["AGE_CUT"].str.upper()

agg_df["customers_level_based"] = [agg_df[kolon].str.upper()+"_" if kolon != "AGE_CUT" else agg_df[kolon].str.upper() for kolon in ["COUNTRY", "SOURCE", "SEX", "AGE_CUT"]]

# Formatla yapılabilir mi acaba ?
agg_df["customers_level_based"] = "{0}_{1}_{2}_{3}".format(agg_df["COUNTRY"].str.upper(),
                                                           agg_df["SOURCE"].str.upper(),
                                                           agg_df["SEX"].str.upper(),
                                                           agg_df["AGE_CUT"].str.upper())

agg_df["customers_level_based"].value_counts().reset_index()
# ??? birden fazla üyesi olan gruplara ortalamarı atıyoruz.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()

# Segmentlere ayırma işlemi
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "sum", "max"]})

# Yeni gelen müşterilen segmentasyona sokulması
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

agg_df["AGE_CUT"].value_counts()
df
new_user = "TUR_ANDROID_FEMALE_25_34"

fransiz = "FRA_IOS_FEMALE_34_66"

agg_df[agg_df["customers_level_based"] == fransiz]

