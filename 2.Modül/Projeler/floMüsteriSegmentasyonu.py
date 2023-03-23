#####################################
# floMüsteriSegmentasyonu
#####################################
import pandas as pd
import datetime as dt

df_ = pd.read_csv(r"C:\Users\Sade\Desktop\VBO\paylaşılamazCaseler\flo_data_20k.csv")

df = df_.copy()

df.head(10)
df.columns
df.describe().T
df.isnull().sum()
df.info()

# total alış veriş sayısı ve total yapılan harcamaların hesaplanması
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# date içeren kolonların date e çevrilmesi

for datecolumn in [col for col in df.columns[df.columns.str.contains("date")]]:
    df[datecolumn] = pd.to_datetime(df[datecolumn])

# alışveriş kanallarındaki müşteri sayısının toplam ürün sayısının ve toplam harcamaların dağılımına bakın.
df.groupby("order_channel").agg({"master_id": ["count"],
                                 "order_num_total": ["sum"],
                                 "customer_value_total": ["sum"]})

# en fazla kazancı getiren 10 müşteri
df.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total", ascending=False).head(
    10)

df.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(10)


def data_prepare(dataframe):
    print("#####################################")
    print(dataframe.head(10))
    print("#####################################")
    print(dataframe.columns)
    print("#####################################")
    print(dataframe.describe().T)
    print("#####################################")
    print(dataframe.isnull().sum())
    print("#####################################")
    print(dataframe.info())
    print("#####################################")

    # total alış veriş sayısı ve total yapılan harcamaların hesaplanması
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]

    # date içeren kolonların date e çevrilmesi

    for datecolumn in [col for col in dataframe.columns[dataframe.columns.str.contains("date")]]:
        dataframe[datecolumn] = pd.to_datetime(dataframe[datecolumn])

    # alışveriş kanallarındaki müşteri sayısının toplam ürün sayısının ve toplam harcamaların dağılımına bakın.
    print(f"ALIŞVERİŞ KANALLARINDAKİ KIRLIM: \n ")
    print(dataframe.groupby("order_channel").agg({"master_id": ["count"],
                                                  "order_num_total": ["sum"],
                                                  "customer_value_total": ["sum"]}))

    # en fazla kazancı getiren 10 müşteri
    dataframe.groupby("master_id").agg({"customer_value_total": "sum"}).sort_values("customer_value_total",
                                                                                    ascending=False).head(10)

    dataframe.groupby("master_id").agg({"order_num_total": "sum"}).sort_values("order_num_total", ascending=False).head(
        10)


data_prepare(df)

####### RFM METRİKLERİNİ OLUŞTURMA ##############

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)

# 2 rfm metriklerini hesaplayalım.

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                                   "order_num_total": lambda order_total_num: order_total_num.max(),
                                   "customer_value_total": lambda customer_value_total: customer_value_total.max()})

rfm.columns = ["recency", "frequency", "monetary"]

##################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

rfm.describe().T


######## RF SKORLARININ SEGMENT OLARAK TANIMLANMASI ###############
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex = True)

#####################################
rfm.groupby("segment").agg({"recency":"mean",
                            "frequency": "mean",
                            "monetary": "mean"})

rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

## Kampanya

df["mean_order_value"] = df["customer_value_total"] / df["order_num_total"]

rfm_k = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == 'loyal_customers')].index.values

df[df["master_id"].isin(rfm_k) & (df["interested_in_categories_12"].str.contains("KADIN")) & (df["mean_order_value"] > 250)  ].loc[:, "master_id"]

#### farklı bir deneme
rfm_k = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == 'loyal_customers')].reset_index()

df_k = df.merge(rfm_k, how ="inner" , on ="master_id")

df_k[(df_k["segment"] == "champions") | (df_k["segment"] == 'loyal_customers') & (df_k["interested_in_categories_12"].str.contains("KADIN") & (df["mean_order_value"] > 250) )]

# farklı bir deneme




## KAMPANYA

df_k[((df_k["segment"] == "hibernating") | (df_k["segment"] == 'cant_loose') | (df_k["segment"] == 'new_customers')) &
     (df_k["interested_in_categories_12"].str.contains("ERKEK") | df_k["interested_in_categories_12"].str.contains("ÇOCUK"))].loc[:,"master_id"]


df[rfm[(rfm["segment"] == "champions") | (rfm["segment"] == 'loyal_customers')].reset_index().iloc[:,0]]



rfm.to_csv("tocsv")

tocsv = rfm[(rfm["segment"] == "cant_loose") |
            (rfm["segment"] == 'new_customers') | (rfm["segment"] == 'hibernating')].reset_index().iloc[:,0]

