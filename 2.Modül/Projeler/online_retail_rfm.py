import pandas as pd
import datetime as dt

df_ = pd.read_excel("2.Modül/crm_analytics/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()
df.describe().T
df.isnull().sum()
df.info()

# null değerleri düşürüyorum.
df.dropna(inplace = True)

# Eşsiz ürün sayısı
df["StockCode"].nunique()

# hangi üründen kaç tane var
df["StockCode"].value_counts()

# en çok sipariş edilen ilk 5 ürün
df["StockCode"].value_counts().sort_values(ascending= False).head(5)

# İadeleri dışarı çıkartıyoruz
df = df[~df["Invoice"].str.contains("C", na = False)]

# TotalPrice değişkenini ekliyoruz.
df["TotalPrice"] = df["Quantity"] * df["Price"]

# rfm metriklerini atıcaz

df["InvoiceDate"].max()
today_date = dt.datetime(2011,12,11)


rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,
                                     "Invoice": lambda x: x.nunique(),
                                     "TotalPrice": lambda x: x.sum()
                                     })

rfm.columns = ["recency", "frequency", "monetary"]

rfm = rfm[rfm["monetary"] > 0]


###### RFM SKORLARININ ÜRETİLMESİ ###########

rfm["recency_score"] = pd.qcut(rfm["recency"], 5 , labels= [5, 4, 3, 2, 1])

rfm["frequency"].describe().T
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1, 2, 3, 4, 5])

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

## Segment Haritasının Tanımlanması

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

## Segment Yorumları

rfm[(rfm["segment"] == "promising") | (rfm["segment"] == "new_customer") | (rfm["segment"] == "champions")]

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"]).sort_values(by= ("recency", "mean"))

# Loyal customers listesini excel olarak alalım.
exportdf = rfm[rfm["segment"] == "loyal_customers"].index.astype(int)

exportdf.to_csv("loyal_customers")