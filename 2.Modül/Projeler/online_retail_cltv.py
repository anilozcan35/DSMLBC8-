import pandas as pd
import datetime as dt

from lifetimes import BetaGeoFitter, GammaGammaFitter

df_ = pd.read_excel("2.Modül/crm_analytics/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

def outlier_thresholds(dataframe, variable): # OUTLIER HESAPLAMAK İÇİN
    quartile1 = dataframe[variable].quantile(0.01) # EŞİKLER NORMALDE 0.25 ÜZERİNDEN YAPILIRDI NORMALDE PROBLEM BAZINDA UCUNDAN TIRAŞLAMAK İÇİN BÖYLE YAPIYORUZ.
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df.head()
df.describe().T

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.isnull().sum()
df.dropna(inplace = True)
df = df[~df["Invoice"].str.contains("C", na = False)]

df["TotalPrice"] = df["Quantity"] + df["Price"]

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) # BU RECENCY MÜŞTERİNİN KENDİ İÇİNDE İLK SATIN ALMASI - SON SATIN ALMASI
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış) # TODAY E GÖRE MÜŞTERİNİN İLK ALIŞ VERİŞİ FARKI
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç # ALIŞIK OLDUĞUMUZDAN FARKLI OLARAK TOPLAM KAZANÇ DEĞİL SATIN ALMA BAŞINA ORTALAMA KAZANÇ

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                        lambda x: (today_date - x.min()).days],
                                         "Invoice": lambda x: x.nunique(),
                                         "TotalPrice": lambda x: x.sum()
                                         })

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7 # BG/NBD İSTEDİĞİ GİBİ HAFTALIK CİNSE ÇEVİRİYORUM

cltv_df["T"] = cltv_df["T"] / 7
##########
# MODEL NESNESİ OLUŞTURUCAM DAHA SONRA FİT EDİCEM
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


bgf.conditional_expected_number_of_purchases_up_to_time(1, # 1 HAFTANIN 1 İ
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

#cltv score hesaplanması
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_1_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_final_1month = cltv_df.merge(cltv_1_month, on="Customer ID", how="left")

cltv_final_1month.sort_values(by="clv", ascending=False).head(10)

###### cltv 12 months

cltv_12_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv_final_12month = cltv_df.merge(cltv_12_month, on="Customer ID", how="left")

cltv_final_12month.sort_values(by="clv", ascending=False).head(10)

## segmentlere ayrılması

cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})







































########################################################################

df = df_.copy()
df.head()
df.isnull().sum()
df = df[~df["Invoice"].str.contains("C", na=False)] # İADE OLAN ÜRÜNLERİ ÇIKARTALIM
df.describe().T
df = df[(df['Quantity'] > 0)] # BETİMSEL İSTATİSTİKTE QUANTITY VE PRICE DEĞİŞKNELERİ NEGATİF GELİYOR BUNDAN KURTULMAMIZ LAZIM
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'TotalPrice': lambda x: x.sum()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c.head()
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c.head()
cltv_c.shape[0]
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10


##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c["purchase_frequency"]

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv", ascending=False).head()


##################################################
# 8. Segmentlerin Oluşturulması
##################################################

cltv_c.sort_values(by="cltv", ascending=False).tail()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_c.sort_values(by="cltv", ascending=False).head()

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltc_c.csv")