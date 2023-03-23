import pandas as pd
import datetime as dt

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

df_ = pd.read_csv(r"C:\Users\Sade\Desktop\VBO\paylaşılamazCaseler\flo_data_20k.csv")

df = df_.copy()

df.describe().T
df.info()


def outlier_thresholds(dataframe, variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quantile3 - quantile1
    up_limit = round(quantile3 + 1.5 * interquantile_range)
    low_limit = round(quantile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# order_num_total_ever_online
# order_num_total_ever_offline
# customer_value_total_ever_offline
# customer_value_total_ever_online

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# toplam alış veriş sayısı ve toplam alışveriş değerini buluyoruz
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# date kolonlarını date'e çeviriyoruz
for datecolumn in [col for col in df.columns[df.columns.str.contains("date")]]:
    df[datecolumn] = pd.to_datetime(df[datecolumn])

###########################################################
df.loc[:, df.columns.str.contains("date")].max()

today_date = dt.datetime(2021, 6, 1)

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) # BU RECENCY MÜŞTERİNİN KENDİ İÇİNDE İLK SATIN ALMASI - SON SATIN ALMASI
# T:tenure Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış) # TODAY E GÖRE MÜŞTERİNİN İLK ALIŞ VERİŞİ FARKI
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç # ALIŞIK OLDUĞUMUZDAN FARKLI OLARAK TOPLAM KAZANÇ DEĞİL SATIN ALMA BAŞINA ORTALAMA KAZANÇ


# cltv_df i oluşturmaya çalışıyoruz.
# df.drop("recency_date", inplace=True, axis= 1)
df["recency_date"] = df["last_order_date"] - df["first_order_date"]
df["recency_date"] = df["recency_date"].apply(lambda x: x.days)



cltv_df = df.groupby("master_id").agg(
    {"first_order_date": lambda first_order_date: (today_date - first_order_date.min()).days,  # T olacak değiişken
     "customer_value_total": lambda customer_total_value: customer_total_value.max(),
     "order_num_total": lambda order_num_total: order_num_total.max(),
     "recency_date": lambda recency_date: recency_date.max()
     })
cltv_df = cltv_df.reset_index()

cltv_df["first_order_date"] = cltv_df["first_order_date"] / 7
cltv_df["recency_date"] = cltv_df["recency_date"] / 7
cltv_df["customer_value_total"] = cltv_df["customer_value_total"] / cltv_df["order_num_total"]

cltv_df.columns = ["customer_id", "T_weekly", "monetary_cltv_avg", "frequency", "recency_cltv_weekly"]

# bg/nbd model kurulumu ve fit edilmesi
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içindeki tahminler
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,
                                                                                       cltv_df["frequency"],
                                                                                       cltv_df["recency_cltv_weekly"],
                                                                                       cltv_df["T_weekly"])
# 6 ay içindeki tahminler

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                                                       cltv_df["frequency"],
                                                                                       cltv_df["recency_cltv_weekly"],
                                                                                       cltv_df["T_weekly"])

plot_period_transactions(bgf)
plt.show()

# gamma gamma model kurulumu ve fit edilmesi

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["expected_avg_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary_cltv_avg"])

cltv_df.sort_values("expected_avg_profit", ascending= False)

# cltv hesaplanması
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,cltv_df["frequency"],cltv_df["recency_cltv_weekly"],cltv_df["T_weekly"],cltv_df["monetary_cltv_avg"],
                                              time=6,discount_rate=0.01, freq = "W")

cltv_df.sort_values("cltv", ascending= False).head(20)