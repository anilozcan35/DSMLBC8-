#################################################
#### import işlemleri ve ayarlar
#################################################
import pandas as pd
from eda import *
from set_option import *
import datetime
from matplotlib import pyplot as plt

set_pandas_options()

#################################################
#### datasetin okunması
#################################################

df_ = pd.read_csv("Attempt/Test.csv - test.csv")
df = df_.copy()

df.head()
df.tail()
# Tahmin edilen değişken upfront_price. Bunu drop edip biz tahmin etmeye çalışacağız.

upfront_price = df["upfront_price"]
df.drop(labels="upfront_price", inplace=True, axis=1)


#################################################
#### Keşifçi veri analizi
#################################################

check_df(df, head=10)
# metered price 20 na değer var --> DROP
# device_token --> hepsi na
# change_reason_pricing --> 300 dolu değer var kontrol edilebilir. --> Boş ise değişiklik yok veya fiyatı etkilemedi anlamına geliyor.

df[df["metered_price"].isnull()]
df[df["device_token"].isnull()]
df[df["change_reason_pricing"].isnull()]

# metered_price'daki null değişkenleri düşürüyoruz.
df = df[~df["metered_price"].isnull()]

# device_token tamamen boş drop ediyoruz.
df.drop(labels="device_token", inplace=True, axis=1)

# calc_created kolonunu datetime olarak değiştiriyoruz.
df['calc_created'] = df["calc_created"].apply(pd.to_datetime)

check_df(df)

# değişkenlerin yakalanması

for col in df.columns:
    print(col + ":" + str(df[col].nunique()))

# df["device_name"][4941] = df["device_name"].apply(lambda x: x.split()[0])

df.info()
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=9, car_th=35)

######### Kategorik değişken analizi

for col in cat_cols:
    cat_summary(df, col, plot= True)

df["prediction_price_type"]
df.head(200)

df["rider_app_version"].value_counts() # CA CI olarak 2 tane versiyon var

# driver_app_version --> DA DI olarak 2 tane var
# b_state içerisinde sadece bir değer var
# predicted_price_type --> yolculuktan önce ve yolculuk esnasında edilen tahmin olarak 2 ye indirgenebilir
# order_state --> 1 tane active değeri var drop etmek mantıklı olabilir

######### Nümerik değişken analizi

for col in num_cols:
    num_summary(df, col, plot=True)

# driver_device_uid_new drop edilebilir id kolonu
# ticket_id_new id değişkeni drop edilebilir
# order_id
# order_id_new

######### Hedef değişken analizi

df["metered_price"].describe([0, 0.1, 0.2, 0.5, 0.75, 0.8 , 0.9, 0.95, 0.99])

df[df["metered_price"] == df["metered_price"].max()]

for col in cat_cols:
    target_summary_with_cat(df, "metered_price", col)

df["metered_price"].hist()
plt.show()
# hedef değişkendeki outlierları baskılamayı düşünüyorum. 200000 birimlik yolculuklar mantıklı gelmiyor

df[df["eu_indicator"] == 0]



import seaborn as sns

sns.boxplot(df["metered_price"])
plt.show()