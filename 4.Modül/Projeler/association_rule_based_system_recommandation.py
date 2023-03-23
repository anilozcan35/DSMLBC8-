import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("4.Modül/Projeler/datasets/online_retail_II.xlsx", sheet_name= "Year 2010-2011")
df = df_.copy()
df.head()
df.info()

# post her faturaya eklenen bedel ürünü ifade etmiyor. Datasetten çıkartılıyor.
df = df[~(df["StockCode"] == "POST")]
# null değerler drop ediliyor.
df.dropna(inplace = True)
# invoice içerisindeki C iadeyi ifade ediyor. Datasetten çıkartılıyor.

df = df[~df["Invoice"].str.contains("C", na = False)]

# Price değeri 0'dan küçük olan değerlere bir bakalım
df.isnull().sum()
df[df["Price"] < 0]

# price ve quantityler eğer uçuk kaçıksa ise baskılıyoruz.
df.describe().T

def outlier_thresholds(dataframe, variable): # OUTLIER HESAPLAMAK İÇİN
    quartile1 = dataframe[variable].quantile(0.01) # EŞİKLER NORMALDE 0.25 ÜZERİNDEN YAPILIRDI NORMALDE PROBLEM BAZINDA UCUNDAN TIRAŞLAMAK İÇİN BÖYLE YAPIYORUZ.
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON

# outlier sınırları ile değerleri değiştiren fonksiyon
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON

# aykırı değerleri baskılıyoruz.
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df,"Price")

# apriori datasetini oluşturmak
df.head()

df = df[df["Country"] == "Germany"]

df.groupby(["Invoice","Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:5,0:5]

# null değerleri 0 ile dolduralım

data = df.groupby(["Invoice","Description"]).agg({"Quantity": "sum"}).unstack().fillna(0)

df["Country"].unique()

# datayı apriori nin istediği hale getirmek için
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

data = create_invoice_product_df(df, id = True)

# kuralları oluşturacak fonksyion

frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

rules.sort_values(by = "support", ascending=False)
# stockcodeu bulabilmek için
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, 22326)
