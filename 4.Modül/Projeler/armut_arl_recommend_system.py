import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv("4.Modül/Projeler/datasets/armut_data.csv")
df = df_.copy()
df.head()
type(df.values)

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)


# tarihin içerisinden ay ve yıl değişkenini çekiyoruz. ve user_id + new_date şeklinde sepet_id oluşturucaz.
df.info()

df["CreateDate"] = pd.to_datetime(df["CreateDate"]) # datetime a çevirmemiz gerekiyor.
df["new_date"] = df["CreateDate"].dt.to_period('M') # ay ve yılı çekiyoruz. --> strftime() ile de yapılabilir.

df["sepet_id"] = df["UserId"].astype(str) + "_" + df["new_date"].astype(str) # sepet id yi oluşturuyoruz.

# appriopinin bizden istediği şekilde dataframe'i oluşturuyoruz.
df.pivot_table(values = "UserId", index = df["sepet_id"], columns= df["Hizmet"], aggfunc="count")

df_pivot = pd.pivot_table(df, values= "UserId", index = "sepet_id", columns= "Hizmet", aggfunc= "count", fill_value= 0)

df_pivot = df_pivot.applymap(lambda x: 1 if x >= 1 else 0)

# APPRIPRO
frequent_itemsets = apriori(df_pivot,
                            min_support=0.01,
                            use_colnames=True
                            )

# birliktelik kuralları yazılıyor
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01
                          )

# kuralları özelleştiriyorum
rules[(rules["lift"] > 2) & (rules["confidence"] > 0.1)]

# arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

# recommender fonksiyon
def arl_recommender(rules_df, hizmet_id):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, hizmet in enumerate(sorted_rules["antecedents"]):
        for j in list(hizmet): # FROZEN SETTEN KURTULMAK İÇİN
            if j == hizmet_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list


arl_recommender(rules, "2_0")



