import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


def cat_summary(dataframe, col_name, plot=False, groupby="survived"):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

    print(f"The mean of {groupby} to {col_name} = \n {df.groupby(groupby)[col_name].mean()}")
    ## col_name'e nümerik girilmezse hata vericektir iki tane categoriyi grouplayamaz.


cat_summary(df, "age", True)


def check_df(dataframe, head=5):
    """
    Gönderilen dataframein checkini yapar. Shape, missing control ,head ve tail ve dtypeları yazdırır.
    Parameters
    ----------
    dataframe: pd.dataframe
    head : int

    Returns
    -------

    Notes
    -------
    Buraya ne not yazsam bilemedim.

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


######################## 2. kısımı ###############################

df = sns.load_dataset("titanic")

df.head()
# 2
df["sex"].value_counts()
df.info()
df["sex"].nunique()
# 3
for col in df.columns:
    print(f"{col} : {df[col].nunique()}")

# 4
df["pclass"].nunique()
df["pclass"].value_counts()

# 5
liste = ["pclass", "parch"]

for col in liste:
    print(df[col].value_counts())

# 6 Object olarak görünüyordu pandas categorysine çevirdik
df["embarked"] = df["embarked"].astype("category")

# 7
df[df["embarked"] == "C"]

# 8
df[df["embarked"] != "S"]

# 9
df[(df["sex"] == "male") & (df["age"] < 30)]

# 10
df[(df["fare"] > 500) | (df["age"] > 70)]

# 11
df.isnull().sum()

# 12
df.drop("who", axis=1, inplace=True)

# 13 burada seriden ençok tekrar eden kısmı direk fillna'nın içerisine nasıl gömebiliriz.
type(df[["deck"]].value_counts())
# en çok tekrar eden

df["deck"].mode()[0]

df["deck"].fillna("C", inplace = True)

# 14
df[["age"]].median()
df["age"].fillna(df["age"].median())

# 15
dict = {"pclass": ["mean", "sum", "count"],
        "sex": ["mean", "sum", "count"]
        }
# pclass ve sex kırılımda , survived değişkeninin aggregateleri
df.groupby(["pclass", "sex"]).agg({"survived": ["mean", "sum", "count"]})

# 16 age_flag

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# 17

df = sns.load_dataset("tips")

# 18

df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

# 19

df.groupby(["time", "day"]).agg({"total_bill": ["sum", "min", "max", "mean"]})
df.head(100)

# 20

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                          "tip": ["sum", "min", "max", "mean"]})

# 21

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

# 22

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# 23

df.groupby("sex").agg({"total_bill": "mean"}).loc["Male"]


def total_bill_flag_func(sex, total_bill):
    if sex == "Male":
        if total_bill > df.groupby("sex").agg({"total_bill": "mean"}).loc["Male"][0]:
            return 1
        else:
            return 0
    else:
        if total_bill > df.groupby("sex").agg({"total_bill": "mean"}).loc["Female"][0]:
            return 1
        else:
            return 0


df["total_bill_flag"] = df.apply(
    lambda x: total_bill_flag_func(x["sex"],x["total_bill"]), axis=1)

#24

df.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag" : "count"})

#25
df_yeni = df.sort_values("total_bill_tip_sum").head(30)


