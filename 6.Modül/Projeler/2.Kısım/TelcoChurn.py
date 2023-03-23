# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
# geliştirilmesi beklenmektedir.

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
# bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
# gösterir

# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


##########

################ Kütüphanelerin importu ################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter(action="ignore")

import warnings
import numpy as np
import pandas as pd
from eda import *
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, \
    RobustScaler  # STANDARTLAŞTIRMA DÖNÜŞTÜRME


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

df_ = pd.read_csv("6.Modül/Projeler/2.Kısım/Telco-Customer-Churn.csv")
df = df_.copy()
df.head()
df.info()

check_df(df, head=5)

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df.loc[ df["TotalCharges"] == " "   , "TotalCharges"] = "0"
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

df[df["TotalCharges"] == 0]
check_df(df, head=5)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

for col in cat_cols:
    cat_summary(df, col,plot=True)

for col in num_cols:
    num_summary(df, col, plot = True)

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].count()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# Adım 5: Aykırı gözlem var mı inceleyiniz.

for col in num_cols:
    print(col, ":",check_outlier(df,col))

# Adım 6: Eksik gözlem var mı inceleyiniz.

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# Adım 2: Yeni değişkenler oluşturunuz.

df.loc[(df["MonthlyCharges"] >= 0) & (df["MonthlyCharges"] <= 50) , "NEW_IncomeValue"] = "Low"
df.loc[(df["MonthlyCharges"] > 50) & (df["MonthlyCharges"] <= 100) , "NEW_IncomeValue"] = "Medium"
df.loc[(df["MonthlyCharges"] > 100) , "NEW_IncomeValue"] = "High"
df.head()
df.describe().T
df.loc[(df["tenure"] >= 0) & (df["tenure"] <=12), "NEW_CustomerStatus"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <=24), "NEW_CustomerStatus"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <=36), "NEW_CustomerStatus"] = "2-3 Year"
df.loc[(df["tenure"] > 36), "NEW_CustomerStatus"] = "3+ Year"

# df["NEW_Yield"] = pd.cut(df["TotalCharges"], [0,2000,6000,df["TotalCharges"].max()], include_lowest=True,right=True,labels= ["low-yield", "mid-yield", "high-yield"], ordered=False)
df.loc[(df["TotalCharges"] >= 0) & (df["TotalCharges"] <=2000), "NEW_Yield"] = "low-yield"
df.loc[(df["TotalCharges"] > 2000) & (df["TotalCharges"] <=6000), "NEW_Yield"] = "mid-yield"
df.loc[(df["TotalCharges"] > 6000) & (df["TotalCharges"] <= df["TotalCharges"].max()), "NEW_Yield"] = "high-yield"



df["NEW_NoProt"] = df.apply(lambda x: 1 if (x["DeviceProtection"] != 'Yes') or (x["OnlineSecurity"] != "Yes") or (x["DeviceProtection"] != 'Yes') else 0, axis = 1)

df["New_Service"] = df.apply(lambda x: 1 if (x["PhoneService"] == "Yes") & (x["InternetService"] != "No") else 0, axis =1 )

df["New_SumService"] = ((df[["PhoneService", "OnlineSecurity", "OnlineBackup" , "DeviceProtection","TechSupport", "StreamingTV", "StreamingMovies"]] == "Yes")).sum(axis = 1)
df["New_SumService"] = df.apply(lambda x: x["New_SumService"]+1 if x["InternetService"] != "No" else x["New_SumService"] ,axis= 1)

df["NEW_averagefee"] = df["MonthlyCharges"] / df["New_SumService"]

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
binary_cols = [col for col in cat_cols if df[col].nunique() == 2 and df[col].dtype not in ("int64", "int32","float32", "float64")]
# cat_cols.append("NEW_Yield") # Kateogirik değişkeni yakalayamadı.
# num_cols.remove("NEW_Yield")

for col in binary_cols:
    label_encoder(df,col)

df.head()

ohe_cols = [col for col in cat_cols if df[col].nunique() > 2 and df[col].nunique() < 10 ]

df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.head()

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()
df.shape

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis = 1)

# Modelleme
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # 0.79
print(f"Recall: {round(recall_score(y_pred,y_test),2)}") # 0.64
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # 0.51
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # 0.57
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # 0.73

random_forest = RandomForestClassifier().fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # 0.78
print(f"Recall: {round(recall_score(y_pred,y_test),2)}") # 0.63
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # 0.47
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # 0.54
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # 0.72

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # 0.77
print(f"Recall: {round(recall_score(y_pred,y_test),2)}") # 0.59
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # 0.52
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # 0.56
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # 0.71

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

lgbm_model = LGBMClassifier(random_state=17).fit(X_train, y_train)
lgbm_model.get_params()
y_pred = lgbm_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # 0.79
print(f"Recall: {round(recall_score(y_pred,y_test),2)}") # 0.64
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # 0.52
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # 0.57
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # 0.73

xgboost = XGBClassifier().fit(X_train, y_train)
xgboost.get_params()
y_pred = xgboost.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}") # 0.77
print(f"Recall: {round(recall_score(y_pred,y_test),2)}") # 0.58
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}") # 0.49
print(f"F1: {round(f1_score(y_pred,y_test), 2)}") # 0.53
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}") # 0.70

# Hiper parametre optimizasyonu
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}


lgbm_best_grid = GridSearchCV(lgbm_model ,lgbm_params,
                              n_jobs= -1,
                              cv = 5,
                              verbose = True).fit(X,y)
lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X,y)

cv_result = cross_validate(lgbm_final, X, y, cv= 5,scoring = ["accuracy","precision","roc_auc","recall"])

cv_result["test_accuracy"].mean() # 0.80
cv_result["test_precision"].mean() # 0.66
cv_result["test_roc_auc"].mean() # 0.84
cv_result["test_recall"].mean() # 0.52

lgbm_n_params = {"n_estimators": [500, 1000, 5000, 2500,7500,10000]}
lgbm_best_grid = GridSearchCV(lgbm_model ,lgbm_n_params,
                              n_jobs= -1,
                              cv = 5,
                              verbose = True).fit(X,y)

lgbm_best_grid.best_params_