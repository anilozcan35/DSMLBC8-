################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("6.Modül/machine_learning/datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X) # X'İ STANDARTLAŞTIRIYORUM.

X = pd.DataFrame(X_scaled, columns=X.columns) # ARRAYİN KOLONLARININ ADINI GİRİYORUM.

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45) # 1 TANE RANDOM USER ALIYORUM.

knn_model.predict(random_user) # BU RANDOM USER' I TAHMİN EDİYORUZ.

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X) # BÜTÜN GÖZLEM DEĞERLERİNİ TAHMİN ET.

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1] # 1 SINIFINA AİT OLMA OLASILIKLARI.

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob) # 1 OLMA OLASILIKLARININ THRESHOLDLARI DEĞİŞTİRİLEREK ELDE EDİLEN ERĞİNİN ALANI
# 0.90

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"]) # CROSS VAL SCORE METHODUNDAN FARKLI BİR METHOD
# 5 KATLI OLARAK YAPIYOR

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 0.73
# 0.59
# 0.78

# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params() ## KNN MODELİNİN PARAMETRELERİ

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)} # 2 DEN ELLİYE KADAR KOMŞULUKLARI DEĞİŞTİRİCEK.

knn_gs_best = GridSearchCV(knn_model, # IZGIRADA KOMBİNASYONLARI DENİCEK.
                           knn_params, # PARAMETRE SETİ
                           cv=5, # 5 KATLI CROSS VALİDATION
                           n_jobs=-1, # İŞLEMCİLERİ TAM PERFORMANS İLE ÇALIŞTIRIR.
                           verbose=1).fit(X, y) # RAPOR BEKLEDİĞİME DAİR VERBOSE'U 1 YAPIYORUM.

knn_gs_best.best_params_ # BEST PARAMETRELERİ ALIYORUZ.

# 17 KOMŞULUK SAYISI FİNAL MODEL İÇİN DAHA İYİMİŞ BUNU BULDUK.

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y) # KWARGS İLE DİCT YAPISI PARÇALANIP PARAMETRE OLARAK VERİLİD.

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# 3 ÖLÇÜ METRİĞİ DE ARTMIŞ

random_user = X.sample(1)

knn_final.predict(random_user)











