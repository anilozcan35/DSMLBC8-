import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import LabelEncoder,  StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

# 1. DataFrameleri okutuyoruz.
df_ = pd.read_csv("6.Modül/Projeler/2.Kısım/scoutium_attributes.csv", sep=";")
df_atr = df_.copy()

df_ = pd.read_csv("6.Modül/Projeler/2.Kısım/scoutium_potential_labels.csv", sep = ";")
df_lbls = df_.copy()

df_atr.head()
df_lbls.head()

# 2. DataFrameleri Merge fonksiyonu ile birleştiriyoruz.
df = df_atr.merge(df_lbls, how = "inner", on =["task_response_id", "match_id", "evaluator_id", "player_id"])

# 3. Position_id = 1 (Kaleci) olanları datasetten kaldırıyoruz.
df = df[~(df["position_id"] == 1)]

# 4. Below Avarage sınıfını veri setinden kaldırıyoruz.
df = df[~(df["potential_label"] == "below_average")]

# 5. Pivot Table işlemleri
# 5.1  İndekste “player_id”,“position_id” ve“potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların
# oyunculara verdiği puan “attribute_value” olacakşekilde pivot table’ı oluşturunuz

df_pivot = pd.pivot_table(data=df, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns="attribute_id")

# 5.2 DataFrame indexlerini değişken haline getiriyoruz.
df_pivot = df_pivot.reset_index()

# 6. Potantial_label kolonunu label_encoder ile sayısal hale getiriyoruz.
df_pivot["potential_label"].value_counts()
le = LabelEncoder()
df_pivot["potential_label"] = le.fit_transform(df_pivot["potential_label"])
df_pivot.head()

# 7. Sayısal Değişkenleri ayırıyoruz.

num_cols = [col for col in df_pivot.columns if df_pivot[col].dtypes in ["int64","int32","float32","float64"] and col != "potential_label"]

df_pivot.columns.value_counts().sum()
len(num_cols) # tüm kolonlar nümerik olarak bulundu.

# 8. Veriyi ölçeklendirmek adına StandartScaler kullanılıyoruz.
ss = StandardScaler()
df_pivot[num_cols] = ss.fit_transform(df_pivot[num_cols])


# 9. Model kuruyoruz.
y = df_pivot["potential_label"]
X = df_pivot.drop("potential_label", axis = 1)

rf_model = RandomForestClassifier().fit(X, y)

cv_model = cross_validate(rf_model, X,y, cv=3 ,scoring= ["accuracy", "f1", "precision", "recall","roc_auc"])

cv_model['test_accuracy'].mean()
# 0.8597069597069598

cv_model['test_f1'].mean()
# 0.5689781482884931

cv_model['test_precision'].mean()
# 0.775

cv_model['test_recall'].mean()
# 0.46296296296296297

cv_model['test_roc_auc'].mean()
# 0.8872555917048751

# 10. Feature importance görselleştirmesi

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model,X)