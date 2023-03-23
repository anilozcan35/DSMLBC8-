
# Outlier sınırlarını belirleyen fonksyion
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
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # AYKIRI DEĞELERİ BASKILAMAK İÇİN KULLANICAĞIMIZ FONKSYİON

def grab_col_names(dataframe, cat_th=10, car_th=20): # SCRİPT BAZINDA DEĞİŞKENLERİ AYIRMAK İÇİN KULLANILAN FONKSYİON
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"] # NUMERİK GÖRÜNEN AMA KATEGORİK FONKSİYONLAR
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] # ÖLÇÜLEMEZ KATEGORİKLER
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # KATEGORİK DEĞİŞKENLERİN SON HALİ

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] # INTEGER VE FLOAT OLANLAR GELECEK
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# AXİS = 1 NE İŞE YARAR FARKI NEDİR
import pandas as pd
df = pd.read_csv("as")
df.notnull().all(axis=1)

# KATEGORİK DEĞİŞKENLERİN ÖZETİ
def cat_summary(dataframe, col_name, plot=False): # DEĞİŞKENİN İSMİ VE İLGİLİ DEĞİŞKENİN SINIFLARININ DAĞILIMI
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# NUMERİK DEĞİŞKENLERİN SUMMARYSİ
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
# max_rows almamızı sağlar
pd.set_option('display.max_rows', None)


########## CONFUSION MATRIX ##############

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


# CART ELİMİZDEKİ MODELDEKİ DEĞİŞKENLERİN ÖNEM SERİSİNİ GÖRSELLEŞTİREN FONKSİYON
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

# HİPERPARAMETRELERE GÖRE HATALARI GÖRSELLEŞTİREN FONKSİYON
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
# ELİMİZDEKİ MODELDEKİ DEĞİŞKENLERİN ÖNEM SERİSİNİ GÖRSELLEŞTİREN FONKSİYON