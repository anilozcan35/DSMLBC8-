import pandas as pd
from HousePriceProject.helpers import *

train = get_train_data()
test = get_test_data()

df = train.append(test).reset_index()

def data_preprocessing(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=17, car_th=26)

    # train ve test içerisindeki null değerlere missing ataması yapıyoruz çünkü anlam ifade ediyorlar.
    for col in cat_cols:
        dataframe[col].fillna("missing", inplace=True)  # kategorik değişkenlerdeki missing değerler anlam ifade ediyor.

    # nümerik null değerleri test ve trainde ortalama ile dolduruyoruz.
    for col in num_cols:
        dataframe[col].fillna(dataframe.groupby("Neighborhood")[col].transform("mean"), inplace=True)

    # Outliers
    for col in num_cols:
        if col != "SalePrice":
            replace_with_thresholds(dataframe, col)


