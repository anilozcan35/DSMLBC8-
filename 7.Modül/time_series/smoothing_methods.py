##################################################
# Smoothing Methods (Holt-Winters)
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing # hotwinters
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose # zaman serisini bileşene ayırmak için
import statsmodels.tsa.api as smt

warnings.filterwarnings('ignore')


############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas() # HAFTALIK CO2 SALINIMI DATASETİ
y = data.data

y = y['co2'].resample('MS').mean() # HAFTALIKTAN ZİYADE AYLARA GÖRE GÖZLEMLEMEK İÇİN

y.isnull().sum() # NULL DEĞERLER

y = y.fillna(y.bfill()) # MOD MEDYANDAN ZİYADE GÖZLEM KENDİNDEN ÖNCEKİ GÖZLEMLE DOLDURULUR

y.plot(figsize=(15, 6))
plt.show()


############################
# Holdout
############################

train = y[:'1997-12-01'] #1997 DEN ÖNCEKİ DATALARI TRAIN OLARAK AL
len(train)  # 478 ay

# 1998'ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':] # 98 İN İLK AYINDAN İTİBAREN DE TEST SETİ
len(test)  # 48 ay

##################################################
# Zaman Serisi Yapısal Analizi
##################################################

# Durağanlık Testi (Dickey-Fuller Testi) # DURAĞANLIK

def is_stationary(y):

    # "HO: Non-stationary" # P VALUE 0.05 TEN DÜŞÜKSE H0 RED EDİLİR
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)

# Zaman Serisi Bileşenleri ve Durağanlık Testi # LEVEL, TREND, SEAOSANAL, RESIUDALS BİLEŞENLERİ TEST ETMEK İÇİN FONKSİYON
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)


##################################################
# Single Exponential Smoothing
##################################################

# SES = Level 'I MODELLER

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5) # TRAINE FIT EDIYORUM ALFA HİPERPARAMETRESİ

y_pred = ses_model.forecast(48) # BURDA PREDICT YERINE FORECAST KULLANILIR

mean_absolute_error(test, y_pred) # TAHMİN SONUÇLARI İLE GERÇEK DEĞERLERİN FARKLARIN FARKIYLA HATALAR ÖLÇÜLÜR.

train.plot(title="Single Exponential Smoothing") # MODEL TAHMİNİ İLE TEST SONUÇLARI GÖRSELLEŞTİRMESİ
test.plot()
y_pred.plot()
plt.show()


train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()


def plot_co2(train, test, y_pred, title): # CO2 DATASETİNDE TAHMİNLER İLE TEST VERİLERİ GÖRSELLEŞTİRMESİ
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params # SMOOTHING LEVEL 0.5 ALFA PARAMETRESİ

############################
# Hyperparameter Optimization
############################

def ses_optimizer(train, alphas, step=48):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01) # SES ZAYIF Bİ MODEL O YÜZDEN ÖĞRENME PARAMETRESİNİ REMEMBERA GÖRE YÜKSEK TUTUYORUM

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

ses_optimizer(train, alphas) # BEST ALPHA 0.99

best_alpha, best_mae = ses_optimizer(train, alphas)

############################
# Final SES Model
############################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha) # FINAL SES MODELI
y_pred = ses_model.forecast(48) # 48 ADIMLIK TAHMİN ET.

plot_co2(train, test, y_pred, "Single Exponential Smoothing") # VERİLEN


##################################################
# Double Exponential Smoothing (DES)
##################################################

# DES: Level (SES) + Trend

#!!!!!!!!!!!!!!!!!! MEVSİMSELİK VE ARTIKLAR GRAFİKTE TRENDE BAĞLI DEĞİLSE TOPLAMSALDIR.

# y(t) = Level + Trend + Seasonality + Noise # ADDİTİVE
# y(t) = Level * Trend * Seasonality * Noise # ÇARPIMSAL

# MUL VE ADD OLARAK DEĞER VER HANGİSİ DAHA AZ HATA VERİYORSA ONU AL.

ts_decompose(y) # --> ADDITIVE

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_trend=0.5)

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

############################
# Hyperparameter Optimization
############################


def des_optimizer(train, alphas, betas, step=48): # HİPERPARAMETRE OPT İÇİN KULLANACAĞIMIZ DES FONKSİYONU
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae
# DES İÇERİSİNDE HEM ALFA HEM BETAYI ARIYOR OLACAĞIZ.

alphas = np.arange(0.01, 1, 0.10) # ALFALARI GEZECEĞİ ARALIK
betas = np.arange(0.01, 1, 0.10) # BETALARI GEZECEĞİ ARALIK

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas) # BEST DEĞERLER.




############################
# Final DES Model
############################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha, # TRAIN = 'MUL' ÇARPIMSAL
                                                               smoothing_slope=best_beta) # TREND'I ADDITIVE OLARAK GİRDİK ÇÜNKÜ ADDITIVE BİR MODEL

y_pred = final_des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")


##################################################
# Triple Exponential Smoothing (Holt-Winters)
##################################################

# TES = SES + DES + Mevsimsellik


tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5, # MEVSIMSELLIK 12 YANI 12 AYDA BIR SEZON TAMAMLANIYOR
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)

y_pred = tes_model.forecast(48)
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

############################
# Hyperparameter Optimization
############################

alphas = betas = gammas = np.arange(0.20, 1, 0.10) # ARRAYLERİN HEPSİNE EŞİT

abg = list(itertools.product(alphas, betas, gammas)) # OLASI 3 PARAMETRENİN KOMBİNASYONULARINI GETİRİYOR.


def tes_optimizer(train, abg, step=48): # TES METHODUNU OPTIMIZE ETMEK İÇİN KULLANILAN FONKSİYON
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


############################
# Final TES Model
############################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")








