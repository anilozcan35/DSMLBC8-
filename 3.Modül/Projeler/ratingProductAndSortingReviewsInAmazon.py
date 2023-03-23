import math
import scipy.stats as st
import pandas as pd
import datetime as dt

df_ = pd.read_csv("3.Modül/Projeler/amazon_review.csv")
df = df_.copy()

df.head(500)

# ürünün ortalama puanı

df["overall"].mean()

# Ürünün tarihe göre ağırlıklı ortalaması.
df.info()

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()

df["gunlerfarkı"] = (current_date - df["reviewTime"]).dt.days

df["gunlerfarkı"].describe().T

df.loc[df["gunlerfarkı"] <= 280, "overall"].mean() * 35 / 100 + \
df.loc[(df["gunlerfarkı"] <= 430) & (df["gunlerfarkı"] > 280), "overall"].mean() * 30/100 + \
df.loc[(df["gunlerfarkı"] <= 600) & (df["gunlerfarkı"] > 430), "overall"].mean() * 20/100 + \
df.loc[(df["gunlerfarkı"] <= 1063), "overall"].mean() * 15/100

# ürün geçmişe göre daha iyi skorlar alıyor bunu time - based ortalama ile yansıtıyoruz.

# helpful_no değişkenini üretelim
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(up, down): # YORUMLARI SIRALAMAK İÇİN YANLILIĞA ÇOK AÇIKTIR. ÇOĞU YERDE YANLIŞ KULLANILIR.
    return up - down

def score_average_rating(up, down): # AZ ÖNCEKİ YANLILIĞA ÇÖZÜM GETİRMEK İÇİN
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# skorların arasındaki negatif ve pozitiflikleri bulmak için
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],x["helpful_no"]), axis = 1)

# up / up + down
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]), axis = 1)

# istatistiksel olarak
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"], confidence= 0.95), axis = 1)

df.sort_values(by = "wilson_lower_bound" , ascending= False).head(50)