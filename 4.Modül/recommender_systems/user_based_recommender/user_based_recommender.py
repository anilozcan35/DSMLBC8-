############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df(): # USER LARIN MOVILERE NE VERDİĞİNİ BARINDIRAN DATAFRAMEİ OLUŞTURAN FONKSŞYON, ITEM BASED İÇERİSİNMDE BU KISIMLAR YAPILDI
    import pandas as pd
    movie = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values) # USERLARDAN ÖRNEKLEM ALIYORUZ


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user] # VERİ SETİNİ USER BAZINDA İNDİRGİYORUZ

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist() # KULLANICININ İZLEDİĞİ FİLMLERİ GETİRİR

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"] # TEYİT ETMEK İÇİN BİR BAKALIM


len(movies_watched) # İZLEDİĞİ TOPLAM FİLM SAYISI



#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

movies_watched_df = user_movie_df[movies_watched] # VERİ SETİNİ İZLENEN FİLMLER ÜZERİNDEN İNDİRGİYORUZ

user_movie_count = movies_watched_df.T.notnull().sum() #HER BİR KULLANICININ BU DATAFRAME'E GÖRE KAÇ TANE FİLM İZLEDİĞİNİ HESPALICAZ.
#TRANPOZE EDİP USERLARI KOLON OLARAK ALIYORUZ, VE FİLMLER NAN DEĞİLSE TOPLUYORUZ.

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False) #20 DEN FAZLA FİLM İZLEMİŞLER VE SİNANLA ORTAK

user_movie_count[user_movie_count["movie_count"] == 33].count() # SİNANIN İZLEDİĞİ TÜM FİLMLERİ İZLEYEN KULLANICI SAYISI


users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"] #SİNAN İLE EN AZ 20 TANE FİLM İZLEMİŞ KULLANICILAR.

# YÜZDE 60 I EN AZINDAN SİNANLA AYNI FİLMİ İZLEMİŞ OLSUN
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]]) # SİNAN VE DİĞER KULLANICILARIN İZLEDİĞİ FİLMLERİ VE VERDİĞİ PUANLARI ALT ALTA EKLİYORUZ

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates() # TÜM KULLANICLAR ARASINDAKİ KORELASYONU OLUŞTURUYORUZ

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']#DF İN İNDEXLERİNİ DÜZENLİYORUZ

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True) # SİNANLA 0.65 TEN FAZLA KORELE OLAN USERLAR

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


rating = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/rating.csv') # SİNANLA KORELE ÇIKAN USERLAR FİLMLERE KAÇ PUAN VERDİ
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner') # KORELASYONU YÜKSEK KULLANICLARIN FİLMLERE VERDİĞİ PUANLAR

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user] #SİNANI KORELASYONDAN ÇIKARTIYORUZ KENDİSİYLE KORELASYONU 1 GELİYOR


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################
# KULLANICININ VERDİĞİ PUANI MI GÖZ ÖNÜNDE BULUNDURMALI YA DA KORELASYONU MU GÖZ ÖNÜNDE BULUNDURMALI
#  SİNANLA BÜTÜN USERLAR AYNI ORANDA KORELASYONLU DEĞİL

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating'] # KORELASYONLA RATINGİ ÇARPTIK KULLANICILARIN RATINGLERINI KORELASYONA GORE DUZENLEDIK

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}) # FİLMLERE GÖRE GRUPBY YAPIP ORTALAMASINI ALICAZ HEM KORELASYON HEM PUANI ÖNERİ İÇİN İŞLEME ALMIŞ OLUCAZ

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False) # SİNANIN 3.5 ÜZERİNDE PUAN VERMESİ MUHTEMEL FİLMLER

movie = pd.read_csv("4.Modül/recommender_systems/datasets/movie_lens_dataset/movie.csv")
movies_to_be_recommend.merge(movie[["movieId", "title"]]) # ÖNERİCEĞİM FİLMLERİN İSİMLERİNİ BULUYORUM



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


