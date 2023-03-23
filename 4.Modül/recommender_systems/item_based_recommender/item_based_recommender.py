 ###########################################
# Item-Based Collaborative Filtering
###########################################

# ÜRÜN BENZERLİKLERİ, İTEM BENZERLİKLERİ AMA BU BENZERLİKLER BEĞENİLME YAPISINA DAİR.

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

# AMACIMIZ BİR FİLM VERİLDİĞİNDE BU FİLME BENZER BEĞENİLME DAVRANIŞI OLAN FİLMLERİ GETİRMEK
# TOPLUM 2 FİLME BENZER REAKSİYONLAR VERMİŞ

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################
# İNDİRGEME İŞLEMİLERİ AŞAMASI. MESELA YORUM SAYISI 1000 DEN AZ OLAN FİLMLER LİSTEDEN ÇIKARTILIR GİBİ

df.head()
df.shape

df["title"].nunique()

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())  # HANGİ FİLME KAÇ TANE YORUM YAPILMIŞ
rare_movies = comment_counts[comment_counts["title"] <= 1000].index # 1000 DEN AZ YORUM ALMIŞ FİLMLER
common_movies = df[~df["title"].isin(rare_movies)] # RARE DIŞINDAKİLERİ ALIRSAM COMMONLARI BULMUŞ OLURUM
common_movies.shape # SADECE 1000DEN AZ YORUM ALAN FİLMLERİ ELEYEREK 300K YORUM ELEDİK
common_movies["title"].nunique()
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating") # KULLANICILARIN FİLMLERE VERDİKLERİ PUANLAR
# BURDA KULLANICI BİR TANE YORUM YAPMIŞ OLDUĞUNDA BİLE DATAFRAME'E GELECEK
user_movie_df.shape
user_movie_df.columns


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################
# FİLMLERİN BEĞENİLME YAPISINA BENZER FİLMLERİN ÖNERİLMESİ KISMI
# KORELASYON KISMI YANİ
movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name] # DEĞİŞKEN SEÇER GİBİ FİLMİ SEÇİYORUM
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10) # YOLLADIĞIM FİLM İLE DİĞER KOLONLARIN KORELASYONUNA BAK


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0] # ELİMİZDEKİ DATAFRAMEDEN BİR FİLM İÇİN RASTGELE FİLM ÖNERİSİ
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df): # BU FİLMİN KEYWORDUNU BARINDIRAN BÜTÜN FİLMLER GELDİ
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df) # İÇİNDE İNSOMNİA OLAN FİLMLERİ GETİRİCEK MESELA


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df(): # BU FONKSİYON DO ONE THİNG PRENSİBİNE AYKIRIDIR
    import pandas as pd # AMA SADECE USER_MOVIE_DF İSTEDİĞİMİZ İÇİN KULLANIP ATMAK DA KABUL EDİLİR.
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df): # BANA FİLM ÖNER FONKSİYONU
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df) #

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





