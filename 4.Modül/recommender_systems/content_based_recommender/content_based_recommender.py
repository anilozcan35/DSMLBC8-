#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################


# METİN VEKTÖRLEŞTİRİCEZ VE BENZERLİKLERİ HESAPLICAZ.
# VE BU VEKTÖRLERİN BİRBİRİNE UZAKLIKLARINA BAKICAZ COSİNE SİMİLARİTY İLE BİRLİKTE.
# YANİ FİLMLERİN AÇIKLAMALARINA GÖRE ÖNERİ YAPICAZ.


#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması # METİNLERİ VEKTÖRLEŞTİRMEK İÇİN KULLANILAN YÖNTEM
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("4.Modül/recommender_systems/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english") # TFIDF METHODU STOPWORDSLERİ ÇIKARTIYORUZ. METHOD İLE BİRLİKTE
# SANIRIM NESNE OLUŞTURUYORUZ

# df[df['overview'].isnull()] # OVERWİEV İÇERİSİNDE BOŞ DEĞERLER
df['overview'] = df['overview'].fillna('') # BOŞLUK ATIYORUZ

tfidf_matrix = tfidf.fit_transform(df.loc[0:20000,'overview']) # FİT UYGULAMADIR. TRANSFORM İSE ESKİYİ BU FİTE ÇEVİRİR.

tfidf_matrix.shape

df['title'].shape

tfidf.get_feature_names() # KELİMELERİN TAMAMI

tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape # MATRİX HER BİR FİLM ÇİFTİ ARASINDAKİ SİMİLARİTY SCORU E VERİR
cosine_sim[1] # BU ŞEKLİYLE BİRAZ OKUNMASI ZOR


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

indices = pd.Series(df.index, index=df['title']) # İNDEXLERE FİLMLERİN ADINI YAZIYORUZ

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='last')] # DAHA ÖNCE 11 KERE ÇEKİLMİŞ FİLM AMA BEN 11. Yİ İSTİYORUM ÇÜNKÜ EN GÜNCEL O

indices["Cinderella"] # İNDEX BİLGİSİNİ TEK HALE GETİRDİK

indices["Toy Story"]

movie_index = indices["Toy Story"] # TOY STORY İLE DİĞER FİLMLER ARASINDA SİMİLARİTY

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"]) # TOY STORY İLE BENZER TÜM FİLMLERİN SKORU

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index # 0. GÖZLEMDE FİLMİN KENDİSİ VAR

df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
