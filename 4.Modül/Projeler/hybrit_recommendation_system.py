import pandas as pd

# dataframeleri oluşturuyoruz.
movie_ = pd.read_csv("4.Modül/Projeler/datasets/movie_lens_dataset/movie.csv")
movie = movie_.copy()
movie.isnull().sum()
movie.info()

rating_ = pd.read_csv("4.Modül/Projeler/datasets/movie_lens_dataset/rating.csv")
rating = rating_.copy()
rating.isnull().sum()
rating.info()

# dataframleri joinliyoruz.
df = rating.merge(movie, on="movieId", how="left")
df.head()

# toplam oy kullanılma sayısı 1000 in altında olan filmleri çıkartalım

comment_counts = df["title"].value_counts()
rare_movies = comment_counts[comment_counts <= 1000].index
comman_movies = df[~df["title"].isin(rare_movies)]

# pivot table oluşturucaz. çünkü user based için gerekli yapı öyle.

user_movie_df = comman_movies.pivot_table(values = "rating", index = "userId", columns="title", aggfunc="mean")
user_movie_df.shape
user_movie_df.columns

# fonksiyonlaştırma

def user_movie_df_func(dataframe):
    comment_counts = dataframe["title"].value_counts()
    rare_movies = comment_counts[comment_counts <= 1000].index
    comman_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = comman_movies.pivot_table(values="rating", index="userId", columns="title", aggfunc="mean")

    return user_movie_df

user_movie_df = user_movie_df_func(df)
user_movie_df.shape()

# rastgele user_id

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state = 35).values)

# seçilen kullanıcıya ait gözlem birimleri
random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()

# izlenenen filmleri dataframeden alıcaz

movies_watched_df = user_movie_df[(movies_watched)]

# her bir user ın bu filmlerden kaçını izlediği bilgisini taşıyan df

user_movie_count = movies_watched_df.T.notna().sum()
user_movie_count=user_movie_count.reset_index()

user_movie_count.columns= ["userId", "movie_counts"]

# seçilen userla ortak yüzde 60 veya daha fazla film izleyenler

user_same_movies = user_movie_count[user_movie_count["movie_counts"] > len(movies_watched) * 6 / 10]["userId"]

# kullacılar ile random userı aynı df içerisine alıyoruz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(user_same_movies)],
                      random_user_df[movies_watched]])

# userlar arasındaki korelasyona bakıcaz

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

# korelasyonu yüksek olan top kullanıcıları alıyoruz.
corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']#DF İN İNDEXLERİNİ DÜZENLİYORUZ

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] > 0.65)][["user_id_2","corr"]].reset_index(drop = True)
top_users.columns = ["userId", "corr"]

# rating dataframei ile birleştiriyoruz.
top_user_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how = "inner")

# ratingleri korelasyon ile çarparak ortadaki ölçüm sorununu ortadan kaldırıyoruz.
top_user_ratings["weighted score"] = top_user_ratings["corr"] * top_user_ratings["rating"]

#film id lere göre ortalama weighted scoreları buluyoruz.
recommendation_df = top_user_ratings.groupby("movieId").agg({"weighted score": "mean"}).sort_values(by="weighted score", ascending=False)

recommendation_df = recommendation_df.reset_index()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted score"] > 3.5].sort_values("weighted score", ascending=False)

# filmlerin adını getiriyoruz.

movies_to_be_recommend.merge(movie[["movieId", "title"]])

# önerilecek ilk 5 film

movies_to_be_recommend.merge(movie[["movieId", "title"]])[0:5]

###### ITEM BASED RECOMMANDATION ######

# kullanıcının en güncel 5 puan verdiği filme benzer beğenilme yapısındaki film önerilecke
movie_ = pd.read_csv("4.Modül/Projeler/datasets/movie_lens_dataset/movie.csv")
movie = movie_.copy()

rating_ = pd.read_csv("4.Modül/Projeler/datasets/movie_lens_dataset/rating.csv")
rating = rating_.copy()

df = rating.merge(movie, on="movieId", how="left")
df.head()

rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)

# user_movie_df üzerinden filmi bulucaz
user_movie_df
# df üzerindeki kolonlar title yerine id olarak değiştiriyorum.
user_movie_df = comman_movies.pivot_table(values = "rating", index = "userId", columns="movieId", aggfunc="mean")

movie = user_movie_df[923]

# en önerilebilir 5 film
user_movie_df.corrwith(movie).sort_values(ascending=False)[1:].head(5)
