#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('4.Modül/recommender_systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
# İNDİRGİYORUZ
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5)) # READERA 1 VE 5 ARASINDA TANIMLIYORUZ. ÇÜNKÜ DATAMIZ ÖYLE

data = Dataset.load_from_df(sample_df[['userId', # UNUTMA MATRİX FACTORİZATION ZATEN FİLM ÖNERMEK İÇİN BULUNMUŞ BİR METHOD
                                       'movieId',
                                       'rating']], reader) # SUPRISE KUTUPHANESININ KENDİ İSTEDİĞİ HALE GETİRDİK VERİ YAPISINI

##############################
# Adım 2: Modelleme
##############################
# MODELİ EĞİTİM SETİNDE KURUP SONRA TEST EDERİZ.

trainset, testset = train_test_split(data, test_size=.25) # TEST VE TRAIN OLMAK ÜZERE DATAYI 2YE BÖLÜYORUZ.
svd_model = SVD() # MATRIX FACTORIZATION
svd_model.fit(trainset) # TRAIN SETİNİN ÜZERİNDEN ÖĞRENİYOR
predictions = svd_model.test(testset) # TEST SETİ ÜZERİNDE TEST YAP

accuracy.rmse(predictions) # ROOT MEAN SQUARE ERROR
# TAHMİN ESNASINDA ORTALAMA HATAM


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################
# MODELİN TAHMİN PERFORMANSINI ARTIRMAYA ÇALIŞMAK. HİPERPARAMETRELERİ NASIL OPTİMİZE EDİCEZ.
param_grid = {'n_epochs': [5, 10, 20], # İTERASYON SAYISI
              'lr_all': [0.002, 0.005, 0.007]} # LEARNİNG RATE

# DENEME YANILMA YAPICAK GRIDSEARCH SVD MODELİ ÜZERİNDE
gs = GridSearchCV(SVD,
                  param_grid, # PARAMETRE ÇİFTLERİNİ TEK TEK DENE
                  measures=['rmse', 'mae'], # HATANI NASIL DEĞERLENDİRMEK İSTERSİN
                  cv=3, # 3 KATLI ÇAPRAZ DOĞRULAMA. VERİ SETİNİ 3 E BÖL 2 SİNDEN ÖĞREN 1İNDE TEST ET. 3Ü İÇİN YAP VE ORTALAMASINI GETİR.
                  n_jobs=-1, # İŞLEMCİLERİ FULL PERFORMANS KULLAN
                  joblib_verbose=True) # BANA RAPORLAMA YAP

gs.fit(data)

gs.best_score['rmse'] # BEST SCORU ALABİLMEK İÇİN
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse']) # SVD NİN İÇERİSİNE KEYWORDED ARGUMANLARI GİRİYORUZ

data = data.build_full_trainset() # BÜTÜN VERİ SETİNİ TRAİNE ALDIM
svd_model.fit(data) # MODELİ FİT EDİYORUZ

svd_model.predict(uid=1.0, iid=541, verbose=True)






