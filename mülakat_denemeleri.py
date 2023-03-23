#############################################
import pandas as pd
import seaborn as sns
import matplotlib as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df.value_counts()

df['sex'].value_counts().plot(kind='bar')
plt.show()

df.groupby(["embark_town","embarked"]).agg(["count","mean","max"])["fare"]