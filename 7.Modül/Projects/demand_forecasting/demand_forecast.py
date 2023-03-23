import lightgbm as lgm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

test = pd.read_csv("7.Modül/Projects/demand_forecasting/test.csv")
train = pd.read_csv("7.Modül/Projects/demand_forecasting/train.csv")

test.head()
train.head()

df = pd.concat([train, test], sort= False)
df.head()