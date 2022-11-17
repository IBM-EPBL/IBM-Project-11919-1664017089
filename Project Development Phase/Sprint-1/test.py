import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

weather_df = pd.read_csv("static/dataset/weather_train.csv")
print(weather_df.shape)
weather_df.head()


print(weather_df.isnull().sum())

unknown_weather_df = pd.read_csv("static/dataset/weather_test.csv")
print(unknown_weather_df.shape)
unknown_weather_df.head()

