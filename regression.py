import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split   #split training and testing data
from sklearn.feature_extraction import text #text -> feature
from sklearn.metrics import accuracy_score              #to evaluate model (how many good predictions is making)
from sklearn.linear_model import LogisticRegression

data_set_path = "dataset/spam.csv"

#data pre processing
raw = pd.read_csv(data_set_path, encoding="ISO-8859-1")
data = raw.where((pd.notnull(raw)),'')

print(data.shape)