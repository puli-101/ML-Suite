import pandas as pd
import numpy as np

#split training and testing data
from sklearn.model_selection import train_test_split   
#text -> feature
from sklearn.feature_extraction.text import TfidfVectorizer
#to evaluate model (how many good predictions is making)
from sklearn.metrics import accuracy_score              
from sklearn.linear_model import LogisticRegression

data_set_path = "dataset/spam.csv"

#data pre processing
raw = pd.read_csv(data_set_path, encoding="ISO-8859-1")
data = raw.where((pd.notnull(raw)),'')

#Relabeling (Encoding)
#spam -> 0
#ham -> 1
data.loc[data['type'] == 'spam', 'type',] = 0
data.loc[data['type'] == 'ham', 'type',] = 1

X = data['message'] 
Y = data['type'].astype('int')

#splitting data into training & test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2) #random_state=1

#Feature extraction
# transform text data to feature vectors for logistic regression
feat_extractor = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_feat = feat_extractor.fit_transform(X_train)
X_test_feat = feat_extractor.transform(X_test)

#Training the ML model - Logistic Regression
model = LogisticRegression()
model.fit(X_train_feat, Y_train)

#Model evaluation
prediction = model.predict(X_train_feat)
accuracy = accuracy_score(Y_train, prediction)
print("Accuracy on training : ",accuracy,"%")

prediction = model.predict(X_test_feat)
accuracy = accuracy_score(Y_test, prediction)
print("Accuracy on testing : ",accuracy,"%")

#Construction of predictive system
input = ["Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet."]
# extract vector from text
input_feat = feat_extractor.transform(input)
# make prediction
predictions = model.predict(input_feat)

for prediction in predictions:
    if prediction == 0:
        print("Spam")
    else:
        print("Ham")
