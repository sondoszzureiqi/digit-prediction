import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
from sklearn.model_selection import train_test_split

TrainingData = pd.read_csv('Data/data.csv')
TestData= pd.read_csv('Data/true.csv')
MergeData = pd.merge(TrainingData,TestData, on='ID')

x = MergeData.drop(["ID", 'label'], axis=1)
y = TestData['label']

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.3,random_state=0)

model = LogisticRegression(max_iter=10)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, predictions)

print("accuracy= ", accuracy)

pickle.dump(model, open('model.pkl', 'wb'))
