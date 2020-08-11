#!/usr/bin/env python
# coding: utf-8
from sklearn import datasets
cancer = datasets.load_breast_cancer()

#Labels for cancer dataset
labels ={
  0: "malignant",
  1: "benign"
}

#Split the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=0)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 0)
rf_model = rf.fit(X_train, y_train)

#Export the model
import pickle
pickle.dump(rf_model, open('model.pkl','wb'))

#Load and test the model
model = pickle.load(open('model.pkl','rb'))
prediction = model.predict(X_test)
print(labels[prediction[0]])