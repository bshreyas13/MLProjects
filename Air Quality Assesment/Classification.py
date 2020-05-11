import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import collections
from pymongo import MongoClient

client = MongoClient(port=27017)
db = client.insert_db
print(db)
result = db.collections.find_one({"CityName" : "Beijing","RequiredAttribute" : "PM2.5", "PredictionType" : "Classification"})

X_train = pd.read_csv(result["TrainFileName"], usecols = result["OtherAttributes"])
y_train = pd.read_csv(result["TrainFileName"], usecols = [result["RequiredAttribute"]])
y_train = y_train.values

X_test = pd.read_csv(result["TestFileName"], usecols = result["OtherAttributes"])
y_test = pd.read_csv(result["TestFileName"], usecols = [result["RequiredAttribute"]])
y_test = y_test.values

y_train_binary = []
for i in y_train:
    for j in i:
        if j <= 30:
            y_train_binary.append(1)
        elif j > 30 or j <= 60:
            y_train_binary.append(2)
        elif j > 60 or j <= 90:
            y_train_binary.append(3)
        elif j > 90 or j <= 120:
            y_train_binary.append(4)
        elif j > 120 or j <= 250:
            y_train_binary.append(5)
        elif j > 250:
            y_train_binary.append(6)

y_test_binary = []
for i in y_test:
    for j in i:
        if j <= 30:
            y_test_binary.append(1)
        elif j > 30 or j <= 60:
            y_test_binary.append(2)
        elif j > 60 or j <= 90:
            y_test_binary.append(3)
        elif j > 90 or j <= 120:
            y_test_binary.append(4)
        elif j > 120 or j <= 250:
            y_test_binary.append(5)
        elif j > 250:
            y_test_binary.append(6)

count1 = collections.Counter(y_train_binary)
count2 = collections.Counter(y_test_binary)

model = MLPClassifier(hidden_layer_sizes = (10,5), activation = 'tanh')

model.fit(X_train, y_train_binary)
y_pred = model.predict(X_test)

print("MLP Classifier")
print(confusion_matrix(y_test_binary, y_pred))
print(model.score(X_test, y_test_binary))
print(precision_score(y_test_binary, y_pred))
print(recall_score(y_test_binary, y_pred))
print(f1_score(y_test_binary, y_pred))
print("\n")


model = DecisionTreeClassifier()

model.fit(X_train, y_train_binary)
y_pred = model.predict(X_test)

print("Decision Tree Classifier")
print(confusion_matrix(y_test_binary, y_pred))
print(model.score(X_test, y_test_binary))
print(precision_score(y_test_binary, y_pred))
print(recall_score(y_test_binary, y_pred))
print(f1_score(y_test_binary, y_pred))
print("\n")


model = KNeighborsClassifier(n_neighbors = 10)

model.fit(X_train, y_train_binary)
y_pred = model.predict(X_test)

print("KNN Classifier")
print(confusion_matrix(y_test_binary, y_pred))
print(model.score(X_test, y_test_binary))
print(precision_score(y_test_binary, y_pred))
print(recall_score(y_test_binary, y_pred))
print(f1_score(y_test_binary, y_pred))
print("\n")


model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train_binary)
y_pred = model.predict(X_test)

print("Random Forest Classifier")
print(confusion_matrix(y_test_binary, y_pred))
print(model.score(X_test, y_test_binary))
print(precision_score(y_test_binary, y_pred))
print(recall_score(y_test_binary, y_pred))
print(f1_score(y_test_binary, y_pred))
print("\n")

