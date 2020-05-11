import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pymongo import MongoClient

client = MongoClient(port=27017)
db = client.insert_db
print(db)
result = db.collections.find_one({"CityName" : "Beijing","RequiredAttribute" : "PM2.5", "PredictionType" : "Regression"})

X_train = pd.read_csv(result["TrainFileName"], usecols = result["OtherAttributes"])
y_train = pd.read_csv(result["TrainFileName"], usecols = [result["RequiredAttribute"]])

X_test = pd.read_csv(result["TestFileName"], usecols = result["OtherAttributes"])
y_test = pd.read_csv(result["TestFileName"], usecols = [result["RequiredAttribute"]])

model = MLPRegressor(hidden_layer_sizes = (10,10), activation = 'relu')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_test, y_test))
print(mean_absolute_error(y_test, y_pred))
print(sqrt(mean_squared_error(y_test, y_pred)))

y_actual = y_test.values
y_actual = [i for v in y_actual for i in v]

y_obs = [i for i in y_pred]

x_axis = []
for i in range(715):
    x_axis.append(i)

plt.figure()
plt.plot(x_axis, y_actual, color = 'b', label = 'actual')
plt.plot(x_axis, y_pred, color = 'r', label = 'predicted')
plt.xlabel('hour in december')
plt.ylabel('PM 2.5')
plt.legend()
plt.show()