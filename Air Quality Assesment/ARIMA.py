import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pymongo import MongoClient
from math import sqrt

client = MongoClient(port=27017)
db = client.insert_db
print(db)
result = db.collections.find_one({"CityName" : "Beijing","RequiredAttribute" : "PM2.5", "PredictionType" : "TimeSeries"})

train = pd.read_csv(result["TrainFileName"], usecols = result["Attributes"])

test = pd.read_csv(result["TestFileName"], usecols = result["Attributes"])

train.set_index('datetime', inplace=True)
test.set_index('datetime', inplace=True)

train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)

train = train.values
test = test.values

train = [j for i in train for j in i]
test = [j for i in test for j in i]

train = pd.Series(train)
test = pd.Series(test)

history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

print(mean_absolute_error(test, predictions))
print(sqrt(mean_squared_error(test, predictions)))

plt.figure()
plt.plot(test, color = 'b', label = 'actual')
plt.plot(predictions, color = 'r', label = 'predicted')
plt.xlabel('hour in december')
plt.ylabel('PM 2.5')
plt.legend()
plt.show()