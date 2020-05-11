import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymongo import MongoClient

client = MongoClient(port=27017)
db = client.itsl_db
print(db)
collection = db.itsl_state_collection
print(collection)
result = db.itsl_state_collection.find_one({})
print(result["EntityName"])

def print_heat_map():
	data = pd.read_csv("Data/data_2014.csv")
	cm = data.corr()
	sns.heatmap(cm,square=True)
	plt.yticks(rotation=0)
	plt.xticks(rotation=90)
	plt.show()


if __name__ == "__main__":
	print_heat_map()
