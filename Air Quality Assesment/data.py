import pandas as pd
import os

data_his = pd.read_csv("Data/his_data_2014.csv", encoding = 'unicode_escape', usecols = ['Date (LST)', 'Value'])

data_his['Date (LST)'] =  pd.to_datetime(data_his['Date (LST)'], format='%m/%d/%Y %H:%M')
data_his = data_his.convert_objects(convert_numeric=True)
data_his.columns = ['datetime', 'PM2.5']
print(data_his)
print(data_his.dtypes)


with open("Data/met_data_2014.csv") as f:
    content = f.readlines()

content = [x.strip().replace('"','').replace(',','').split(';') for x in content]
content.pop(0)

for i in range(len(content)):
    if content[i][11] == "10.0 and more":
        content[i][11] = "10.0"

content.reverse()

data_met = pd.DataFrame(content, columns=['datetime', 'T', 'P0', 'P', 'U', 'DD', 'FF', 'ff10', 'WW', 'WW_', 'c', 'VV', 'Td', 'haha'])
data_met.drop(['haha'], axis = 1, inplace = True)
data_met.drop(['DD', 'FF', 'ff10', 'WW', 'WW_', 'c'], axis = 1, inplace = True)
data_met['datetime'] =  pd.to_datetime(data_met['datetime'], format='%d.%m.%Y %H:%M')
data_met = data_met.convert_objects(convert_numeric=True)
data_met = data_met[data_met['datetime'].dt.minute == 0]
print(data_met)
print(data_met.dtypes)

data_final = pd.merge(data_met, data_his, on='datetime')
data_final = data_final[data_final['PM2.5'] != -999]
data_final.dropna()
print(data_final)

data_final.to_csv('Data/data_2014.csv', index = False)

def split_data():
    if not os.path.exists("Data/Train"):
        os.makedirs("Data/Train")
    if not os.path.exists("Data/Test"):
        os.makedirs("Data/Test")
    
    data_train = data_final[data_final['datetime'].dt.month != 12]
    data_test = data_final[data_final['datetime'].dt.month == 12]
    print(data_train.shape)
    print(data_test.shape)
    
    data_train.to_csv("Data/Train/train_data_2014.csv", index = False)
    data_test.to_csv("Data/Test/test_data_2014.csv", index = False)

split_data()