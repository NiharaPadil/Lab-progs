import pandas as pd
from sklearn .preprocessing import MinMaxScaler

data=pd.read_csv('Student.csv')

print(data)

data=data.dropna()

data=data.drop_duplicates(subset='Name')

print('\n',data)

scaler=MinMaxScaler()

scaled_data = scaler.fit_transform(data[['Age','GPA']])

data[['Age','GPA']] = scaled_data

data['Student_Info'] = data['Name'] + '(' + data['Grade'] + ')'

print('\n',data.head())
