import pandas as pd
data = {
    'Name':['John','Emma','Sant','Lisa','Tom'],
    'Age':[25,30,28,32,27],
    'Country' : ['USA','Canada','India','UK','Australia'],
    'Salary':[50000,60000,70000,80000,65000]
}
df = pd.DataFrame(data)
print("Original DataFrame")
print(df)

name_age=df[['Name','Age']]
print("Original Data")
print(df)

name_age=df[['Name','Age']]
print("Name and age columns")
print(name_age)

filtered_df=df[df['Country']=='USA']
print(filtered_df)

sorted_df = df.sort_values("Salary",ascending=False)
print("\n Sorted DataFrame(by salary in descending order)")
print(sorted_df)

average_Salary = df['Salary'].mean()
print("\n Average Salary",average_Salary)

df['Experience']=[3,6,4,8,5]
print('DataFrame with addded experience')
print(df)

df.loc[df['Name']=='Emma','Salary']=65000
print("\n DataFrame with updating emma salary")
print(df)

df=df.drop('Experience',axis=1)
print("\n DataFrame after deleting the column")
print(df)
