import numpy as np
from tqdm import tqdm
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative(x):
    return x*(1-x)
epoch=5000
lr=10
input_neurons=2
hidden_neurons=3
output_neurons=1
wh=np.random.uniform(size=(input_neurons,hidden_neurons))
bh=np.random.uniform(size=(1,hidden_neurons))
wout=np.random.uniform(size=(hidden_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    hinpl=np.dot(x,wh)
    hinp=hinpl+bh
    hlayer_act=sigmoid(hinp)
    outinpl=np.dot(hlayer_act,wout)
    outinp=outinpl+bout
    output=sigmoid(outinp)
    E0=y-output
    outgrad=derivative(output)
    d_output=E0*outgrad
    EH=d_output.dot(wout.T)
    hiddengrad=derivative(hlayer_act)
    d_hiddenlayer=EH*hiddengrad
    wout+=hlayer_act.T.dot(d_output)*lr
    wh+=x.T.dot(d_hiddenlayer)*lr
print("Input: \n"+str(x))
print("Actual Output: \n"+str(y))
print("Predicted output: \n",output)

----------------------------


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
