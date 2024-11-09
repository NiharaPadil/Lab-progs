import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

 

def kernel(point, xmat, k):

    m,n = np.shape(xmat)

    weights = np.mat(np.eye((m)))

    for j in range(m):

        diff = point - X[j]

        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))

    return weights

 

def localWeight(point, xmat, ymat, k):

    wei = kernel(point,xmat,k)

    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))

    return W

    

 

def localWeightRegression(xmat, ymat, k):

    m,n = np.shape(xmat)

    ypred = np.zeros(m)

    for i in range(m):

        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)

    return ypred

    

# load data points

data = pd.read_csv('10-dataset.csv')

bill = np.array(data.total_bill)

tip = np.array(data.tip)

 

#preparing and add 1 in bill

mbill = np.mat(bill)

mtip = np.mat(tip)

 

m= np.shape(mbill)[1]

one = np.mat(np.ones(m))

X = np.hstack((one.T,mbill.T))

 

#set k here

ypred = localWeightRegression(X,mtip,0.5)

SortIndex = X[:,1].argsort(0)

xsort = X[SortIndex][:,0]

 

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.scatter(bill,tip, color='green')

ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=5)

plt.xlabel('Total bill')

plt.ylabel('Tip')

plt.show();

Import numpy as mp

Dataset=np.genfromtxt(‘c:/users/Sahyadri/downloads/normal distribution.csv:delimiter=’,’)

 

#Mean

#np.mean of third now

np.mean(dataset[2])

 

#mean of the last column

Np.mean(dataset[:-1])

 

#mean of the insersection of first 3 rows & first 3 columns

Np.mean(dataset[0:3,0:3])

# median

#Median of last row

Np.median(dataset[-1])

#median of last 3 columns

Np.median(dataset[:,-3])

 

# median of each row

Np.median(dataset ,axis=1)

 

#varience

#varience of each column

Np.var(dataset[-26,:2]

 

#standard deviation

#standard deviation for the dataset

np.std(dataset)

 

#Indexing

Import numpy as np

Dataxt=1 nnp.genfromtxt(‘c:user/admin/normal_distribution_spittable-csv’,\delimeter=’,’)

#indexing the 1st row of  the dataset (1st row)

Dataset[-1]

 

#indexing the last  element if the dataset (2nd row)

Dataset[-1]

#indexing the first value of the firs row(1st row,1st value)

Dataset[0][0]

#indexing the last value of the second to last now (we want to usethe combined acess syntax here)

 

#slicing

#slicing an intersection of the elements (2*2 of the first 2 rows and 1st 2 columns

Dataset[1:3,1:2]

#slicing every second element of 5th row

Dataset[4:2]

 

#reversing the entry order, selecting the first 2 rows in reversed order dataset[-1,::-1]

 

#splitting

#splitting  our dataset horizontally on inorder 2

Ver_splits=np.vsplit(hur-splits[0],(2,))

 

 

#requested sub selection of our dataset which has only half the amount of rows and column

Print(“Dataset”,dataset-sp shape)

Print(“subset”,ver_splits[0],shape)

 

#Iterating

#Iterating over dataset c each value in each now)

Ever_index=0

For x in np.mditer(dataset)

Print (x,ever-index)

curv-index+=1

# iterating over whole dataset with index matching the position in the dataset.

For index,value in mp.ndenumerats(dataset):

Print(index,value)

 

#Filtering

Dataset [dataset>105]

np.extract((dataset>90)&(dataset<95),dataset)

rows.cols=np.where labs(datset-100)<1)

[[rows[index],cols[index]]for(index]]for(index,-)in np.ndenemeterate(rows)]

 

 

#sorting

np.sort(dataset)

np.sort(dataset,axis=0)

index_sorted=np.argsort(dataset{0])

dataset[0][index_sorted]

 #Combining

Thirds=np.hspirit(dataset,(3))

Halfed_first=np.vsplit(thirds[0],(2))

halfed_first[0]

first_col=np.vstack([halfed_first[0],halfed_first[1]])

first_col

 

first_seond_col=np.hstack([first_col,thirds[1]])

first_second_col

 

np.hstack([first_second_col,thirds[0])

#Reshaping

np.reshape(dataset,(1,-1)

dataset.reshape(-1,2)

 
