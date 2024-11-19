mport matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load data points
data = pd.read_csv('10-dataset.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

# Preparing and adding 1 to bill
mbill = np.mat(bill)
mtip = np.mat(tip)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

# Set k here
ypred = localWeightRegression(X, mtip, 0.5)
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()

# Load another dataset
dataset = np.genfromtxt('normal_distribution.csv', delimiter=',')

# Mean
# Mean of the third row
print(np.mean(dataset[2]))

# Mean of the last column
print(np.mean(dataset[:, -1]))

# Mean of the intersection of the first 3 rows & first 3 columns
print(np.mean(dataset[0:3, 0:3]))

# Median
# Median of last row
print(np.median(dataset[-1]))

# Median of the last 3 columns
print(np.median(dataset[:, -3:]))

# Median of each row
print(np.median(dataset, axis=1))

# Variance
# Variance of each column
print(np.var(dataset[-24, :2]))

# Standard deviation
# Standard deviation for the dataset
print(np.std(dataset))

# Indexing
# Indexing the 1st row of the dataset (1st row)
print(dataset[0])

# Indexing the last element of the dataset (2nd row)
print(dataset[-1])

# Indexing the first value of the first row (1st row, 1st value)
print(dataset[0, 0])

# Indexing the last value of the second-to-last row
print(dataset[-2, -1])

# Slicing
# Slicing an intersection of the elements (2x2 of the first 2 rows and first 2 columns)
print(dataset[1:3, 1:2])

# Slicing every second element of the 5th row
print(dataset[4, ::2])

# Reversing the entry order, selecting the first 2 rows in reversed order
print(dataset[:2, ::-1])

# Splitting
# Splitting the dataset horizontally into 2 parts
ver_splits = np.vsplit(dataset, 2)

# Requested sub-selection of the dataset, which has only half the amount of rows and columns
print("Dataset shape:", dataset.shape)
print("Subset shape:", ver_splits[0].shape)

# Iterating
# Iterating over the dataset with each value in each row
ever_index = 0
for x in np.nditer(dataset):
    print(x, ever_index)
    ever_index += 1

# Iterating over the whole dataset with index matching the position in the dataset
for index, value in np.ndenumerate(dataset):
    print(index, value)

# Filtering
print(dataset[dataset > 105])
print(np.extract((dataset > 90) & (dataset < 95), dataset))

rows, cols = np.where(np.abs(dataset - 100) < 1)
print([[rows[index], cols[index]] for index in range(len(rows))])

# Sorting
print(np.sort(dataset))
print(np.sort(dataset, axis=0))
index_sorted = np.argsort(dataset[0])
print(dataset[0][index_sorted])

# Combining
thirds = np.hsplit(dataset, 3)
halfed_first = np.vsplit(thirds[0], 2)
print(halfed_first[0])

first_col = np.vstack([halfed_first[0], halfed_first[1]])
print(first_col)

first_second_col = np.hstack([first_col, thirds[1]])
print(first_second_col)

final_combined = np.hstack([first_second_col, thirds[0]])
print(final_combined)

# Reshaping
print(np.reshape(dataset, (1, -1)))
print(dataset.reshape(-1, 2))
