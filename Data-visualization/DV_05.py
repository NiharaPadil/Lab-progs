import matplotlib.pyplot as plt
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

data = pd.read_csv('10-dataset.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)
mtip = np.mat(tip)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

ypred = localWeightRegression(X, mtip, 0.5)

SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()

dataset = np.genfromtxt('normal_distribution.csv', delimiter=',')
print("Dataset shape:", dataset.shape)

print("Mean of the third row:", np.mean(dataset[2]))
print("Mean of the last column:", np.mean(dataset[:, -1]))
print("Mean of the intersection of the first 3 rows & columns 1-3:", np.mean(dataset[0:3, 0:3]))

print("Median of the last row:", np.median(dataset[-1]))
print("Median of the last 3 columns:", np.median(dataset[:, -3:]))
print("Median of each row:", np.median(dataset, axis=1))

print("Variance of the first two elements in the last row:", np.var(dataset[-1, :2]))

print("Standard deviation for the dataset:", np.std(dataset))

print("First row of the dataset:", dataset[0])
print("Last row of the dataset:", dataset[-1])
print("First value of the first row:", dataset[0, 0])
print("Last value of the second-to-last row:", dataset[-2, -1])

print("Sliced (2x2) of the first 2 rows and first 2 columns:", dataset[1:3, 1:2])
print("Every second element of the 5th row:", dataset[4, ::2])
print("First 2 rows in reversed order:", dataset[:2, ::-1])

ver_splits = np.vsplit(dataset, 2)
print("Shape of dataset:", dataset.shape)
print("Shape of subset after vertical split:", ver_splits[0].shape)

ever_index = 0
for x in np.nditer(dataset):
    print(x, ever_index)
    ever_index += 1

for index, value in np.ndenumerate(dataset):
    print(index, value)

print("Values greater than 105:", dataset[dataset > 105])

print("Values between 90 and 95:", np.extract((dataset > 90) & (dataset < 95), dataset))

rows, cols = np.where(np.abs(dataset - 100) < 1)
print("Positions where abs(dataset - 100) < 1:", [[rows[index], cols[index]] for index in range(len(rows))])

print("Sorted dataset (flattened):", np.sort(dataset))
print("Sorted dataset by columns:", np.sort(dataset, axis=0))

index_sorted = np.argsort(dataset[0])
print("Sorted first row:", dataset[0][index_sorted])

thirds = np.array_split(dataset, 3, axis=1)
halfed_first = np.vsplit(thirds[0], 2)
print("First half of the first third:", halfed_first[0])

first_col = np.vstack([halfed_first[0], halfed_first[1]])
print("First column after vstack:", first_col)

first_second_col = np.hstack([first_col, thirds[1]])
print("First and second columns after hstack:", first_second_col)

final_combined = np.hstack([first_second_col, thirds[2]])
print("Final combined columns:", final_combined)

print("Reshaped dataset (1D):", np.reshape(dataset, (1, -1)))
print("Reshaped dataset to (-1, 2):", dataset.reshape(-1, 2))
