import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define kernel function for local weighted regression
def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    return weights

# Define function for local weighted regression
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T))
    return W

# Define function for performing local weighted regression on the dataset
def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load data points
data = pd.read_csv('10-dataset.csv')  # Ensure you have the correct path to this CSV
bill = np.array(data.total_bill)
tip = np.array(data.tip)

# Preparing and adding 1 to bill
mbill = np.mat(bill)
mtip = np.mat(tip)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

# Set k for local weighted regression
ypred = localWeightRegression(X, mtip, 0.5)

# Sorting for smooth plotting
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]

# Plotting the data and the regression line
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()

# Load another dataset for statistical operations
dataset = np.genfromtxt('normal_distribution.csv', delimiter=',')
print("Dataset shape:", dataset.shape)  # Print the shape of the dataset

# Mean calculations
print("Mean of the third row:", np.mean(dataset[2]))  # Mean of the third row
print("Mean of the last column:", np.mean(dataset[:, -1]))  # Mean of the last column
print("Mean of the intersection of the first 3 rows & columns 1-3:", np.mean(dataset[0:3, 0:3]))  # Mean of the first 3 rows and 3 columns

# Median calculations
print("Median of the last row:", np.median(dataset[-1]))  # Median of the last row
print("Median of the last 3 columns:", np.median(dataset[:, -3:]))  # Median of the last 3 columns
print("Median of each row:", np.median(dataset, axis=1))  # Median of each row

# Variance calculation
print("Variance of the first two elements in the last row:", np.var(dataset[-1, :2]))  # Variance of the first two elements in the last row

# Standard deviation
print("Standard deviation for the dataset:", np.std(dataset))  # Standard deviation of the dataset

# Indexing
print("First row of the dataset:", dataset[0])  # First row
print("Last row of the dataset:", dataset[-1])  # Last row
print("First value of the first row:", dataset[0, 0])  # First value of the first row
print("Last value of the second-to-last row:", dataset[-2, -1])  # Last value of the second-to-last row

# Slicing
print("Sliced (2x2) of the first 2 rows and first 2 columns:", dataset[1:3, 1:2])  # Slicing first 3 rows and 2 columns
print("Every second element of the 5th row:", dataset[4, ::2])  # Every second element of the 5th row
print("First 2 rows in reversed order:", dataset[:2, ::-1])  # Reversing the entry order for the first 2 rows

# Splitting
# Split the dataset vertically into 2 parts
ver_splits = np.vsplit(dataset, 2)
print("Shape of dataset:", dataset.shape)
print("Shape of subset after vertical split:", ver_splits[0].shape)

# Iterating
ever_index = 0
for x in np.nditer(dataset):
    print(x, ever_index)
    ever_index += 1

# Iterating with index matching the position
for index, value in np.ndenumerate(dataset):
    print(index, value)

# Filtering: Print values greater than 105
print("Values greater than 105:", dataset[dataset > 105])

# Extract values between 90 and 95
print("Values between 90 and 95:", np.extract((dataset > 90) & (dataset < 95), dataset))

# Find the rows and columns where the absolute difference from 100 is less than 1
rows, cols = np.where(np.abs(dataset - 100) < 1)
print("Positions where abs(dataset - 100) < 1:", [[rows[index], cols[index]] for index in range(len(rows))])

# Sorting
print("Sorted dataset (flattened):", np.sort(dataset))
print("Sorted dataset by columns:", np.sort(dataset, axis=0))

# Sorting a row (0th row in this case)
index_sorted = np.argsort(dataset[0])
print("Sorted first row:", dataset[0][index_sorted])

# Combining: Splitting and recombining
thirds = np.array_split(dataset, 3, axis=1)  # Split into 3 parts (columns)
halfed_first = np.vsplit(thirds[0], 2)  # Split the first third vertically into 2
print("First half of the first third:", halfed_first[0])

first_col = np.vstack([halfed_first[0], halfed_first[1]])  # Vertically stack the halves
print("First column after vstack:", first_col)

# Horizontal stacking
first_second_col = np.hstack([first_col, thirds[1]])  # Horizontal stack first_col with the second third
print("First and second columns after hstack:", first_second_col)

final_combined = np.hstack([first_second_col, thirds[2]])  # Combine with the third third
print("Final combined columns:", final_combined)

# Reshaping
print("Reshaped dataset (1D):", np.reshape(dataset, (1, -1)))  # Reshape to 1D array
print("Reshaped dataset to (-1, 2):", dataset.reshape(-1, 2))  # Reshape to (-1, 2)
