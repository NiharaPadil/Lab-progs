import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def kernel(point, xmat, tau):
    weights = np.mat(np.eye(len(xmat)))
    for i in range(len(xmat)):
        diff = point - xmat[i]
        weights[i, i] = np.exp(diff * diff.T / (-2.0 * tau**2))
    return weights
def locally_weighted_regression(xmat, ymat, tau):
    y_pred = np.zeros(len(xmat))
    for i in range(len(xmat)):
        weights = kernel(xmat[i], xmat, tau)
        theta = (xmat.T * (weights * xmat)).I * (xmat.T * (weights * ymat.T))
        y_pred[i] = xmat[i] * theta
    return y_pred
# Load and preprocess data
data = pd.read_csv('10-dataset.csv')
bill = np.array(data['total_bill'])
tip = np.array(data['tip'])
X = np.hstack((np.ones((len(bill), 1)), bill.reshape(-1, 1)))
y = tip
# Perform Locally Weighted Regression
tau = 0.5
ypred = locally_weighted_regression(np.mat(X), np.mat(y), tau)
# Plot
sorted_indices = X[:, 1].argsort()
plt.scatter(bill, tip, color='blue', label='Data Points')
plt.plot(X[sorted_indices, 1], ypred[sorted_indices], color='red', label='LWR Prediction', linewidth=2)
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend()
plt.title('Non-Parametric Locally Weighted Regression')
plt.show()
