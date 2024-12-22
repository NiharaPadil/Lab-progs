#1a
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mean, median, variance, std_dev = np.mean(data), np.median(data), np.var(data), np.std(data)

print("mean:", mean, "median:", median, "variance:", variance, "std_dev:", std_dev)
print("element at index 2:", data[2])
print("sliced data (index 2 to 5):", data[2:6])
print("split data:", np.array_split(data, 2))

print("iterating over elements:")
[print(element) for element in data]

filter_data = data[data > 5]
print("filtered data:", filter_data)
print("sorted data:", np.sort(data))

additional_data = np.array([11, 12, 13])
combined_data = np.concatenate((data, additional_data))
print("combined data:", combined_data)
print("reshaped data (2x5):", data.reshape(2, 5))


#1b

import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mean, median, variance, std_dev = data.mean(), data.median(), data.var(), data.std()

print(f"mean: {mean}, median: {median}, variance: {variance}, std_dev: {std_dev}")
print("element at index 2:", data[2])
print("sliced data (index 2 to 5):", data[2:6])

print("iterating over elements:")
[print(element) for element in data]

filter_data = data[data > 5]
print("filtered data:", filter_data)
print("sorted data:", data.sort_values())

reshaped_data = pd.DataFrame(data).values.reshape(2, 5)
print("reshaped data (2x5):", reshaped_data)
