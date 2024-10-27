#1a

import numpy as np
data =np.array([1,2,3,4,5,6,7,8,9,10])
mean=np.mean(data)
median=np.median(data)
variance=np.var(data)
std_dev=np.std(data)
print("mean:",mean)
print("median:",median)
print("variance:",variance)
print("standard deviation:",std_dev)
print("element at index 2:",data[2])
sliced_data=data[2:6]
print("sliced data from index 2 to 5",sliced_data)
split_data=np.array_split(data,2)
print("split data:",split_data)
print("iterating over elements")
for element in data:
    print(element)
filter_data=data[data>5]
print("filtered data:",filter_data)
sorted_data=np.sort(data)
print("sorted data:",sorted_data)
additional_data=np.array([11,12,13])
combined_data=np.concatenate((data,additional_data))
print("combined data:",combined_data)
reshaped_data=data.reshape(2,5)
print("reshaped data(2x5):",reshaped_data)


#1b

import pandas as pd
data= pd.Series([1,2,3,4,5,6,7,8,9,10])
mean=data.mean()
median=data.median()
variance=data.var()
std_dev=data.std()
print("mean:",mean)
print("median:",median)
print("variance:",variance)
print("standard deviation:",std_dev)
print("element at index 2:",data[2])
sliced_data=data[2:6]
print("sliced data from index 2 to 5",sliced_data)
print("iterating over elements")
for element in data:
    print(element)
filter_data=data[data>5]
print("filtered data:",filter_data)
sorted_data=data.sort_values()
print("sorted data:",sorted_data)
df=pd.DataFrame({'values':data})
reshaped_data=df.values.reshape(2,5)
print("reshaped data(2x5):",reshaped_data)
