import pandas as pd 

# Read the CSV file
df = pd.read_csv("iris.csv")

# Print first few rows
print("First few rows:")
print(df.head())

# Print summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Filtered data
filtered_data = df[df['SepalLengthCm'] > 5.0]  # Filtering for SepalLengthCm greater than 5.0
print("\nFiltered data (SepalLengthCm > 5.0):")
print(filtered_data)

# Sorting data
sorted_data = df.sort_values(by='SepalWidthCm', ascending=False)  # Sorting by SepalWidthCm
print("\nSorted data (by SepalWidthCm):")
print(sorted_data)

# Creating a new column 'PetalLengthBonus'
df['PetalLengthBonus'] = df['PetalLengthCm'] * 0.1  # Multiplying PetalLengthCm by 0.1
print("\nData with new column 'PetalLengthBonus':")
print(df)

# Writing data to Excel file
df.to_excel('Output.xlsx', index=False)
print("\nData written to Output.xlsx")


#-------Bar chart Plotting for sepalwidth
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

bins = [2, 2.5, 3, 3.5, 4, 4.5]

labels = ['2-2.5', '2.6-3', '3.1-3.5', '3.6-4', '4.1-4.5']

df['SepalWidthCategory'] = pd.cut(df['SepalWidthCm'], bins=bins, labels=labels)


category_counts = df['SepalWidthCategory'].value_counts()

category_counts.plot(kind='bar')
plt.title('Distribution of Sepal Width Categories')
plt.xlabel('Sepal Width Category')
plt.ylabel('Frequency')
plt.xticks(rotation=360)  
plt.show()


#-------Scatter Plotting for sepalwidth
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

bins = [2, 2.5, 3, 3.5, 4, 4.5]

labels = ['2-2.5', '2.6-3', '3.1-3.5', '3.6-4', '4.1-4.5']

df['SepalWidthCategory'] = pd.cut(df['SepalWidthCm'], bins=bins, labels=labels)

category_counts = df['SepalWidthCategory'].value_counts()

bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

plt.scatter(bin_midpoints, category_counts)
plt.title('Scatterplot of Sepal Width Categories')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  
plt.grid(True)
plt.show()

#----Central limit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Iris dataset
iris_df = pd.read_csv("iris.csv")

# Extract the sepal length data
sepal_length_data = iris_df['SepalLengthCm']

# Set the parameters for the sampling distribution
num_samples = 100  # Number of samples to draw
sample_size_range = range(5, 50, 5)  # Range of sample sizes to test

# Initialize a figure for plotting
plt.figure(figsize=(12, 6))

# Iterate over different sample sizes
for sample_size in sample_size_range:
    # Initialize an array to store sample means
    sample_means = np.zeros(num_samples)
    
    # Draw samples and calculate sample means
    for i in range(num_samples):
        sample = np.random.choice(sepal_length_data, size=sample_size, replace=False)
        sample_means[i] = np.mean(sample)
    
    # Plot the histogram of sample means
    plt.hist(sample_means, bins=30, alpha=0.5, label=f'Sample Size: {sample_size}')

# Add labels and title to the plot
plt.title('Distribution of Sample Means (Central Limit Theorem)')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()




#---Not there for exam ---
#converting csv to excel
#import pandas as pd
#df = pd.read_csv("iris.csv")
#df.to_excel('iris_excel.xlsx', index=False)
#print("Data successfully converted to Excel and saved as 'iris_excel.xlsx'")

