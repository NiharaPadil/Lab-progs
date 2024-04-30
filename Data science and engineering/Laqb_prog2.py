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
filtered_data = df[df['sepal_length'] > 5.0]  # Filtering for sepal_length greater than 5.0
print("\nFiltered data (sepal_length > 5.0):")
print(filtered_data)

# Sorting data
sorted_data = df.sort_values(by='sepal_width', ascending=False)  # Sorting by sepal_width
print("\nSorted data (by sepal_width):")
print(sorted_data)

# Creating a new column 'Bonus'
df['petal_length_bonus'] = df['petal_length'] * 0.1  # Multiplying petal_length by 0.1
print("\nData with new column 'petal_length_bonus':")
print(df)

# Writing data to Excel file
df.to_excel('Output.xlsx', index=False)
print("\nData written to Output.xlsx")
