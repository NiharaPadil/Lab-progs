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
