import pandas as pd 

# Step 1: Read the CSV file
df_csv = pd.read_csv("iris.csv")

# Step 2: Write the data to an Excel file
df_csv.to_excel('iris.xlsx', index=False)

# Step 3: Read the data from the Excel file
df = pd.read_excel('iris.xlsx')

# Step 4: Print first few rows
print("First few rows:")
print(df.head())

# Step 5: Print summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Step 6: Filtered data
filtered_data = df[df['SepalLengthCm'] > 5.0]  # Filtering for SepalLengthCm greater than 5.0
print("\nFiltered data (SepalLengthCm > 5.0):")
print(filtered_data)

# Step 7: Sorting data
sorted_data = df.sort_values(by='SepalWidthCm', ascending=False)  # Sorting by SepalWidthCm
print("\nSorted data (by SepalWidthCm):")
print(sorted_data)

# Step 8: Creating a new column 'PetalLengthBonus'
df['PetalLengthBonus'] = df['PetalLengthCm'] * 0.1  # Multiplying PetalLengthCm by 0.1
print("\nData with new column 'PetalLengthBonus':")
print(df)

# Step 9: Writing data to Excel file
df.to_excel('Output.xlsx', index=False)
print("\nData written to Output.xlsx")
