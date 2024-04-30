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





#---Not there for exam ---
#converting csv to excel
#import pandas as pd
#df = pd.read_csv("iris.csv")
#df.to_excel('iris_excel.xlsx', index=False)
#print("Data successfully converted to Excel and saved as 'iris_excel.xlsx'")

