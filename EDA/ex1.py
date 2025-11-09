import pandas as pd

# Load sales data
df = pd.read_csv('EDA/sales_data.csv')

# Ask: What's the shape and structure?
print(df.shape)  # (1000, 6) - 1000 rows, 6 columns
print(df.columns)  
# ['date', 'product', 'category', 'region', 'quantity', 'revenue']

# Ask: What does the data look like?
#print (df.head())

#print (df.groupby('date')['revenue'].count())

# Question 2: Which category generates most revenue?

print (df.groupby('category')['revenue'].sum().sort_values(ascending=False))

# Question 3: Are there regional differences?
print (df.groupby('region')['revenue'].mean())
       