import pandas as pd

df = pd.read_csv('EDA/data_2.csv')

Q1 = df['revenue'].quantile(0.25)
Q3 = df['revenue'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['revenue'] < lower_bound) | 
              (df['revenue'] > upper_bound)]
"""
print(f"Found {len(outliers)} outliers")

print (outliers)


a = 1/0

correlation_matrix = df[['quantity', 'revenue', 
                          'price', 'discount']].corr()

print(correlation_matrix)


"""
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot shows outliers as dots
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='revenue', data=df)
plt.title('Revenue Distribution by Category')
plt.xticks(rotation=45)
plt.show()
    

