import pandas as pd

df = pd.read_csv('EDA/data_2.csv')

# Group by single column
category_summary = df.groupby('category')['revenue'].count()
print(category_summary)

# Output:
# category
# Electronics    125000
# Clothing        85000
# Home           65000
# Sports         55000

# Sort to find top performer
print (category_summary.sort_values(ascending=False))


# Multiple aggregations at once
summary = df.groupby('region').agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': 'sum'
})

print(summary)


# Get insights per region
region_insights = df.groupby(['region', 'category'])['revenue'].sum()
print(region_insights.unstack())
    
# Multiple aggregations at once
summary = df.groupby('region').agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': 'sum'
})

print(summary)

# Get insights per region
region_insights = df.groupby(['region', 'category'])['revenue'].sum()
print(region_insights.unstack())
    

