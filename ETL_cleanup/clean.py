import pandas as pd
folder = 'ETL_cleanup'

# Load messy data
customers = pd.read_csv(f'{folder}/customers.csv')
orders = pd.read_csv(f'{folder}/orders.csv')

# Clean customers: remove duplicates
customers_clean = customers.drop_duplicates(subset=['email'])
print(f"Removed {len(customers) - len(customers_clean)} duplicates")

# Clean orders: handle missing values
orders['amount'].fillna(orders['amount'].mean(), inplace=True)
orders['date'] = pd.to_datetime(orders['date'], errors='coerce')
orders_clean = orders.dropna(subset=['date'])

# Standardize formats
customers_clean['email'] = customers_clean['email'].str.lower()
customers_clean['name'] = customers_clean['name'].str.title()

# Combine datasets
merged_data = pd.merge(
    orders_clean, 
    customers_clean, 
    on='customer_id', 
    how='left'
)

print(f"Final dataset: {len(merged_data)} clean records")
merged_data.to_csv('clean_customer_orders.csv', index=False)
