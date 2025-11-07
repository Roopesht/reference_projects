import sqlite3
conn = sqlite3.connect('stockdata/stocks.db')
cursor = conn.cursor()


cursor.execute(
    "SELECT Close FROM Ticker "
)
prices = [row[0] for row in cursor.fetchall()]

# Calculate analytics
max_price = max(prices)
min_price = min(prices)
avg_price = sum(prices) / len(prices)
trend = "UP" if prices[-1] > prices[0] else "DOWN"

print(f"Max: {max_price}")
print(f"Min: {min_price}")
print(f"Avg: {avg_price:.2f}")
print(f"Trend: {trend}")

conn.close()