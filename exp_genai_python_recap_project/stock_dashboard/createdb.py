import sqlite3

# Connect to database (creates if doesn't exist)
conn = sqlite3.connect('stockdata/stocks.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    Create table Ticker (
            DateTime TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL
    )
''')

# Insert data
cursor.execute(
    "INSERT INTO prices VALUES (?, ?, ?)",
    ('AAPL', 150.25, '2025-11-06')
)

conn.commit()
conn.close()