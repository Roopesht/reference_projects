import sqlite3
import csv
conn = sqlite3.connect('stockdata/stocks.db')
cursor = conn.cursor()

cursor.execute('''
    Create table Ticker (
            Ticker TEXT,
            DateTime TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL
    )
''')

with open('stockdata/data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        #Ticker,Date/Time,Open,High,Low,Close,Volume,Open Interest
        # Create table for Ticker,DateTime,Open,High,Low,Close,Volume
        # insert in to rows (insert commands)
        sql = "INSERT INTO Ticker VALUES (?, ?, ?, ?, ?, ?, ?)" 
        cursor.execute(sql, (row['Ticker'], row['Date/Time'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))

conn.commit()

