url = "https://www.nseindia.com/api/quote-equitsdfsadfy?symbol=RELIANCE"

import requests

# API endpoint for stock data

# Make GET request
headers = {
    'User-Agent': 'Mozilla/5.0'
}

response = requests.get(url, headers=headers)

# Check if successful
print ("code: ", response.status_code)
if response.status_code == 200:
    data = response.json()
    price = data['priceInfo']['lastPrice']
    print(f"Stock price: {price}")
else:
    print("Failed to fetch data")

