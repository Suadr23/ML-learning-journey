import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_excel('house_prices.xlsx')  
X = data[['Size']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

plt.scatter(X, y, color='green', label='Actual Prices')
plt.plot(X, predictions, color='orange', label='Predicted Prices')
plt.xlabel('House Size (sqm)')
plt.ylabel('Price (OMR)')
plt.title('House Size vs Price')
plt.legend()
plt.show() 