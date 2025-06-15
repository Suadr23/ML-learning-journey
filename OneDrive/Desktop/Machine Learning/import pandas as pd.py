import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# تحميل البيانات
data = pd.read_excel(r'C:\Users\USER PC\OneDrive\Desktop\Machine Learning\house_prices.xlsx')
X = data[['Size']]
y = data['Price']

# تدريب النموذج
model = LinearRegression()
model.fit(X, y)

# التنبؤ
predictions = model.predict(X)

# الرسم البياني
plt.scatter(X, y, color='green', label='Actual Prices')
plt.plot(X, predictions, color='orange', label='Predicted Prices')
plt.xlabel('House Size (sqm)')
plt.ylabel('Price (SAR)')
plt.title('House Size vs Price')
plt.legend()
plt.show()