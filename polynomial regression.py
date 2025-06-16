import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = np.sort(5 * np.random.rand(600, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.01, X.shape[0])

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [1, 2, 5]

plt.scatter(X, y, color='blue', label='Data')

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # تحويل بيانات التدريب
    X_train_poly = poly_features.fit_transform(X_train)
    
    # اختيار النموذج (LinearRegression، Ridge، أو Lasso)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    
    # تحضير نقاط رسم أكثر نعومة
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = poly_model.predict(X_plot_poly)
    
    # رسم المنحنى
    plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    # توقع على بيانات الاختبار لحساب المقاييس
    X_test_poly = poly_features.transform(X_test)
    y_test_pred = poly_model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print(f"Degree {degree} - MSE: {mse:.5f}, R²: {r2:.5f}")

plt.title('Polynomial Regression with Different Degrees')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

