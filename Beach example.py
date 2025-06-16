import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic wave-like data
np.random.seed(42)
X = np.sort(6 * np.random.rand(200, 1), axis=0)  # Random X values between 0 and 6
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Sin wave + some noise

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Plot original data
plt.scatter(X, y, color='skyblue', label='Data points (wave shape)')

# 4. Try different polynomial degrees
for degree in [1, 3, 5]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Generate smooth curve for plotting
    X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_curve_poly = poly.transform(X_curve)
    y_curve = model.predict(X_curve_poly)

    plt.plot(X_curve, y_curve, label=f'Degree {degree}')

    # Evaluate model on test data
    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f" Degree {degree} - MSE: {mse:.4f}, R²: {r2:.4f}")

# 5. Show plot
plt.title("Simple Wave Simulation using Polynomial Regression")
plt.xlabel("X (e.g. time)")
plt.ylabel("y (e.g. wave height)")
plt.legend()
plt.show()
