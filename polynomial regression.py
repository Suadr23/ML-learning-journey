import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.sort(5 * np.random.rand(250, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.05, X.shape[0])

X.shape, y.shape
plt.scatter(X, y, color='blue')
plt.title("Generated Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
