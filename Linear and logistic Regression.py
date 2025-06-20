import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_informative=1, random_state=42)

clf = LogisticRegression(C=250)
clf.fit(X, y)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

x_test = np.linspace(X[:, 0].min(), X[:, 0].max(), 300).reshape(-1, 1)

x_test_full = np.hstack([x_test, np.full_like(x_test, X[:,1].mean())])

y_prob = clf.predict_proba(x_test_full)[:, 1]

y_lin = lin_reg.predict(x_test_full)

plt.scatter(X[:, 0], y, c=y, cmap='bwr', alpha=0.5)

plt.plot(x_test, y_lin, color='green', label='Linear Regression')

plt.plot(x_test, y_prob, color='black', label='Logistic Regression Probability')
plt.xlabel('Feature 1')
plt.ylabel('Target / Probability')
plt.legend()
plt.title('Logistic vs Linear Regression on 1 Feature')
plt.show()