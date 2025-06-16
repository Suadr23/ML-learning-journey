import numpy as np
import matplotlib.pyplot as plt

# توليد بيانات
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# رسم البيانات فقط
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='black', label='Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Noisy Sine Data')
plt.legend()
plt.show()
