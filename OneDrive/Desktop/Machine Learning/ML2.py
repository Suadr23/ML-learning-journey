import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

X = 10 * np.random.rand(250, 1) - 10
y = 0.3 * X**2 + X + 2 + np.random.randn(250, 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, y, color='purple')
plt.show()  
