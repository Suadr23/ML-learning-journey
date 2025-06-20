from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

iris = load_iris()

X = iris.data[:, 2:]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree_clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42, criterion="entropy")
tree_clf.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
tree.plot_tree(tree_clf, filled=True, feature_names=iris.feature_names[2:], class_names=iris.target_names)
plt.title("Decision Tree")
plt.show()

plt.figure(figsize=(8, 6))
plot_decision_boundary(tree_clf, X_train, y_train, legend=True)
plt.title("Decision Boundary (Training Data)")
plt.text(1.40, 1.0, "Depth=0", fontsize=13)
plt.text(3.2, 1.80, "Depth=1", fontsize=12)
plt.show()

y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("🔍 Test Accuracy:", round(accuracy, 2))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
