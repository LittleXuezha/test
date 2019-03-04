from sklearn.datasets.samples_generator import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
X, y = make_circles(100, factor=0.1, noise=.1)
clf = SVC(kernel='linear')
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()
print(X)