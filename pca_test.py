import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2), rng.randn(2, 200)).T

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print(X.shape)
print(X_pca.shape)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.scatter(X_pca[:, 0], np.zeros(200))
plt.show()
print(X_new.shape)
print(X_pca)
