from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
X, y_ture = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.6, random_state=0)
X = X[:, ::-1]
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)

probs = gmm.predict_proba(X)
print(probs[:5].round(3))

size = 50 * probs.max(1) ** 4

plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, cmap='viridis')
plt.show()