from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

X, y = make_moons(200, noise=.05, random_state=0)

model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')

labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.show()