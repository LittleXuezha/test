from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;sns.set()
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
digits = load_digits()
print(digits.data.shape)
tsne = TSNE(n_components=2, init='pca', random_state=0)
digits_proj = tsne.fit_transform(digits.data)
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, cmap=plt.cm.binary)
# plt.show()
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

print(accuracy_score(digits.target, labels))

mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)

plt.xlabel('true label')
plt.ylabel('predict label')
plt.show()