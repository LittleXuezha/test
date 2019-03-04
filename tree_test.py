from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from visualize_test import *
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
model = RandomForestClassifier(n_estimators=1, random_state=0)
visual_classifier(model, X, y)
plt.show()
