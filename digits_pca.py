from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
digits = load_digits()
print(digits.data.shape)

pca = PCA()
pro = pca.fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('components')
plt.ylabel('累计方差')
plt.show()

