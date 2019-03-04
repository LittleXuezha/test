from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
Xmoon, ymoon = make_moons(200, noise=0.05, random_state=0)

gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
labels = gmm16.fit_predict(Xmoon)
Xnew ,ynew= gmm16.sample(200)
# plt.scatter(Xnew[:, 0], Xnew[:, 1])
print(Xnew)
# plt.show()
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Xmoon)
          for n in n_components]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='bic')

plt.plot(n_components, [m.aic(Xmoon) for m in models], label='aic')
plt.xlabel('n_components')
plt.show()