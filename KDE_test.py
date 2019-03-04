import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


x = make_data(1000)
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])
x_d = np.linspace(-4, 8, 2000)
loggrob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(loggrob), alpha=0.5)
plt.plot(x, np.full_like(x, 0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.show()