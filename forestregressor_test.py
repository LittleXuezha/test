import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)


def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))
    return slow_oscillation + fast_oscillation +noise


y = model(x)
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

xfit = np.linspace(0, 10, 100)
yfit = forest.predict(xfit[:, None])

ytrue = model(xfit, sigma=0)
plt.errorbar(x, y, 0.3, fmt='o', alpha=0.3)
plt.plot(xfit, yfit, 0.3, '-r')
plt.plot(xfit, ytrue, '-k')
plt.show()