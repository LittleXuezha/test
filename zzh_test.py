from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
model = make_pipeline(PolynomialFeatures(20), LinearRegression())
x = 10 * np.random.rand(50)
y = np.sin(x) + 0.1 * np.random.randn(50)
X = x[:, np.newaxis]
model.fit(X, y)
xfit = np.linspace(0, 10, 100)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
