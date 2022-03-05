import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("maaslar.csv")

x = data[["Egitim Seviyesi"]].values
y = data[["maas"]]

poly_reg = PolynomialFeatures(degree=6)
x_poly = poly_reg.fit_transform(x)

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x_poly), color="green")
plt.show()
