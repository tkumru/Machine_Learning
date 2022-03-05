import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVR as svr

# MUST TO SCALE DATAS

data = pd.read_csv("maaslar.csv")

x = data[["Egitim Seviyesi"]].values
y = data[["maas"]].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)

svr_reg = svr(kernel="rbf")
svr_reg.fit(x_scaled, y_scaled)

plt.scatter(x_scaled, y_scaled, color='red')
plt.plot(x_scaled, svr_reg.predict(x_scaled), color='green')
plt.show()
