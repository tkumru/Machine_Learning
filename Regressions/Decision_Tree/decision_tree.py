import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("maaslar.csv")

x = data[["Egitim Seviyesi"]].values
y = data[["maas"]].values

dec_tr = DecisionTreeRegressor(random_state=0)
dec_tr.fit(x, y)

plt.scatter(x, y, color='red')
plt.plot(x, dec_tr.predict(x), color='green')
plt.show()
