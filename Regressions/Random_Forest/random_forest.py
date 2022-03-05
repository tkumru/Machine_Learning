import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("maaslar.csv")

x = data[["Egitim Seviyesi"]].values
y = data[["maas"]].values

# n_estimators = calculate how many tree will draw
ran_for = RandomForestRegressor(n_estimators=10, random_state=0)
ran_for.fit(x, y.ravel())

print(r2_score(y, ran_for.predict(x)))

plt.scatter(x, y, color='red')
plt.plot(x, ran_for.predict(x), color='blue')
plt.show()
