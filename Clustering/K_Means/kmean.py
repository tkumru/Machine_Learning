import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datas = pd.read_csv('müsteri.csv')

x = datas.iloc[:, 3:].values

results = list()

for i in range(1, 10):
    km = KMeans(n_clusters=i)
    km.fit(x)
    results.append(km.inertia_)

plt.plot(range(1, 10), results)
plt.show()
