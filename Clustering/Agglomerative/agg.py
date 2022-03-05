import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy as sch

datas = pd.read_csv('müsteri.csv')

x = datas.iloc[:, 3:].values

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_pred = ac.fit_predict(x)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=100, c='red')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=100, c='blue')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=100, c='green')
plt.show()

dengrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()
