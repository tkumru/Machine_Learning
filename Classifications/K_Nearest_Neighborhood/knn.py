import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

datas = pd.read_csv("veriler.csv")

x = datas.iloc[:, 1: 4].values
y = datas.iloc[:, 4:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

con_matrix = confusion_matrix(y_test, y_pred)
print(con_matrix)
