import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("satislar.csv")

months = df[['Aylar']]
sells = df[['Satislar']]

x_train, x_test, y_train, y_test = train_test_split(months, sells, test_size=0.3)

scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
Y_train = scaler.fit_transform(y_train)
Y_test = scaler.fit_transform(y_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

estimation = pd.DataFrame(lin_reg.predict(X_test), columns=["Estimation"])
Y_test = pd.DataFrame(Y_test, columns=["Real"])

print(pd.concat([estimation, Y_test], axis=1))
