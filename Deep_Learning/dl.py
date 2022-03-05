import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

datas = pd.read_csv("Churn_Modelling.csv")

x = datas.iloc[:, 3: 13].values
y = datas.iloc[:, 13].values

le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])

le2 = LabelEncoder()
x[:, 2] = le2.fit_transform(x[:, 2])

ohe = ColumnTransformer([('ohe', OneHotEncoder(dtype=float), [1])], remainder="passthrough")
x = ohe.fit_transform(x)
x = x[: , 1:]

x_train, x_test, y_train, y_test = train_test_split(x, y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

classifier = Sequential()

classifier.add(Dense(6, activation='relu', input_dim=11))
classifier.add(Dense(6, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, epochs=50)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
