import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("odev_tenis.csv")

data = data.apply(LabelEncoder().fit_transform)

outlook = data[["outlook"]].values
one_hot_encode = OneHotEncoder()
outlook = one_hot_encode.fit_transform(outlook).toarray()

outlook_df = pd.DataFrame(data=outlook, index=range(outlook.shape[0]), columns=["overcast", "sunny", "rainy"])

df = pd.concat([outlook_df, data.iloc[:, 1:]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    df[['overcast', 'sunny', 'rainy', 'temperature', "windy", "play"]].values, 
    df[["humidity"]], 
    test_size=0.33)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


