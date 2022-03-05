import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('veriler.csv')

# Missing Value

# it means convert nan values to column's mean.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
int_columns = df.iloc[:, 1: 4].values
imputer = imputer.fit(int_columns[:, 1: 4])
int_columns[:, 1: 4] = imputer.transform(int_columns[:, 1: 4])

#------------------------------------------------------------------

# Categorize Data

country = df.iloc[:, 0: 1].values
label_encode = preprocessing.LabelEncoder()
country[:, 0] = label_encode.fit_transform(df.iloc[:, 0])  # it categorize data to in column

one_hot_encode = preprocessing.OneHotEncoder()
country = one_hot_encode.fit_transform(country).toarray()  # it convert (categorized data from column) to 3 column.

sex = df.iloc[:, -1:].values
label_encode = preprocessing.LabelEncoder()
sex[:, -1] = label_encode.fit_transform(df.iloc[:, -1])

one_hot_encode = preprocessing.OneHotEncoder()
sex = one_hot_encode.fit_transform(sex).toarray()

#------------------------------------------------------------------

# Merge Datas

country_df = pd.DataFrame(data=country, index=range(country.shape[0]), columns=['fr', 'tr', 'us'])
intColumns_df = pd.DataFrame(data=int_columns, index=range(country.shape[0]), columns=["boy", "kilo", "yas"])
sex_df = pd.DataFrame(data=sex[:,:1], index=range(sex.shape[0]), columns=['cinsiyet'])

df_x = pd.concat([country_df, intColumns_df], axis=1)
df = pd.concat([df_x, sex_df], axis=1)
print(df.corr()["cinsiyet"])

#------------------------------------------------------------------

# Divide Datas

x_train, x_test, y_train, y_test = train_test_split(df_x, sex_df, test_size=0.30)

#------------------------------------------------------------------

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
