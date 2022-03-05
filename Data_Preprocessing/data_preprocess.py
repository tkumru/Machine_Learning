import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#------------------------------------------------------------------

# Merge Datas

country_df = pd.DataFrame(data=country, index=range(country.shape[0]), columns=['fr', 'tr', 'us'])

intColumns_df = pd.DataFrame(data=int_columns, index=range(country.shape[0]), columns=["boy", "kilo", "yas"])

sex = df.iloc[:, -1].values
sex_df = pd.DataFrame(data=sex, index=range(sex.shape[0]), columns=['cinsiyet'])

df_x = pd.concat([country_df, intColumns_df], axis=1)
df = pd.concat([df, sex_df], axis=1)

#------------------------------------------------------------------

# Divide Datas

x_train, x_test, y_train, y_test = train_test_split(df_x, sex_df, test_size=0.30)

#------------------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
