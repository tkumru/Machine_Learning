import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

datas = pd.read_csv('Wine.csv')
x = datas.iloc[:, 0: 13].values
y = datas.iloc[:, 13].values

x_train, x_test, y_train, y_test = train_test_split(x, y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=2)
x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

lda  = LDA(n_components=2)
x_train3 = lda.fit_transform(x_train, y_train)
x_test3 = lda.transform(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2, y_train)

classifier3 = LogisticRegression(random_state=0)
classifier3.fit(x_train3, y_train)

y_pred = classifier.predict(x_test)
y_pred2 = classifier2.predict(x_test2)
y_pred3 = classifier3.predict(x_test3)

print("Actual / Without PCA")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Actual / With PCA")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
print("Actual / With LDA")
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)
