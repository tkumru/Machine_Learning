import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import re
import nltk

# PREPROCESSING

datas = pd.read_csv("Restaurant_Reviews.csv")

stop = nltk.download("stopwords")

ps = PorterStemmer()

total = list()
for i in range(1000):
    comment = re.sub('[^a-zA-Z]', ' ', datas['Review'][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = " ".join(comment)
    total.append(comment)

# FEATURE STRUCTURE

count = CountVectorizer(max_features=100)

x = count.fit_transform(total).toarray() # independent
y = datas.iloc[:, 1].values  # dependent

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
