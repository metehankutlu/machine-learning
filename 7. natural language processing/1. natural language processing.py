import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('./datasets/Restaurant_Reviews.csv')

#preprocessing
sw = nltk.download('stopwords')
ps = PorterStemmer()

reviews = []
for i in range(0, len(data.values)):
    rev = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if not word in set(stopwords.words('english'))]
    reviews.append(' '.join(rev))

#feature extraction
cv = CountVectorizer()

x = cv.fit_transform(reviews).toarray()
y = data['Liked'].values

#train
from sklearn.cross_validation import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0) 

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

#test
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)