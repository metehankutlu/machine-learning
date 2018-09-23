import pandas as pd

#NAIVE BAYES CLASSIFICATION
veriler = pd.read_csv("./datasets/veriler.csv")

x = veriler.iloc[:, 1:4].values
y = veriler.iloc[:, [4]].values

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
x_scaled = sc.fit_transform(x)

from sklearn.cross_validation import train_test_split #dataseti test ve train olarak ikiye böler
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=42) 

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)