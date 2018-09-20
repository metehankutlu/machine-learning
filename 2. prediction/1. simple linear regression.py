import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#SIMPLE LINEAR REGRESSION

veriler = pd.read_csv('./datasets/satislar.csv')
#x = veriler.iloc[:,0:-1].values
#y = veriler.iloc[:,-1:].values
x = veriler[['Aylar']]
y = veriler[['Satislar']]

from sklearn.preprocessing import StandardScaler
ssx = StandardScaler()
ssy = StandardScaler()
x = ssx.fit_transform(x)
y = ssy.fit_transform(y)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

plt.scatter(ssx.inverse_transform(x_test), ssy.inverse_transform(y_test), color = 'red')
plt.plot(ssx.inverse_transform(x_test), ssy.inverse_transform(lr.predict(x_test)), color = 'blue')

from sklearn.metrics import r2_score
print("r2 score: " + str(r2_score(y_test, lr.predict(x_test))))