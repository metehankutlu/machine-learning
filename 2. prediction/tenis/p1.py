import pandas as pd
import numpy as np
veriler = pd.read_csv("./datasets/odev_tenis.csv")

from sklearn.preprocessing import LabelEncoder
'''labelEncoder = LabelEncoder()
outlook = veriler.iloc[:,0:1].values
windy = veriler.iloc[:,3:4].values
play = veriler.iloc[:,4:].values
outlook[:,0] = labelEncoder.fit_transform(outlook[:,0])
windy_enc = labelEncoder.fit_transform(windy[:,0])
play[:,0] = labelEncoder.fit_transform(play[:,0])'''

veriler2 = veriler.iloc[:,[0,3,4]].apply(LabelEncoder().fit_transform)

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features='all')
outlook = oneHotEncoder.fit_transform(veriler2.iloc[:,0:1]).toarray()

outlookDF = pd.DataFrame(data = outlook, index = range(14), columns = ['overcast', 'rainy', 'sunny'])
windyplayDF = pd.DataFrame(data = veriler2.iloc[:,1:], index = range(14), columns = ['windy', 'play'])
##playDF = pd.DataFrame(data = play, index = range(14), columns = ['play'])


tempDF = pd.DataFrame(data = veriler.iloc[:,1:2], index = range(14), columns = ['temperature'])

y = veriler.iloc[:,2:3];
x = pd.concat([outlookDF, windyplayDF, tempDF], axis = 1)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = x, axis = 1)
x_l = x.iloc[:, [5]].values
r = sm.OLS(endog=y.astype(float), exog=x_l.astype(float)).fit()
print(r.summary())

x_train, x_test, y_train, y_test = train_test_split(x.iloc[:, [0,1,2,4,5]], y, test_size=0.33, random_state=42)
lr.fit(x_train, y_train)
y_pred2 = lr.predict(x_test)
