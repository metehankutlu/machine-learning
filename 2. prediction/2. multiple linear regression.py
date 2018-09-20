import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("./datasets/veriler.csv")
#MULTİPLE LINEAR REGRESSION
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
yas = veriler.iloc[:, 2:4].values
imputer = imputer.fit(yas)
yas = imputer.transform(yas)
yasDF = pd.DataFrame(data = yas, index = range(22), columns = ['kilo', 'yas'])
ulke = veriler.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
ulke[:,0] = labelEncoder.fit_transform(ulke[:,0])
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features='all')
ulke = oneHotEncoder.fit_transform(ulke).toarray()
ulkeDF = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])
c = veriler.iloc[:,-1:].values
labelEncoder = LabelEncoder()
c[:,0] = labelEncoder.fit_transform(c[:,0])
oneHotEncoder = OneHotEncoder(categorical_features='all')
c = oneHotEncoder.fit_transform(c).toarray()
cinsiyetDF = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['erkek'])
y = pd.DataFrame(data = veriler.iloc[:,1:2].values, index = range(22), columns = ['boy'])
x = pd.concat([ulkeDF, yasDF, cinsiyetDF], axis = 1)
#backward elimination
import statsmodels.formula.api as sm
#x = pd.concat([pd.DataFrame(data = np.ones((22,1)).astype(int)), x], axis = 1)
#x_l = x.iloc[:, [0,1,2,3,4,5]].values
x_l = x.iloc[:, [0,1,2,3,5]].values
r = sm.OLS(endog=y, exog=x_l).fit()
print(r.summary())
#train-test split
from sklearn.cross_validation import train_test_split #dataseti test ve train olarak ikiye böler
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score
print("r2 score: " + str(r2_score(y_test, lr.predict(x_test))))
