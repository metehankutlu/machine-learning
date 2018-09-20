import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#READ DATA
veriler = pd.read_csv('./datasets/maaslar_yeni.csv')

#PREPROCESSING
raw_x = veriler.iloc[:, 2:5].values
raw_y = veriler.iloc[:, [5]].values

#SCALING
from sklearn.preprocessing import StandardScaler
ssx = StandardScaler()
ssy = StandardScaler()
x = ssx.fit_transform(np.float64(raw_x))
y = ssy.fit_transform(np.float64(raw_y)).ravel()

#BACKWARD ELIMINATION
import statsmodels.formula.api as sm
x_l = veriler.iloc[:, [2,4]].values
r = sm.OLS(endog=y, exog=x_l).fit()
print(r.summary())

x = ssx.fit_transform(np.float64(x_l))

#TRAIN-TEST SPLIT
'''from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)'''

#MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

mlr_r2 = r2_score(y, lr.predict(x))
print("mlr r2 score: " + str(mlr_r2))

#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
plr = LinearRegression()
poly_feat = PolynomialFeatures(degree = 6)
x_poly = poly_feat.fit_transform(x)
plr.fit(x_poly, y)

pr_r2 = r2_score(y, plr.predict(x_poly))
print("pr r2 score: " + str(pr_r2))

#SUPPORT VECTOR REGRESSION
from sklearn.svm import SVR
svr = SVR(kernel = "rbf")
svr.fit(x, y)

svr_r2 = r2_score(y, svr.predict(x))
print("svr r2 score: " + str(svr_r2))

#DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x, y)

dtr_r2 = r2_score(y, dtr.predict(x))
print("dtr r2 score: " + str(dtr_r2))

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=50)
rfr.fit(x, y)

rf_r2 = r2_score(y, rfr.predict(x))
print("rf r2 score: " + str(rf_r2))

ceo = ssx.transform(np.reshape([10, 100], (1, -1)))

print("CEO lr prediction: ")
print(ssy.inverse_transform(lr.predict(ceo)))
print("CEO plr prediction: ")
print(ssy.inverse_transform(plr.predict(poly_feat.transform(ceo))))
print("CEO svr prediction: ")
print(ssy.inverse_transform(svr.predict(ceo)))
print("CEO dtr prediction: ")
print(ssy.inverse_transform(dtr.predict(ceo)))
print("CEO rf prediction: ")
print(ssy.inverse_transform(rfr.predict(ceo)))