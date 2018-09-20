import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#POLYNOMIAL REGRESSION

veriler = pd.read_csv("./datasets/maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(x);
lin_reg.fit(x_poly, y)
y_pred = lin_reg.predict(x_poly)

plt.scatter(x, y, color= 'red')
plt.plot(x, y_pred, color= 'blue')

from sklearn.metrics import r2_score
print("r2 score: " + str(r2_score(y, lin_reg.predict(x_poly))))
