import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#SUPPORT VECTOR REGRESSION

veriler = pd.read_csv("./datasets/maaslar.csv")
x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

#svr'da scaler kullanmak zorunlu!!!!

from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
x = scx.fit_transform(x)
scy = StandardScaler()
y = scy.fit_transform(y)

from sklearn.svm import SVR
svr = SVR(kernel = "rbf")
svr.fit(x, y)

plt.scatter(scx.inverse_transform(x), scy.inverse_transform(y))
plt.plot(scx.inverse_transform(x), scy.inverse_transform(svr.predict(x)))

from sklearn.metrics import r2_score
print("r2 score: " + str(r2_score(y, svr.predict(x))))
print("r2 score: " + str(r2_score(scy.inverse_transform(y), scy.inverse_transform(svr.predict(x)))))