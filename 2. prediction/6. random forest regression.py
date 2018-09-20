import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#RANDOM FOREST REGRESSİON

veriler = pd.read_csv("./datasets/maaslar.csv")
x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=10)
rfr.fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, rfr.predict(x), color = 'blue')

from sklearn.metrics import r2_score
print("r2 score: " + str(r2_score(y, rfr.predict(x))))