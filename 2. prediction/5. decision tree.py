import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#DECISION TREE REGRESSİON

veriler = pd.read_csv("./datasets/maaslar.csv")
x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, dtr.predict(x), color = 'blue')

from sklearn.metrics import r2_score
print("r2 score: " + str(r2_score(y, dtr.predict(x))))