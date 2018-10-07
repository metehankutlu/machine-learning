import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('./datasets/musteriler.csv')

x = veriler.iloc[:, 3:].values

from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(x)
y_pred = agg.fit_predict(x)

plt.scatter(x[y_pred==0, 0], x[y_pred==0, 1], s=100, color='red')
plt.scatter(x[y_pred==1, 0], x[y_pred==1, 1], s=100, color='blue')
plt.scatter(x[y_pred==2, 0], x[y_pred==2, 1], s=100, color='green')
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))