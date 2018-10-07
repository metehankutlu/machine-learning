import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('./datasets/musteriler.csv')

x = veriler.iloc[:, 3:].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

#plt.plot(range(1,11), wcss)
#plt.show()


kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
kmeans.fit(x)
y_pred = kmeans.fit_predict(x)

plt.scatter(x[y_pred==0, 0], x[y_pred==0, 1], s=100, color='red')
plt.scatter(x[y_pred==1, 0], x[y_pred==1, 1], s=100, color='blue')
plt.scatter(x[y_pred==2, 0], x[y_pred==2, 1], s=100, color='green')
plt.show()