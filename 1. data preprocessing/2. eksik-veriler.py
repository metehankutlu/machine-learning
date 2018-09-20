#eksik veriler

import pandas as pd

veriler = pd.read_csv("./datasets/eksikveriler.csv")

from sklearn.preprocessing import Imputer #eksik veriler için kullanılır

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) # axis 0 -> column 1 -> row

yas = veriler.iloc[:, 1:4].values #iloc -> kesme | [rows, columns] | a:b -> a ve b arasındakileri(b dahil değil)
imputer = imputer.fit(yas)
yas = imputer.transform(yas)
print(yas)