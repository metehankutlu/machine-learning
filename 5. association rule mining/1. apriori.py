import sys
sys.path.append('./libraries')
from apyori import apriori

import pandas as pd

data = pd.read_csv('./datasets/sepet.csv', header = None)
data.fillna(method = 'ffill',axis = 1, inplace = True)
t = []
for i in range(0, len(data)):
    t.append([str(data.values[i,j]) for j in range(0, len(data.values[i,]))])
    
rules = apriori(t,min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 5)

print(list(rules))