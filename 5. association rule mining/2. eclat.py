import sys
sys.path.append('./libraries')
from my_eclat import Eclat

import pandas as pd

data = pd.read_csv('./datasets/sepet.csv', header = None)
data.fillna(method = 'ffill',axis = 1, inplace = True)
eclat = Eclat()


item_dict = dict()
eclat.fit(data)
rules = eclat.transform()
print(rules)