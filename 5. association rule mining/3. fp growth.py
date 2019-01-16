import sys
sys.path.append('./libraries')
import pyfpgrowth
import pandas as pd
import numpy as np

data = pd.read_csv('./datasets/sepet.csv', header = None)

transactions = []
for d in data.values:
    tmp = []
    #print(pd.notna(d))
    for i in d:
        if pd.notna(i):
            tmp.append(i)
    transactions.append(tmp)
    
patterns = pyfpgrowth. find_frequent_patterns(transactions, 10)
rules = pyfpgrowth. generate_association_rules(patterns,0.8)