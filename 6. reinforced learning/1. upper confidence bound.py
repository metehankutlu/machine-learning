import pandas as pd
import matplotlib.pyplot as plt
import sys, math
data = pd.read_csv('./datasets/Ads_CTR_Optimisation.csv')
N = len(data.values)
d = len(data.values.T)
rewards = [0] * d
selections = [0] * d
total_reward = 0
selected_ads = []
for n in range(0, N):
    ad = 0
    max_ucb = 0
    for i in range(0, d):
        if selections[i] > 0:
            average = rewards[i] / selections[i]
            delta = math.sqrt(3/2 * math.log(n)/selections[i])
            upper_bound = average + delta
        else:
            upper_bound = sys.float_info.max
        if max_ucb < upper_bound:
            max_ucb = upper_bound
            ad = i
    selected_ads.append(ad)
    selections[ad] += 1
    reward = data.values[n, ad]
    rewards[ad] += reward
    total_reward += reward
print(total_reward)
plt.hist(selections)
plt.show()