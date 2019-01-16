import pandas as pd
import matplotlib.pyplot as plt
import random
data = pd.read_csv('./datasets/Ads_CTR_Optimisation.csv')
N = len(data.values)
d = len(data.values.T)
rewards = [0] * d
selections = [0] * d
total_reward = 0
selected_ads = []
ones = [0] * d
zeros = [0] * d
for n in range(0, N):
    ad = 0
    max_th = 0
    for i in range(0, d):
        th = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if max_th < th:
            max_th = th
            ad = i
    selected_ads.append(ad)
    selections[ad] += 1
    reward = data.values[n, ad]
    if reward == 1:
        ones[ad] += 1
    else:
        zeros[ad] += 1
    rewards[ad] += reward
    total_reward += reward
print(total_reward)
for i in range(0, d):
    print(selections[i])
plt.hist(selections)
plt.show()