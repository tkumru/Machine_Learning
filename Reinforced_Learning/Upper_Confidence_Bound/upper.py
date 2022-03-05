import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

datas = pd.read_csv("Ads_CTR_Optimisation.csv")

n = 10000 # clicked times
d = 10 # advert types

# Ri(n)
awards = [0] * d

# Ni(n)
clicked = [0] * d

total = 0
selected = list()

for i in range(1, n):
    ad = 0 # selected advert
    max_ucb = 0

    for j in range(d):
        if clicked[j] > 0:
            mean = awards[j] / clicked[j]
            delta = math.sqrt(3 / 2 * math.log(n) / clicked[j])
            ucb = mean + delta

        else:
            ucb = n * 10

        if max_ucb < ucb:
            max_ucb = ucb
            ad = j

    selected.append(ad)
    clicked[ad] += 1
    award = datas.values[i, ad]
    awards[ad] += award
    total += award

print(total)

plt.hist(selected)
plt.show()
