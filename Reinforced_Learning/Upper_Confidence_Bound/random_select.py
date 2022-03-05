import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

datas = pd.read_csv("Ads_CTR_Optimisation.csv")

n = 10000
d = 10
c = 0
selected = list()

for i in range(0, n):
    ad = random.randrange(d)
    selected.append(ad)
    award = datas.values[n, ad]
    c += award

plt.hist(selected)
plt.show()
