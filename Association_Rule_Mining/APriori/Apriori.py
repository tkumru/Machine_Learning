import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from APriori_github import apriori

datas = pd.read_csv('sepet.csv', header=None)

l = list()
for i in range(7501):
    l.append(str(datas.values[i, j]) for j in range(20))

rules = apriori(l, min_support=0.01, min_confidence=0.2, min_lift=2, min_length=2)
