import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = "https://raw.githubusercontent.com/akhandelwal8/globaleconomics/refs/heads/main/class/class_accounting.csv"

df = pd.read_csv(dataset,sep='\t')

print(df.head(5))

#cn = capital
#cgdpo = output

k = df['cn']/df['pop']
y = df['cgdpo']/df['pop']

logy = np.log(y)
A = y/(k**0.3)
logA = np.log(A)
logk = np.log(k)

plt.figure()
plt.scatter(logy,logA,label='logA')
plt.scatter(logy,logk,label='logk')
plt.xlabel("Log(y)")
plt.legend()
plt.show()