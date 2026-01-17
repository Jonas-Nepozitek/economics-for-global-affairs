import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pwd = os.getcwd()

dataset = "https://raw.githubusercontent.com/akhandelwal8/globaleconomics/refs/heads/main/hwk/hwk1_convergence.csv"

df = pd.read_csv(dataset,sep='\t')
                
gdp60 = df['gdppc1960']
gdp00 = df['gdppc2000']
n = 40.0
g = np.log(gdp00/gdp60)/n

# Part 1
plt.figure()
plt.hist(g,bins=np.arange(np.min(g),np.max(g)+0.01,0.01))
plt.title("Average Annual Growth Rate")
plt.xlabel("Growth Rate")

# Part 2
print(f"The percentiles of the growth rates are:\n\t10th: {np.nanpercentile(g,10):.2g} \
      \n\t25th: {np.nanpercentile(g,25):.2g}\n\t50th: {np.nanpercentile(g,50):.2g}\n\t75th: {np.nanpercentile(g,75):.2g} \
      \n\t90th: {np.nanpercentile(g,90):.2g}")

# Part 3
largestctr = df['code'][np.argmax(g)]

print(largestctr)





plt.show()