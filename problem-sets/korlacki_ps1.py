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
df['growthrate'] = g
sorted_df = df.sort_values(by='growthrate')

smallest3 = sorted_df[0:2]
largest3 = sorted_df[np.argmax(sorted_df['growthrate'])-2:np.argmax(sorted_df['growthrate'])+1]

# Part 4
log60 = np.log(gdp60)
df['log1960'] = log60
plt.figure()
plt.scatter(log60,g)
plt.xlabel("log(GDPPC 1960)")
plt.ylabel("Growth Rate")

# Part 5
for reg in df['region'].unique():
    plt.figure()
    plt.scatter(df['log1960'][df['region']==reg],df['growthrate'][df['region']==reg])
    plt.xlabel("log(GDPPC 1960)")
    plt.ylabel("Growth Rate")
    plt.title(reg)
    

# Part 6
edu25 = np.nanpercentile(df['edu1960'],25)
edu50 = np.nanpercentile(df['edu1960'],50)
edu75 = np.nanpercentile(df['edu1960'],75)

plt.figure()
plt.scatter(df['log1960'][df['edu1960']<=edu25],df['growthrate'][df['edu1960']<=edu25])
plt.xlabel("log (GDPPC 1960)")
plt.ylabel("Growth Rate")
plt.title("Growth Rate vs log(GDPPC 1960) Separated by 1960 Education Levels")

plt.scatter(df['log1960'][(edu25<= df['edu1960']) & (df['edu1960']<=edu50)],df['growthrate'][(edu25<= df['edu1960']) & (df['edu1960']<=edu50)])
plt.scatter(df['log1960'][(edu50<= df['edu1960']) & (df['edu1960']<=edu75)],df['growthrate'][(edu50<= df['edu1960']) & (df['edu1960']<=edu75)])
plt.scatter(df['log1960'][edu75<= df['edu1960']],df['growthrate'][edu75<= df['edu1960']])

plt.legend({"Bottom Quartile","Bottom Middle Quartile","Upper Middle Quartile","Upper Quartile"})

plt.show()