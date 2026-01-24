import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import seaborn as sns


dataset = "https://raw.githubusercontent.com/akhandelwal8/globaleconomics/refs/heads/main/hwk/hwk2_accounting.csv"
df = pd.read_csv(dataset,sep='\t')
a = 0.3

# 1A, y = Ak**0.3

# 1B
k60 = df['cn1960']/df['pop1960']
y60 = df['cgdpo1960']/df['pop1960']
y18 = df['cgdpo2018']/df['pop2018']
k18 = df['cn2018']/df['pop2018']
A60 = y60/(k60**a)
A18 = y18/(k18**a)

lnk60 = np.log(k60)
lny60 = np.log(y60)
lnA60 = np.log(A60)
lny18 = np.log(y18)
lnk18 = np.log(k18)
lnA18 = np.log(A18)

# 1B
n = 58.0
g = np.log(y18/y60)/n

x = lny60
x = sm.add_constant(x)
y = g
model1 = sm.OLS(y,x,missing='drop')
res1 = model1.fit(cov_type="HC0")
print(res1.summary(xname=['const','gdppc1960']))
# 1C

print(f"The mean of ln A in 1960 is: {np.mean(lnA60)}\nThe standard deviation of ln A in 1960 is: {np.std(lnA60)}")

# 1D
x1 = lnA60
x1 = sm.add_constant(x1)
y1 = lny60
mod2 = sm.OLS(y1,x1,missing='drop')
res2 = mod2.fit(cov_type="HC0")
print(res2.summary(xname=['const','ln A']))


plt.figure()
sns.regplot(x='lnA', y='lny', data=pd.DataFrame({'lnA' : lnA60, 'lny' : lny60}) ) 
plt.show()


# 1E
USAind = df.index[df['countrycode'] == 'USA']

tfpUSA18 = A18[USAind]

# 1F
hypoy18 = (k18**a)*tfpUSA18
hypog = np.log(hypoy18/y60)/n

x3 = lny60
x3 = sm.add_constant(x3)
y3 = hypog
mod3 = sm.OLS(y3,x3,missing='drop')
res3 = mod3.fit(cov_type="HC0")
print(res3.summary(xname=['const','gdppc1960']))