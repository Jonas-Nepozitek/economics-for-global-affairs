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

# 1E
USAind = df.index[df['countrycode'] == 'USA']
tfpUSA18 = A18[USAind].values[0]

# 1F
hypoy18 = (k18**a)*tfpUSA18
hypog = np.log(hypoy18/y60)/n
x3 = lny60
x3 = sm.add_constant(x3)
y3 = hypog
mod3 = sm.OLS(y3,x3,missing='drop')
res3 = mod3.fit(cov_type="HC0")
print(res3.summary(xname=['const','gdppc1960']))

# 2A
'''
Including human capital in the production function is valid because it provides a weight to the quality of 
labor in a country. Factors like education and skills affect how productive the workers of a nation can be. 
By including the human capital in the production function, it accounts more accurately for the variation in 
output without ascribing it to TFP. 
'''
# 2B
h18 = df['hc2018']
A218 = y18/(k18**a)/(h18**(1-a))
lnA218 = np.log(A218)

print(f"The mean of ln A in 2018 is: {np.mean(lnA218)}\nThe standard deviation of ln A in 2018 is: {np.std(lnA218)}")

# 2C
x4 = lnA218
x4 = sm.add_constant(x4)
y4 = lny18
mod4 = sm.OLS(y4,x4,missing='drop')
res4 = mod4.fit(cov_type="HC0")
print(res4.summary(xname=['const','ln A']))
plt.figure()
sns.regplot(x='lnA', y='lny', data=pd.DataFrame({'lnA' : lnA218, 'lny' : lny18}) ) 
'''
The correlation between ln y and ln A for the production function that accounts for 
human capital is higher than for the simplified model in Question 1 (R = 0.96 vs 0.94). The slope of the
regression between the variables is also steeper, suggesting that when human capital is accounted for using data, the
remaining factors that are part of TFP are better at explaining the differences in output per capita
that are shown in the data. 
'''

# 2 D
plt.figure()
plt.scatter(lnA18,lnA218)
plt.xlabel("ln(A) (No H)")
plt.ylabel("ln(A) (Account for H)")
'''
The plots of ln A show a very strong correlation, with R = 0.98. However, for all countries, the TFP calculated for 2018 using the human capital
were lower than when human capital is not accounted for. This makes sense, as by pulling human capital out of TFP and treating it as a separate input, the
direct contribution of TFP is lower (essentially, if the old TFP values included human capital, in the new calculations
that portion of TFP is taken out and treated as a standalone variable for which we have data, leaving less that needs to be accounted for by TFP).
'''

# 2E
'''
Augmenting the production function with human capital reduces our ignorance because, by definition, 
TFP includes all factors that aren't otherwise treated as an input in the production formula. By collecting data
on a factor and treating it as an input, as in the case of human capital, less of the variations in output have to be
explained by the otherwise rather opaque notion of TFP. The more that can be known about an economy's inputs, the less
has to be ascribed to productivity/TFP. This changes the recommendations, because, by identifying human capital and education
as an input, it provides another avenue by which a country or its policymakers can allocate resources and investments into growing its output
by improving the quality of its labor force instead of just relying on diminishing returns of capital. 
'''
plt.show()
