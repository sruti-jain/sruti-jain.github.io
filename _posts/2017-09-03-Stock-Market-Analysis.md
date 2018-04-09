---
layout: post
title: Data Project - Stock Market Analysis
subtitle: Risk Analysis Monte Carlo
layout: post
tags: [Data Science, Monte Carlo, Python, Stock Market Analysis]
use_math: true
---

We have already learnt about Monte Carlo method in the previous post - [Monte Carlo](https://sruti-jain.github.io/2017-08-17-Monte-Carlo-Method/).
In this post I am going to discuss on how can that method be used for determining the maximum risk involved in investing to a stock.

```python
#import libraries
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# For reading stock data from yahoo
import pandas_datareader as pdr
from datetime import datetime
from __future__ import division
```


```python
import pandas_datareader.data as web
tech_list = ['AAPL','GOOG','MSFT','AMZN']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)

for stock in tech_list:   
    globals()[stock] = web.DataReader(stock,'yahoo',start,end)    
```


```python
#Descriptive & Exploratory analysis
AAPL.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>252.000000</td>
      <td>2.520000e+02</td>
      <td>252.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>115.796667</td>
      <td>116.627698</td>
      <td>115.155357</td>
      <td>115.973928</td>
      <td>3.193192e+07</td>
      <td>115.217969</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.948756</td>
      <td>15.931637</td>
      <td>15.978002</td>
      <td>15.994669</td>
      <td>1.433693e+07</td>
      <td>16.482845</td>
    </tr>
    <tr>
      <th>min</th>
      <td>90.000000</td>
      <td>91.669998</td>
      <td>89.470001</td>
      <td>90.339996</td>
      <td>1.147590e+07</td>
      <td>89.008370</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>104.767498</td>
      <td>105.959999</td>
      <td>104.060001</td>
      <td>105.499998</td>
      <td>2.352475e+07</td>
      <td>104.394087</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>113.225003</td>
      <td>114.055001</td>
      <td>112.389999</td>
      <td>113.069999</td>
      <td>2.818830e+07</td>
      <td>112.006823</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>128.062502</td>
      <td>129.665000</td>
      <td>127.875000</td>
      <td>128.832501</td>
      <td>3.588788e+07</td>
      <td>128.276348</td>
    </tr>
    <tr>
      <th>max</th>
      <td>147.539993</td>
      <td>148.089996</td>
      <td>146.839996</td>
      <td>147.509995</td>
      <td>1.119850e+08</td>
      <td>147.509995</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Let's see historical view of the closing price for the last one year
AAPL['Adj Close'].plot(legend=True,figsize=(10,4))
```
![png](/img/montecarlo/output_5_1.png)

```python
# Now let's plot the total volume of stock being traded each day over the past year
AAPL['Volume'].plot(legend=True,figsize=(10,4))
```
![png](/img/montecarlo/output_6_1.png)


```python
#Calculate and plot the simple moving average (SMA) for upto 10, 20 and 50 days
moving_avg_day = [10,20,50]

for ma in moving_avg_day:
    column_name = "MA for %s days" % (str(ma))
    AAPL[column_name] =pd.rolling_mean(AAPL['Adj Close'],ma)
    
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(15,6))
```

![png](/img/montecarlo/output_7_2.png)



```python
#Now lets analyse the Daily Return Analysis as our first step to determing the volatility of the stock prices

# We'll use pct_change to find the percent change for each day
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
# Then we'll plot the daily return percentage
AAPL['Daily Return'].plot(figsize=(12,4),legend=True,linestyle='--',marker='o')

```

```python
#Average daily return
#Using dropna to eliminate NaN values
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='red')
```

![png](/img/montecarlo/output_9_2.png)


```python
#Now lets do a comparative study to analyze the returns of all the stocks in our list

# Grab all the closing prices for the tech stocks into one new DataFrame
closing_df = web.DataReader(['AAPL','GOOG','MSFT','AMZN'],'yahoo',start,end)['Adj Close'] 
```


```python
# Let's take a quick look
closing_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-05-05</th>
      <td>91.865625</td>
      <td>659.090027</td>
      <td>701.429993</td>
      <td>48.660218</td>
    </tr>
    <tr>
      <th>2016-05-06</th>
      <td>91.353293</td>
      <td>673.950012</td>
      <td>711.119995</td>
      <td>49.098687</td>
    </tr>
    <tr>
      <th>2016-05-09</th>
      <td>91.422261</td>
      <td>679.750000</td>
      <td>712.900024</td>
      <td>48.786888</td>
    </tr>
    <tr>
      <th>2016-05-10</th>
      <td>92.042972</td>
      <td>703.070007</td>
      <td>723.179993</td>
      <td>49.712543</td>
    </tr>
    <tr>
      <th>2016-05-11</th>
      <td>91.146389</td>
      <td>713.229980</td>
      <td>715.289978</td>
      <td>49.741774</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make a new DataFrame for storing stock daily return
tech_rets = closing_df.pct_change()
```


```python
# Using joinplot to compare the daily returns of Google and Microsoft
sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter')
```




    <seaborn.axisgrid.JointGrid at 0x24cfc1bf080>




![png](/img/montecarlo/output_13_1.png)


The correlation coefficient of 0.6 indicates a strong correlation between the daily return values of Microsoft and Google. This inplies that if google's stock value increases, microsoft's stock value and hence daily return increases and vice versa. This analysis can further be extended by first segmentation of stock based on industry and then further analysis can be done to determine the choice of stock within that industry. 


```python
# We can also simply call pairplot on our DataFrame for visual analysis of all the comparisons
sns.pairplot(tech_rets.dropna())
```




    <seaborn.axisgrid.PairGrid at 0x24cfc6a4dd8>




![png](/img/montecarlo/output_15_1.png)



```python
#Another analysis on the closing prices for each stock and their individual comparisons
# Call PairPLot on the DataFrame
close_fig = sns.PairGrid(closing_df)

# Using map_upper we can specify what the upper triangle will look like.
close_fig.map_upper(plt.scatter,color='purple')

# We can also define the lower triangle in the figure, including the plot type (kde) or the color map (BluePurple)
close_fig.map_lower(sns.kdeplot,cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the closing price
close_fig.map_diag(plt.hist,bins=30)
```




    <seaborn.axisgrid.PairGrid at 0x24cfdf8a160>




![png](/img/montecarlo/output_16_1.png)



```python
#For a quick look at the correlation analysis for all stocks, we can plot for the daily returns using corrplot
from seaborn.linearmodels import corrplot as cor
cor(tech_rets.dropna(),annot=True)
```

    C:\Users\saj16\Anaconda3\lib\site-packages\seaborn\linearmodels.py:1290: UserWarning: The `corrplot` function has been deprecated in favor of `heatmap` and will be removed in a forthcoming release. Please update your code.
      warnings.warn(("The `corrplot` function has been deprecated in favor "
    C:\Users\saj16\Anaconda3\lib\site-packages\seaborn\linearmodels.py:1356: UserWarning: The `symmatplot` function has been deprecated in favor of `heatmap` and will be removed in a forthcoming release. Please update your code.
      warnings.warn(("The `symmatplot` function has been deprecated in favor "





    <matplotlib.axes._subplots.AxesSubplot at 0x24cfff485f8>




![png](/img/montecarlo/output_17_2.png)


**Risk Analysis: Bootstraping method, Monte Carlo method**


```python
#It is important to note the risk involved with the expected return for each stock visually before starting the analysis
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)

# Setting the x and y limits & plt axis titles
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])
plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
```


![png](/img/montecarlo/output_19_0.png)


As seen in the above plot the less the expected return, lesser the risk involved. Next lets determine the "Value at Risk", that is the worst daily loss that can be encountered with 95 % Confidence. 


```python
rets['AAPL'].quantile(0.05)
```




    -0.014682686093512198



The 0.05 empirical quantile of daily returns is at -0.015. That means that with 95% confidence, worst daily loss will not exceed 1.5%. If we have a 1 million dollar investment, our one day 5% VaR is 0.015 * 1,000,000 = $15,000.

**Value at Risk using the Monte Carlo method**

Using the Monte Carlo to run many trials with random market conditions, then we'll calculate portfolio losses for each trial. After this, we'll use the aggregation of all these simulations to establish how risky the stock is.
Let's start with a brief explanation of what we're going to do:

We will use the geometric Brownian motion (GBM), which is technically known as a Markov process. This means that the stock price follows a random walk and is consistent with (at the very least) the weak form of the efficient market hypothesis (EMH): past price information is already incorporated and the next price movement is "conditionally independent" of past price movements.
This means that the past information on the price of a stock is independent of where the stock price will be in the future, basically meaning, you can't perfectly predict the future solely based on the previous price of a stock.

The equation for geometric Browninan motion is given by the following equation:
$$\frac{\Delta{S}}{S}= {\mu \Delta{t}}+{\sigma \epsilon \sqrt{\Delta{t}}}$$

Where S is the stock price, mu is the expected return (which we calculated earlier),sigma is the standard deviation of the returns, t is time, and epsilon is the random variable.
We can mulitply both sides by the stock price (S) to rearrange the formula and solve for the stock price.
$${\Delta{S}}= S({\mu \Delta{t}}+{\sigma \epsilon \sqrt{\Delta{t}}})$$


Now we see that the change in the stock price is the current stock price multiplied by two terms. The first term is known as "drift", which is the average daily return multiplied by the change of time. The second term is known as "shock", for each tiem period the stock will "drift" and then experience a "shock" which will randomly push the stock price up or down. By simulating this series of steps of drift and shock thousands of times, we can begin to do a simulation of where we might expect the stock price to be.


```python
# Set up time horizon and delta values
days = 365
dt = 1/days

# Calculate mu (drift) from the expected return data we got for AAPL
mu = rets.mean()['GOOG']

# Calculate volatility of the stock from the std() of the average return
sigma = rets.std()['GOOG']
```


```python
def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation, mu, sigma and returns simulated price array'''

    price = np.zeros(days)
    price[0] = start_price    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Calculating and returning price array for number of days
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
    return price
```


```python
# Get start price from GOOG.head()
start_price = 560.85
simulations = np.zeros(100)

for run in range(100):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Google')
```




    <matplotlib.text.Text at 0x24c805d4278>




![png](/img/montecarlo/output_26_1.png)



```python
# Now we'lll define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');
```


![png](/img/montecarlo/output_27_0.png)


This basically means that for every initial stock you purchase there is about $13.61 at risk 1% of the time as predicted by Monte Carlo Simulation. 
