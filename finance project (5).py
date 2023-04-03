#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas_datareader


# In[2]:


#for reading stock data from investors exchange
import pandas_datareader as pdr
from datetime import datetime


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf
import os


# In[4]:


tech_stocks=['AAPL','GOOG','MSFT','AMZN']


# In[5]:


#set end and start times for data load
end=datetime.now()


# In[6]:


end


# In[7]:


start=datetime(end.year-1,end.month,end.day)


# In[8]:


start


# In[9]:


os.environ["IEX_API_KEY"] = "sk_5bdb6470e57d4819a152b9fc1c6508ff"


# In[10]:


#for loop for grabing iex finance data and setting as data frame
for stock in tech_stocks:
    stock_data=pdr.DataReader(stock,'iex',start,end,api_key=os.getenv('IEX_API_KEY'))


# In[11]:


stock_data


# In[12]:


for stock in tech_stocks:
    globals()[stock]=pdr.DataReader(stock,'iex',start,end,api_key=os.getenv('IEX_API_KEY'))


# In[13]:


stock


# In[14]:


AAPL.head()


# In[15]:


GOOG.head()


# In[16]:


MSFT.head()


# In[17]:


AMZN.head()


# In[18]:


AAPL.count()


# In[19]:


GOOG.count()


# In[20]:


MSFT.count()


# In[21]:


AMZN.count()


# In[22]:


AAPL.describe()


# In[23]:


GOOG.describe()


# In[24]:


MSFT.describe()


# In[25]:


AMZN.describe()


# # Historical view of closing price

# In[26]:


AAPL['close'].plot(legend=True,figsize=(12,6))


# In[27]:


AMZN['close'].plot(legend=True,figsize=(12,6))


# In[28]:


MSFT['close'].plot(legend=True,figsize=(12,6))


# In[29]:


GOOG['close'].plot(legend=True,figsize=(12,6))


# # Total volume of stock being traded over the past 5 years

# In[30]:


GOOG['volume'].plot(legend=True,figsize=(12,6))


# In[31]:


MSFT['volume'].plot(legend=True,figsize=(12,6))


# In[32]:


AMZN['volume'].plot(legend=True,figsize=(12,6))


# In[33]:


AAPL['volume'].plot(legend=True,figsize=(12,6))


# # Moving Average for stocks
# Moving Average is widely use indicator in techinical analysis that helps smooth out price action by filtering the "noise" from the random price fluctuations.It is trend following or lagging,indicator because it is based on past prices

# In[34]:


ma_days=[10,20,50]
for ma in ma_days:
    col_name="MA for %s days" %(str(ma))
    AAPL[col_name]=pd.DataFrame.rolling(AAPL['close'],ma).mean()


# In[35]:


AAPL.head(50)


# In[36]:


AAPL[['close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(12,6))
plt.show()


# # Use perctage change for each day

# In[37]:


AAPL['Daily Return']=AAPL['close'].pct_change()


# In[38]:


AAPL.head()


# In[39]:


AAPL['Daily Return'].plot(figsize=(12,6),legend=True,linestyle='--',marker='o')


# # Average Daily Returns

# In[40]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='blue')


# # Returns of all stocks in a list

# In[41]:


closing_df1=pd.DataFrame(AAPL['close'])
close1=closing_df1.rename(columns={"close":"AAPL_close"})

closing_df2=pd.DataFrame(GOOG['close'])
close2=closing_df2.rename(columns={"close":"GOOG_close"})

closing_df3=pd.DataFrame(MSFT['close'])
close3=closing_df3.rename(columns={"close":"MSFT_close"})

closing_df4=pd.DataFrame(AMZN['close'])
close4=closing_df4.rename(columns={"close":"AMZN_close"})

closing_df=pd.concat([close1,close2,close3,close4],axis=1)
closing_df.head()


# # Daily Return for all stocks 

# In[42]:


tech_returns=closing_df.pct_change()


# # Compare daily percentage returns of two stocks and check how correlated.

# In[43]:


sns.jointplot('GOOG_close','GOOG_close',tech_returns,kind='scatter',color='seagreen')


# In[44]:


sns.jointplot('GOOG_close','MSFT_close',tech_returns,kind='scatter',color='seagreen')


# In[45]:


sns.pairplot(tech_returns.dropna())


# In[46]:


corr=tech_returns.dropna().corr()


# In[47]:


sns.heatmap(corr,annot=True)

