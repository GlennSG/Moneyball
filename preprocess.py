
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


data = pd.read_csv("baseball.csv")
data.head(3)


# In[72]:


data.shape


# In[73]:


data["Year"].unique()


# In[74]:


data.isnull().sum()


# In[75]:


data = data.drop(data.iloc[:,10:12],axis=1)
data.head(3)


# In[82]:


# display rows with NaN values
df1 = data[data.isnull().any(axis=1)]
#print(df1)


# In[77]:


data = data.dropna(how="any")
data.isnull().sum()


# In[78]:


data.shape


# In[79]:


# Runs differential
data["RD"] = data["RS"] - data["RA"]
data.head(3)


# ## Correlation Matrix
# 
# Pythagorean Expectation Formula: (Runs scored)^2 / (Runs scored)^2 + (Runs Allowed)^2 = Win % for team
# 
# Find independent features that correlate with RS, RA
# 
# Run Differential (cumulative statistic that combines offense and defense scoring) = RS-RA
# 
# RD can be used to predict the expected win total for a team.

# In[80]:


corrmat = data.corr()
sns.heatmap(corrmat)
plt.show()


# ## Pairplot (Histograms & Scatterplots)

# In[81]:


sns.pairplot(data,markers="+",kind="reg")
plt.show()

