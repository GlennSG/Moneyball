
# coding: utf-8

# # Data Preprocessing

# In[198]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[199]:


# import dataset 
data = pd.read_csv("baseball.csv")
data.head(3)


# In[200]:


# create a copy for data manipulation
df_copy = data
df_copy.head(2)


# In[201]:


# get rows & column counts for data
data.shape


# In[202]:


# see if there are no missing years between 1962 - 2012
data["Year"].unique()


# In[203]:


# get unique team names to keep track after dropping values
print(sorted(list(data["Team"].unique())))


# ## Handling Missing Data

# In[204]:


# find missing data values 
data.isnull().sum()


# In[205]:


# drop RankSeason & RankPlayoffs columns (988 missing values)
data = data.drop(data.iloc[:,10:12],axis=1)
data.head(3)


# In[206]:


# impute missing data for important columns (OOBP & OSLG)
icols = list(data.iloc[:,3:9].columns)
jcols = ["OOBP","OSLG"]


# In[207]:


# create dataframe with non-NaN data for OOBP & OSLG
df1 = data[icols]
df1.head(3)


# In[208]:


# create dataframe isolating OOBP & OSLG
df2 = data[jcols]
df2.head(3)


# In[209]:


# combine the two dataframes (df1,df2)
df3 = pd.concat([df1,df2],axis=1)
df3.head(3)


# In[210]:


# create new dataframe with df3 data (non-NaN)
notnans = df3[jcols].notnull().all(axis=1)
df_notnans = df3[notnans]
df_notnans.head(3)


# In[211]:


# split df_notnans into train & test sets for regression model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_notnans[icols],df_notnans[jcols],test_size=0.20,random_state=4)


# In[212]:


# use Linear Regression model to predict NaN values for OOBP & OSLG
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

regr_multirf = MultiOutputRegressor(LinearRegression())
regr_multirf.fit(X_train,y_train)
score = regr_multirf.score(X_test,y_test)
print("Prediction score: ",score)


# In[213]:


# create a copy df of df3 with NaNs data
df_nans = df3.loc[~notnans].copy()
df_nans.head(3)


# In[214]:


# predict NaN data
df_nans[jcols] = regr_multirf.predict(df_nans[icols])
df_nans.head(3)


# In[215]:


# apply prediction to df3
df3 = df3.fillna(df_nans[jcols])
df3.head(3)


# In[216]:


# check df3 for NaN values
df3.isnull().sum()


# In[217]:


# add Year and Team columns from original dataframe to df3
df3["Year"] = data["Year"]
df3["Team"] = data["Team"]
df3.head(3)


# In[218]:


# add "Run Differential" for predicting wins
df3["RD"] = data["RS"] - data["RA"]
df3.head(3)


# In[219]:


# check shape after imputing data, no data was dropped
df3.shape


# ## Pairplot (data distribution & correlation visuals)

# In[220]:


sns.pairplot(df3,markers="+",kind="reg")
plt.show()


# Disregarding "Year" column visual, the other columns appear to contain normally distributed data.

# ## Create new target columns for experimental prediction analysis

# In[221]:


# commented out for different analysis, not for Moneyball analysis
#df3["RS_Target"] = df3.groupby("Team")["RS"].shift(1)
#df3["RA_Target"] = df3.groupby("Team")["RA"].shift(1)
#df3["W_Target"] = df3.groupby("Team")["W"].shift(1)
#df3.head(3)


# ## Remove years 2003 & above for 1962 to 2002 analysis

# In[222]:


df3 = df3[df3["Year"]<=2002]


# In[223]:


# check if years are between 1962 to 2002
df3["Year"].unique()


# In[224]:


df3.head(3)


# In[225]:


# check data for significant loss of data (1232 vs. 932, 300 data loss)
df3.shape


# ## Boxplots (checking for outliers)

# In[226]:


# normalize large data ("RS","RA","W")
# tried using sklearn normalize, got errors w/ shape (how to use this properly?)
plt.rcParams['figure.figsize'] = (23,13)
def normalize(df):
    x = (df-df.min())/(df.max()-df.min())
    return x 

d_set1 = normalize(df3["RS"])
d_set2 = normalize(df3["RA"])
d_set3 = normalize(df3["W"])
d_set4 = normalize(df3["OBP"])
d_set5 = normalize(df3["SLG"])
d_set6 = normalize(df3["BA"])
d_set7 = normalize(df3["OOBP"])
d_set8 = normalize(df3["OSLG"])
data = [d_set1,d_set2,d_set3,d_set4,d_set5,d_set6,d_set7,d_set8]
col_names = ["","RS","RA","W","OBP","SLG","BA","OOBP","OSLG"]
y_pos = np.arange(len(col_names))
plt.boxplot(data,patch_artist=True)
plt.xticks(y_pos,col_names)
plt.title("Stat Distributions")
plt.show()


# There appears to be some outliers in the dataset. May need to remove outliers to get more accurate model. 

# In[227]:


# use Z-score to detect and remove outliers
from scipy import stats

z = np.abs(stats.zscore(df3.iloc[:,0:8]))


# In[228]:


# use Z-score threshold < 3
df3 = df3[(z<=2).all(axis=1)]


# In[229]:


# check for significant data loss (932 vs. 784, 148 data loss)
df3.shape


# In[230]:


# refit using z-score limits
d_set1 = normalize(df3["RS"])
d_set2 = normalize(df3["RA"])
d_set3 = normalize(df3["W"])
d_set4 = normalize(df3["OBP"])
d_set5 = normalize(df3["SLG"])
d_set6 = normalize(df3["BA"])
d_set7 = normalize(df3["OOBP"])
d_set8 = normalize(df3["OSLG"])
data = [d_set1,d_set2,d_set3,d_set4,d_set5,d_set6,d_set7,d_set8]
col_names = ["","RS","RA","W","OBP","SLG","BA","OOBP","OSLG"]
y_pos = np.arange(len(col_names))
plt.boxplot(data,patch_artist=True)
plt.xticks(y_pos,col_names)
plt.title("Stat Distributions")
plt.show()

