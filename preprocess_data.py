
# coding: utf-8

# # Data Preprocessing

# In[168]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[169]:


data = pd.read_csv("baseball.csv")
data.head(3)


# In[170]:


df_copy = data
df_copy.head(2)


# In[171]:


data.shape


# In[172]:


data["Year"].unique()


# In[173]:


print(sorted(list(data["Team"].unique())))


# In[174]:


data.isnull().sum()


# In[175]:


# drop RankSeason & RankPlayoffs columns
data = data.drop(data.iloc[:,10:12],axis=1)
data.head(3)


# In[176]:


# impute missing data
icols = list(data.iloc[:,3:9].columns)
jcols = ["OOBP","OSLG"]


# In[177]:


# create dataframe with non-NaN data that is relevant to columns with NaN data
df1 = data[icols]
df1.head(3)


# In[178]:


# create dataframe with columns containing NaN values
df2 = data[jcols]
df2.head(3)


# In[179]:


# combine the two dataframes
df3 = pd.concat([df1,df2],axis=1)
df3.head(3)


# In[180]:


# create new dataframe with non-NaN data
notnans = df3[jcols].notnull().all(axis=1)
df_notnans = df3[notnans]
df_notnans.head(3)


# In[181]:


# split df_notnans into train & test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_notnans[icols],df_notnans[jcols],test_size=0.25,random_state=4)


# In[182]:


# use Linear Regression model to predict NaN values
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

regr_multirf = MultiOutputRegressor(LinearRegression())
regr_multirf.fit(X_train,y_train)
score = regr_multirf.score(X_test,y_test)
print("Prediction score: ",score)


# In[183]:


# create a copy df of df3 with NaNs data
df_nans = df3.loc[~notnans].copy()
df_nans.head(3)


# In[184]:


# predict NaN data
df_nans[jcols] = regr_multirf.predict(df_nans[icols])
df_nans.head(3)


# In[185]:


# apply prediction to df3
df3 = df3.fillna(df_nans[jcols])
df3.head(3)


# In[186]:


# check df3 for NaN values
df3.isnull().sum()


# In[187]:


# add Year and Team columns from original dataframe
df3["Year"] = data["Year"]
df3["Team"] = data["Team"]
df3.head(3)


# In[188]:


# add Run Differential for predicting wins
df3["RD"] = data["RS"] - data["RA"]
df3.head(3)


# ## Correlation Matrix

# In[189]:


plt.rcParams['figure.figsize'] = (23,13)

corrmat = df3.corr()
sns.heatmap(corrmat)
plt.show()


# Columns of interest for analysis with most correlated features (0.8-0.9 + range):
# 
# 1. Runs Scored (RS) : OBP, SLG, BA
# 
# 2. Runs Allowed (RA): OOBP, OSLG
# 
# 3. Wins (W): RD

# ## Pairplot (data distribution & correlation visuals)

# In[190]:


sns.pairplot(df3,markers="+",kind="reg")
plt.show()


# Disregarding "Year" column visual, the other columns appear to contain normally distributed data.

# ## Create new target columns for predictions in regression model

# In[191]:


#df3["RS_Target"] = df3.groupby("Team")["RS"].shift(1)
#df3["RA_Target"] = df3.groupby("Team")["RA"].shift(1)
#df3["W_Target"] = df3.groupby("Team")["W"].shift(1)
#df3.head(3)


# In[192]:


df3 = df3[df3["Year"]<=2002]


# In[193]:


df3.head(3)


# In[194]:


df3.shape


# In[195]:


# have some NaN values
df3.isnull().sum()


# In[196]:


# five values can be removed/dropped
df3[df3.isnull().any(axis=1)]


# In[197]:


df3 = df3.dropna(how="any")
df3.isnull().sum()


# In[198]:


with pd.option_context('display.max_rows', None, 'display.max_columns', 15):
    display(df3)


# ## Boxplots (checking for outliers)

# In[199]:


# normalize large data ("RS","RA","W")
# tried using sklearn normalize, got errors w/ shape (how to use this properly?)
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

# In[200]:


# use Z-score to detect and remove outliers
from scipy import stats

z = np.abs(stats.zscore(df3.iloc[:,0:8]))


# In[201]:


# use Z-score threshold < 3
df3 = df3[(z<=2).all(axis=1)]


# In[202]:


df3.shape


# In[203]:


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


# ## Check columns after processing

# In[204]:


df3["Year"].unique()


# In[205]:


print(sorted(list(df3["Team"].unique())))


# In[206]:


df_copy.head(3)


# Removed Miami Marlins (MIA, 2012 to present), Tampa Bay Rays (TBR, 2008 to present), Washington Nationals (WSN, 2005 to present) because focus of analysis is between 1965 to 2002. 
