
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import add_features

from statsmodels.tsa.stattools import adfuller

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


c_dir = "E:\\Datasets\\vindish\\"
e_dir = "E:\\Work/Vindish/created_samples/"


# In[10]:


df = pd.read_csv(c_dir+"VIX Historical data 10 Min level since 2017 UX1_UX5.csv")


# In[11]:


df.head()


# In[12]:


# df[df.UX1.isnull()].head()
print(df.shape)
# df = df[(df.UX1.notnull())&(df.UX2.notnull())&(df.UX3.notnull())&(df.UX4.notnull())&(df.UX5.notnull())]
df = df[df.apply(lambda x: x.notnull().all(), axis=1)]
print(df.shape)


# In[13]:


df.iloc[:, 2:7].plot()


# In[14]:


df.iloc[0:50000:5000, 2:7].T.plot()


# In[ ]:


df = add_features.drop_rows_with_null_dates(df)


# In[ ]:


df.Date = pd.to_datetime(df.Date, dayfirst=True)


# In[ ]:


df = pd.concat([df, add_features.from_date_get_dow_dom_doy(df.Date)], axis=1)


# In[ ]:


df.head(100)


# In[ ]:


diff_cols = [i+"_diff" for i in df.iloc[:, 2:8].columns]
df[diff_cols] = df.iloc[:, 2:8].diff()
df = df.iloc[1:, :]


# In[ ]:


df["day_of_month"] = df.Date.dt.day
df["day_of_week"] = df.Date.dt.dayofweek
df["day_of_year"] = df.Date.dt.dayofyear
df["time_of_day"] = df.Date.dt.hour

df_dow = pd.get_dummies(df.day_of_week)
df_dow_columns = ["dow_"+str(i) for i in df.day_of_week.unique().tolist()]
df_dow.columns = df_dow_columns
# df_dow
df_dom = pd.get_dummies(df.day_of_month)
df_dom_columns = ["dom_"+str(i) for i in df.day_of_month.unique().tolist()]
df_dom.columns = df_dom_columnsdf = pd.concat([df, df_dow, df_dom], axis=1)df.iloc[:, 2:14].head() # check diffs
# In[ ]:


df.memory_usage().sum()//1000000

df.day_of_week.value_counts(dropna=False)adfuller(df.UX2.diff().values[1:])adfuller(df.UX2)
# In[ ]:


rol = df.iloc[:, 2:7].rolling(window=1000)


# In[ ]:


x = df.iloc[:11]


# In[ ]:


rol.mean().plot()


# In[ ]:


ux_vals = x[["UX1", "UX2", "UX3", "UX4", "UX5"]].iloc[:10].values
ux_vals

dow_dummy = pd.get_dummies(x.day_of_week)#, columns=days_list)dow_dummypd.get_dummies(x.day_of_month)
# In[ ]:


df.dow.unique()


# In[ ]:


df.head()

def create_xy_from_df(x, verbose=False):
    time_to_expiration = np.expand_dims(x.iloc[:10].loc[:, "Time_To_Expiration"].values, 1)
    ux_vals = x.iloc[:10].loc[:, ['UX1', 'UX2', 'UX3', 'UX4','UX5']].values
    snp_vals = np.expand_dims(x.iloc[:10].loc[:, "SP500"].values, 1)
    day_of_year = np.expand_dims(x.iloc[:10].loc[:, "day_of_year"].values, 1)

    day_of_week_dummies = x.iloc[:10].loc[:, df_dow_columns].values

    day_of_month_dummies = x.iloc[:10].loc[:, df_dom_columns].values

    diffs = x.iloc[:10].loc[:, diff_cols[:-1]].values
    diff_snp = np.expand_dims(x.iloc[:10].loc[:, diff_cols[-1]].values, 1)

    if verbose:
        print("ux_vals.shape: ", ux_vals.shape)
        print("diffs.shape: ", diffs.shape)
        print("snp_vals.shape: ", snp_vals.shape)
        print("diff_snp.shape: ", diff_snp.shape)
        print("time_to_expiration.shape: ", time_to_expiration.shape)
        print("day_of_year.shape: ", day_of_year.shape)
        print("day_of_week_dummies.shape: ", day_of_week_dummies.shape)
        print("day_of_month_dummies.shape: ", day_of_month_dummies.shape)

    X = np.concatenate([ux_vals, diffs, snp_vals, diff_snp, time_to_expiration, day_of_year, day_of_week_dummies, day_of_month_dummies], axis=1)
    y = x[["UX1", "UX2", "UX3", "UX4", "UX5"]].iloc[10].values
    
    return X,y
# In[ ]:


def create_xy_from_df(x, verbose=False):
    x = x.drop("Date", axis=1)
    X = x.iloc[:10, :].values
    y = x[["UX1", "UX2", "UX3", "UX4", "UX5"]].iloc[10].values
    
    return X,y

i=11
X, y = create_xy_from_df(df.iloc[i-11:i, :], verbose=True)Xx = df.iloc[i-11:i, :]
dow_list = df.Date.dt.dayofweek.unique().tolist()
print(dow_list)
x.iloc[:10]#.loc[:, dow_list].values
# In[ ]:


get_ipython().run_cell_magic('time', '', 'if os.path.exists(e_dir + "X.npy") and os.path.exists(e_dir + "y.npy"):\n    X = np.load(e_dir + "X.npy")\n    y = np.load(e_dir + "y.npy")\nelse:    \n    Xs = []; ys=[]\n    for i in tqdm(np.arange(11, df.shape[0]-1)):\n    #     print(df.iloc[(i-11):i, :].shape)\n        X, y= create_xy_from_df(df.iloc[(i-11):i, :])\n        Xs.append(X)\n        ys.append(y)\n    X = np.stack(Xs)\n    y = np.stack(ys)\n    np.save(e_dir + "X.npy", X)\n    np.save(e_dir + "y.npy", y)\n    np.save(e_dir + "features.npy", np.array([i for i in df.columns if i!="Date"]))')


# In[ ]:


X.shape, y.shape

beta(5) - given from Moses
y(5) - given from dataset
X -> alpha(5) -> maximize(returns) under constraints. if possible minimize std of returns over dataset.

constraint 1: alpha*beta=0: might be a soft constraint?

returns: alpha*y - alpha*X[9, 0:4]

sum_i ( alpha_i(t)*( UX_i(t+1) - UX_i(t) ) )

modified returns to include constraints: (alpha*X[9, 0:4] - alpha*y) + a*(alpha*beta)^2 + b*(sum(abs(alpha))-200)^2pd.DataFrame(X[0])