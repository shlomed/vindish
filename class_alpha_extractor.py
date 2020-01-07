
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
import torch
from torch import nn


# In[2]:


c_dir = "E:\\Datasets\\vindish\\"
df = pd.read_csv(c_dir+"VIX Historical data 10 Min level since 2017 UX1_UX5.csv")
print(df.shape)
df = df[df.apply(lambda x: x.notnull().all(), axis=1)]
print(df.shape)


# In[20]:


df.Date = pd.to_datetime(df.Date)


# In[21]:


df.Time_To_Expiration.describe()


# In[22]:


df.Time_To_Expiration.diff().describe()


# ### use raw ux data only - generator

# In[54]:


window_size = 10
batch_size = 128


# In[63]:


def get_df_t0_t1_y_filter_rule(df, i, window_size=window_size):
    df_t0 = df.iloc[i:i+window_size]
    df_t1 = df.iloc[i+1:i+window_size+1]
    y = df.iloc[i+window_size+1]
    
    filtering_rule1 = (y.Date - df_t0.Date.iloc[0]).total_seconds()//60 == (window_size+1)*10.
    filtering_rule2 = (y.Time_To_Expiration<df_t0.iloc[0].Time_To_Expiration)
    filtering_rule = filtering_rule1 and filtering_rule2
    
    return df_t0, df_t1, y, filtering_rule


# In[64]:


def get_random_batch(df, batch_size=batch_size, window_size=window_size):
    rand_idx = np.random.choice(np.arange(0, df.shape[0]-window_size-1), batch_size)
    
    dfs_t0 = []
    dfs_t1 = []
    ys = []

    for i in rand_idx:
        df_t0, df_t1, y, filtering_rule = get_df_t0_t1_y_filter_rule(df, i)
        if filtering_rule:
            dfs_t0.append(df_t0)
            dfs_t1.append(df_t1)
            ys.append(y)
            
    return dfs_t0, dfs_t1, ys

for i in range(100):
    dfs_t0, dfs_t1, ys = get_random_batch(df)
    print(128-len(dfs_t0))
# In[73]:


class AlphaExtractor(nn.Module):
    def __init__(self):
        super(AlphaExtractor, self).__init__()
        pass
    
    def forward(self, inputs):
        '''
        inputs: list of dfs with the following columns: [UX1(float),UX2(float),UX3(float),UX4(float),UX5(float)]
        '''
        pass


# In[72]:


a = AlphaExtractor()

