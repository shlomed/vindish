
# coding: utf-8

# In[66]:


import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# from torch.utils.tensorboard import SummaryWriter

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[67]:


from data_loader_pytorch_class import VindishDataset


# In[68]:


df_all = pd.read_pickle('E:\\Datasets\\vindish\\df_all.pkl')


# In[69]:


df_all = df_all.sort_values("Date")


# In[70]:


n_train = int(0.8*df_all.shape[0])
df_train = df_all.iloc[:n_train]
df_test = df_all.iloc[n_train:]


# In[71]:


dataset_train = VindishDataset(df_train, is_test=False)
dataset_test = VindishDataset(df_test, is_test=True)


# In[72]:


dataloader = DataLoader(dataset_train, batch_size=1024)
next(iter(dataloader))


# In[73]:


dataset_test.__getitem__()


# In[22]:


df_test

