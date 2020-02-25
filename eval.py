
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# from torch.utils.tensorboard import SummaryWriter

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from data_loader_pytorch_class import VindishDataset


# In[ ]:


df_all = pd.read_pickle('E:\\Datasets\\vindish\\df_all.pkl')


# In[ ]:


df_all = df_all.sort_values("Date")


# In[ ]:


n_train = int(0.8*df_all.shape[0])
df_train = df_all.iloc[:n_train]
df_test = df_all.iloc[n_train:]


# In[ ]:


dataset_train = VindishDataset(df_train, is_test=False)
dataset_test = VindishDataset(df_test, is_test=True)


# In[ ]:


dataloader = DataLoader(dataset_train, batch_size=1024)
next(iter(dataloader))


# In[ ]:


dataset_test.__getitem__()


# In[ ]:


df_test

