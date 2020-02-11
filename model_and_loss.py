
# coding: utf-8

# In[6]:


from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
import pandas as pd
import numpy as np


# In[2]:


from data_loader_pytorch_class import VindishDataset
dataset = VindishDataset()
dataloader = DataLoader(dataset, batch_size=128)


# In[7]:


a = next(iter(dataloader))
a[0].shape


# In[8]:


class AlphaExtractor(nn.Module):
    def __init__(self):
        super(AlphaExtractor, self).__init__()
        self.fc1 = nn.Linear(17, 5)
        
    def forward(self, x):
        return self.fc1(x[:, -1, :])
        


# In[9]:


alpha_extractor = AlphaExtractor()


# In[11]:


alpha_extractor(a[0]).shape


# In[ ]:


alpha_extractor.

