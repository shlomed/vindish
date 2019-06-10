
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import add_features

from statsmodels.tsa.stattools import adfuller
from tensorboardX import SummaryWriter
writer = SummaryWriter()


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import keras
import torch
import torchvision


# ### Data Loader

# In[ ]:


from data_loader import *


# ### Define Model

# In[ ]:


from models import *

model = Model()
model = model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.33, verbose=True)


# In[ ]:


from glob import glob
PATH = glob("chkpnts/model*.tar")[-1]
print(PATH)

model.load_state_dict(torch.load(PATH))


# ### Backtesting with Shlomo

# In[ ]:


class Result():
    pass

    def to_dict(self):
        return self.__dict__


# In[ ]:


def torch_to_numpy(t):
    try:
        t = t.detach()
    except:
        pass
    
    try:
        t = t.cpu()
    except:
        pass
    
    try:
        t = t.numpy()
    except:
        pass
    
    return t


# In[ ]:


def change_alphas(alphas_prev, alphas_cur, y_prev, y_cur, out_after_transaction_prev, fee=0.01):
    """
    we are now at time=t. we have X[t] to derive alphas[t]
    alphas_prev = derived from X[t-1] => alphas_prev = alphas[t-1]
    
    """
    
    
    # tensors to numpy:
    alphas_cur = torch_to_numpy(alphas_cur)
    alphas_prev = torch_to_numpy(alphas_prev)
    y_cur = torch_to_numpy(y_cur)
    y_prev = torch_to_numpy(y_prev)
    
    # calc result
    result = Result()
    result.buy_sell_expenses = np.abs(alphas_cur-alphas_prev).sum()*fee
    result.execution = (alphas_cur-alphas_prev)@y_cur
    
#     print((y_cur - y_prev).shape)
#     print(alphas_prev.shape)
    result.pure_pnl = ((y_cur - y_prev)@alphas_prev.flatten()).sum()
    result.real_pnl = result.pure_pnl - result.buy_sell_expenses
    
    result.in_after_transaction = alphas_cur@y_cur
    result.out_after_transaction = out_after_transaction_prev +                                     (alphas_prev - alphas_cur)@y_cur                                     - np.abs(alphas_cur-alphas_prev).sum()*fee
        
    return result


# In[ ]:


res0 = Result()
res0.in_after_transaction = 0.
res0.out_after_transaction = 0.
all_results = [res0]
model.to("cpu")

i = 0

while True:
    x = X_test[i]
    if x[:, 0].min()>0.3:
        break
    i += 1
        
x = X_test[i]
alphas_cur = model(x.unsqueeze_(0)) * 0.

for x in tqdm(X_test[i+1:]):
#     print(x[:, 8].min(), x[:, 8].max(),)
    alphas_prev = torch_to_numpy(alphas_cur)
    alphas_cur = torch_to_numpy(model(x.unsqueeze(0)))
    y_prev = x[-2, 1:6]
    y_cur = x[-1, 1:6]
    
    if x[:, 0].min()<0.3:
        alphas_cur = np.zeros(5)
#         flag_rebuy = True
        
#     if flag_rebuy and x[:, 0].min() > 0.3:
#         flag_rebuy=False

#     print()
#     print(alphas_cur, alphas_prev)

#     print(alphas_prev, alphas_cur, y_prev, y_cur, all_results[-1].out_before_transaction)
    all_results.append(change_alphas(alphas_prev, alphas_cur, y_prev, y_cur, all_results[-1].out_after_transaction))


# In[ ]:


for i in all_results:
    print(i.in_after_transaction, i.out_after_transaction)

