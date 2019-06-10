
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"


# In[ ]:


c_dir = "C://Users/shlomi/Documents/Work/vindish/data/"
e_dir = "E:\\Work/Vindish/created_samples/"


# In[ ]:


BATCH_SIZE = 64
betas = torch.from_numpy(np.array([0.62, 0.44, 0.32, 0.26, 0.21])).type(torch.Tensor).to(device)


# In[ ]:


X = torch.tensor(np.load(e_dir + "X.npy")).type(torch.float32)
y = torch.tensor(np.load(e_dir + "y.npy")).type(torch.float32)
features = np.load(e_dir+"features.npy")


# In[ ]:


print("X.shape, y.shape", X.shape, y.shape)
print("X.type(), y.type()", X.type(), y.type())


# In[ ]:


print("\n\nFeatures locations:")
for i, j in enumerate(features):
    print(i, j)


# In[ ]:


# subtruct 1 so the dom will be in the [0,30] range for embeddings
X[:,:,8] -= 1


# In[ ]:


n_train = int(X.shape[0]*0.6)
n_val = int(X.shape[0]*0.8)

X_train = X[:n_train]
y_train = y[:n_train]

X_val = X[n_train:n_val]
y_val = y[n_train:n_val]

# dropping all end of period samples:
ser = pd.Series(X_train[:, -1, 0].detach().numpy())
idx_to_keep = ser[ser>0.3].index.values
X_train = X_train[idx_to_keep]
y_train = y_train[idx_to_keep]

ser = pd.Series(X_val[:, -1, 0].detach().numpy())
idx_to_keep = ser[ser>0.3].index.values
X_val = X_val[idx_to_keep]
y_val = y_val[idx_to_keep]


X_test = X[n_val:]
y_test = y[n_val:]


# In[ ]:


train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = TensorDataset(X_val, y_val)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=1)


# In[ ]:


print("\ncreated train_dl, val_dl, test_dl")

