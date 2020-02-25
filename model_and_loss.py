
# coding: utf-8
!python -m pip install --upgrade pip
!pip install tb-nightly
# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
import pandas as pd
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from data_loader_pytorch_class import VindishDataset

dataset = VindishDataset(seed=None)
dataloader = DataLoader(dataset, batch_size=1024)
# In[ ]:


class AlphaExtractor(nn.Module):
    def __init__(self):
        super(AlphaExtractor, self).__init__()
        self.fc1 = nn.Linear(17, 5)
        
    def forward(self, x):
        return self.fc1(x[:, -1, :])


# In[ ]:


class AlphaExtractorFull(nn.Module):
    def __init__(self):
        super(AlphaExtractorFull, self).__init__()
        self.alpha_extractor = AlphaExtractor()
        self.fc2 = nn.Linear(5, 5) # sanity check for no_grad
        
    def forward(self, x1, x2):
        with torch.no_grad():
            alpha_tm2 = self.alpha_extractor(x1)
            alpha_tm2 = self.fc2(alpha_tm2) # sanity check for no_grad
        
        alpha_tm1 = self.alpha_extractor(x2)
        
        return alpha_tm2, alpha_tm1

alpha_extractor_full = AlphaExtractorFull()x_0_tm2, x_1_tm1, y_tm1, y_t = next(iter(dataloader))x_0_tm2.shape, x_1_tm1.shape, y_tm1.shape, y_t.shapealpha_tm2, alpha_tm1 = alpha_extractor_full(x_0_tm2, x_1_tm1)
# In[ ]:


def get_loss(alpha_tm2, alpha_tm1, y_tm1, y_t, cost_per_unit=0.1, desired_n_units=200.,
            w_profit=1., w_costs=1., w_hedge=1., w_tot_units=1.,
            betas = torch.tensor(np.array([1.,2.,3.,4.,5.])) ):

    L_profit = (alpha_tm1.mul(y_t - y_tm1)).sum(axis=1)
    L_costs = torch.abs(alpha_tm2-alpha_tm1).sum(axis=1)*cost_per_unit
    L_hedge = ((alpha_tm1*betas)**2).sum(dim=1)
    L_tot_units = (torch.abs(alpha_tm1).sum(axis=1)-desired_n_units)**2

#     print(L_profit.shape, L_costs.shape, L_hedge.shape, L_tot_units.shape)
    res = {}
    res["tot_loss"] = torch.mean( w_profit*L_profit + w_costs*L_costs + w_hedge*L_hedge + w_tot_units*L_tot_units ) 
    res["L_profit"] = L_profit
    res["L_costs"] = L_costs
    res["L_hedge"] = L_hedge
    res["L_tot_units"] = L_tot_units
    
    return res

alpha_extractor_full.zero_grad()loss = get_loss(alpha_tm2, alpha_tm1, y_tm1, y_t)loss.backward()optimizer = torch.optim.Adam(alpha_extractor_full.parameters(), lr=0.01)optimizer.step()class full_object():
    def __init__(self, HPs):
        self.HPs = HPs
        self.alpha_extractor = AlphaExtractor()
        self._train_extractor = AlphaExtractorFull(self.alpha_extractor)
        
        self.loss = self._get_loss
        self.optim = 1
        
        self.dataloader = 23
        self.schdualer = 15
        
        
    def train():
        pass
    
    def predict(x):
        return self.alpha_extractor(x)
# In[ ]:


def train_step(alpha_tm2, alpha_tm1, y_tm1, y_t, alpha_extractor_full, optimizer):
    alpha_extractor_full.train()
    optimizer.zero_grad()
    losses = get_loss(alpha_tm2, alpha_tm1, y_tm1, y_t)
    losses["tot_loss"].backward()#retain_graph=True)
    optimizer.step()
    return losses


# In[ ]:


torch.cuda.is_available()


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epochs = 10
batch_size = 4000
alpha_extractor_full = AlphaExtractorFull()
alpha_extractor_full.to(device)
optimizer = torch.optim.Adam(alpha_extractor_full.parameters(), lr=0.01)
dataset = VindishDataset(seed=None)

for epoch_num in range(epochs):
    print()
    running_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch_num, (x_0_tm2, x_1_tm1, y_tm1, y_t) in enumerate(dataloader):
        x_0_tm2, x_1_tm1, y_tm1, y_t = x_0_tm2.to(device), x_1_tm1.to(device), y_tm1.to(device), y_t.to(device)
        alpha_tm2, alpha_tm1 = alpha_extractor_full(x_0_tm2, x_1_tm1)
        losses = train_step(alpha_tm2, alpha_tm1, y_tm1, y_t, alpha_extractor_full, optimizer)
        running_loss = (batch_num*running_loss + losses["tot_loss"]) / (batch_num+1)
        str_epoch = f"epoch: {epoch_num}/{epochs}"
        str_batch = f"batch: {batch_num}/{len(dataloader)}"
        str_loss = f"total_loss: {running_loss:.2f}"
        str_L_profit = f"L_profit: {losses['L_profit'].mean().detach().numpy():.2f}"
        str_L_tot_units = f"L_tot_units: {losses['L_tot_units'].mean().detach().numpy():.2f}"
        str_L_hedge = f"L_hedge: {losses['L_hedge'].mean().detach().numpy():.2f}"
        str_L_costs = f"L_costs: {losses['L_costs'].mean().detach().numpy():.2f}"
        print("\r"+", ".join([str_epoch, str_batch, str_loss, str_L_profit, 
                         str_L_tot_units, str_L_hedge, str_L_costs]), flush=True, end="")

