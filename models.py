
# coding: utf-8

# In[1]:


import keras
import torch
import torchvision
import numpy as np

PRINT = False
GAMMA = 1.


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"


# c_dir = "C://Users/shlomi/Documents/Work/vindish/data/"  
# e_dir = "E:\\Work/Vindish/created_samples/"

# X = torch.tensor(np.load(e_dir + "X.npy")).type(torch.float32)  
# y = torch.tensor(np.load(e_dir + "y.npy")).type(torch.float32)  
# features = np.load(e_dir+"features.npy")  

# ##### subtruct 1 so the dom will be in the [0,30] range for embeddings
# X[:,:,8] -= 1

# In[ ]:


class MiniConv1d(torch.nn.Module):
    def __init__(self, init_kernel_size=(3, 2)):
        super(MiniConv1d, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, init_kernel_size, padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(5, 5, (3, 3), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(5, 5, (3, 3), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(5, 5, (3, 3), padding=(1, 1))
        self.conv5 = torch.nn.Conv2d(5, 5, (3, 3), padding=(1, 1))
        self.conv6 = torch.nn.Conv2d(5, 5, (3, 3), padding=(1, 1))
        self.conv7 = torch.nn.Conv2d(5, 5, (3, 3), padding=(1, 1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.conv6(x)
        x = torch.nn.functional.relu(x)
        x = self.conv7(x)
        x = torch.nn.functional.relu(x)
        
        return x


# In[ ]:


class MiniConv2d(torch.nn.Module):
    def __init__(self):
        super(MiniConv2d, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 5, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(5, 5, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(5, 5, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(5, 5, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(5, 5, 3, padding=1)
        self.conv7 = torch.nn.Conv2d(5, 5, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.conv6(x)
        x = torch.nn.functional.relu(x)
        x = self.conv7(x)
        x = torch.nn.functional.relu(x)
        
        return x


# mini_conv2d = MiniConv2d()
# mini_conv2d(X[:3, :5].unsqueeze_(1))

# In[ ]:


class Embeddings(torch.nn.Module):
    def __init__(self, n_categories, n_dims):
        super(Embeddings, self).__init__()
        self.embed = torch.nn.Embedding(n_categories, n_dims)
        
    def forward(self, x):
        x = self.embed(x)
        
        return x


# In[ ]:


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mini_conv_ux = MiniConv2d()
        self.mini_conv_ux_diffs = MiniConv2d()
        self.mini_conv_snp = MiniConv1d()
        self.mini_conv_singles = MiniConv1d(init_kernel_size=(3,3))
        self.embed_dow = Embeddings(7, 3)
        self.embed_dom = Embeddings(32, 5)
        
        self.fc1 = torch.nn.Linear(880, 64)
        self.fc2 = torch.nn.Linear(64, 31)
        self.fc3 = torch.nn.Linear(31, 5)
        
        
    def forward(self, x):
        ux_vals = x[:, :, 1:6].unsqueeze_(1)
        ux_diffs = x[:, :, 10:15].unsqueeze_(1)
        snp_data = x[:, :, [6,15]].unsqueeze_(1)
        singles = x[:, :,  [0, 18, 19]].unsqueeze_(1) # time_to_expiration, doy, time_of_day
        
        dow = x[:, :, 17].type(torch.long)
        dom = x[:, :, 16].type(torch.long)
        
        x_ux = self.mini_conv_ux(ux_vals).view(x.shape[0], -1)
        x_diffs = self.mini_conv_ux_diffs(ux_diffs).view(x.shape[0], -1)
        x_snp = self.mini_conv_snp(snp_data).view(x.shape[0], -1)
        x_dow = self.embed_dow(dow).view(x.shape[0], -1)
        x_dom = self.embed_dom(dom).view(x.shape[0], -1)
        x_singles = self.mini_conv_singles(singles).view(x.shape[0], -1)
        
        x = torch.cat((x_ux, x_diffs, x_snp, x_dom, x_dow, x_singles), 1)
        
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def get_profit(y, x, alphas):
    if PRINT:
        print("Xs:\n", x[0].cpu())
        print("ys:\n", y[0].cpu())

    L3 = ((y - x)*alphas).sum(dim=1)

    if PRINT:
        print("L3s:\n", L3[:2].cpu())
    return L3

def get_dist_from_200(alphas):
    return (alphas.abs().sum(dim=1)-200.)

def get_hedging_score(alphas, betas):
    return (alphas*betas).sum(dim=1)

def calc_loss(alphas, betas, x_batch, y_batch):
    a = 1
    b = 10
    c = 10000
    
    L1 = get_dist_from_200(alphas)**2
    L2 = get_hedging_score(alphas, betas)**2 
    L3 = get_profit(y_batch, x_batch[:, -1, 1:6], alphas)
    L = a*L1 + b*L2 - c*L3def get_profit(y, x, alphas):
    if PRINT:
        print("Xs:\n", x[0].cpu())
        print("ys:\n", y[0].cpu())

    L3 = ((y - x)*alphas).sum(dim=1)

    if PRINT:
        print("L3s:\n", L3[:2].cpu())
    return L3

def get_dist_from_200(alphas):
    return (alphas.abs().sum(dim=1)-200.)

def get_hedging_score(alphas, betas):
    return (alphas*betas).sum(dim=1)

def calc_loss(alphas, betas, x_batch, y_batch):
    a = 1
    b = 10
    c = 10000
    
    L1 = get_dist_from_200(alphas)**2
    L2 = get_hedging_score(alphas, betas)**2 
    L3 = get_profit(y_batch, x_batch[:, -1, 1:6], alphas)
    L = a*L1 + b*L2 - c*L3
    
#     print(L1.size(), L2.size(), L3.size())
    
    return L.sum()
    
#     print(L1.size(), L2.size(), L3.size())
    
    return L.sum()
# In[ ]:


def get_profit(y, x, alphas):
    if PRINT:
        print("Xs:\n", x[0].cpu())
        print("ys:\n", y[0].cpu())

    L3 = ((y - x)*alphas).sum(dim=1)

    if PRINT:
        print("L3s:\n", L3[:2].cpu())
    return L3

def get_dist_from_200(alphas):
    return (alphas.abs().sum(dim=1)-200.)

def get_hedging_score(alphas, betas):
    return (alphas*betas).sum(dim=1)

def update_kpis(avg_kpi_alpha, avg_kpi_beta, avg_kpi_200, dist_from_200, dist_from_beta_0, profit):
    kpi_200 = np.min(np.abs(dist_from_200/WINDOW_200), 0.9999)
    kpi_beta = np.min(np.abs(dist_from_beta_0/WINDOW_BETA), 0.9999)
    kpi_alpha = np.min(np.abs(profit/WINDOW_PROFIT), 0.9999)

    avg_kpi_alpha = alpha*avg_kpi_alpha + (1-alpha)*kpi_alpha
    avg_kpi_beta = alpha*avg_kpi_beta + (1-alpha)*kpi_beta
    avg_kpi_200 = alpha*avg_kpi_200 + (1-alpha)*kpi_200
    
    return avg_kpi_alpha, avg_kpi_beta, avg_kpi_200

def calc_loss(alphas, betas, x_batch, y_batch, avg_kpi_alpha, avg_kpi_beta, avg_kpi_200):
    a = -(1-avg_kpi_200)**GAMMA * np.log(avg_kpi_200)
    b = -(1-avg_kpi_200)**GAMMA * np.log(avg_kpi_beta)
    c = -(1-avg_kpi_200)**GAMMA * np.log(avg_kpi_alpha)
    
    L1 = get_dist_from_200(alphas)**2
    L2 = get_hedging_score(alphas, betas)**2 
    L3 = get_profit(y_batch, x_batch[:, -1, 1:6], alphas)
    L = a*L1 + b*L2 - c*L3
    
#     print(L1.size(), L2.size(), L3.size())
    
    return L.sum()

