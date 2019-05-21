
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

PRINT = False


# In[ ]:


import keras
import torch
import torchvision


# In[ ]:


c_dir = "C://Users/shlomi/Documents/Work/vindish/data/"
e_dir = "E:\\Work/Vindish/created_samples/"


# In[ ]:


X = torch.tensor(np.load(e_dir + "X.npy")).type(torch.float32)
y = torch.tensor(np.load(e_dir + "y.npy")).type(torch.float32)
features = np.load(e_dir+"features.npy")


# In[ ]:


print(X.shape, y.shape)
print(X.type(), y.type())


# In[ ]:


for i, j in enumerate(features):
    print(i, j)


# ### Define Model

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

mini_conv2d = MiniConv2d()
mini_conv2d(X[:3, :5].unsqueeze_(1))
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

mini_conv1d = MiniConv1d()
mini_conv1d(X[:3, :5].unsqueeze_(1))mini_conv_singles = MiniConv1d(init_kernel_size=(3,3))
singles = X[:, :,  [0, 18, 19]].unsqueeze_(1)
mini_conv_singles(singles)
# In[ ]:


class Embeddings(torch.nn.Module):
    def __init__(self, n_categories, n_dims):
        super(Embeddings, self).__init__()
        self.embed = torch.nn.Embedding(n_categories, n_dims)
        
    def forward(self, x):
        x = self.embed(x)
        
        return x

embed = torch.nn.Embedding(6, 2)
embed(torch.tensor([[1,1,3,4], [1,1,3,4]])).shapeembed = Embeddings(7, 2)
embed(X[:, :, 7].type(torch.long)).shape
# In[ ]:


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mini_conv_ux = MiniConv2d()
        self.mini_conv_ux_diffs = MiniConv2d()
        self.mini_conv_snp = MiniConv1d()
        self.mini_conv_singles = MiniConv1d(init_kernel_size=(3,3))
        self.embed_dow = Embeddings(7, 3)
        self.embed_dom = Embeddings(31, 5)
        
        self.fc1 = torch.nn.Linear(880, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 5)
        
        
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


# ### split train/test

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# In[ ]:


BATCH_SIZE = 64
betas = torch.from_numpy(np.array([0.62, 0.44, 0.32, 0.26, 0.21])).type(torch.Tensor).to(device)


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


# ### Run Model

# In[ ]:


model = Model()
# print(model)
# model.fc3.weight.data = (torch.zeros_like(model.fc3.weight, requires_grad=True, device=device))
# model.fc3.bias.data = torch.tensor([200., 0., 0., 0., 0.], requires_grad=True, device=device)

model = model.to(device)
# print(model)

model.forward(X_train[:BATCH_SIZE]).shape, X_train[:BATCH_SIZE].shapefor i in model.parameters():
    print(i)
# In[ ]:


def get_profit(y, x, alphas):
    if PRINT:
        print("Xs:\n", x[0].cpu())
        print("ys:\n", y[0].cpu())

    L3 = ((y - x)*alphas).sum(dim=1)

    if PRINT:
        print("L3s:\n", L3[:2].cpu())
    return L3


# In[ ]:


def get_dist_from_200(alphas):
    return (alphas.abs().sum(dim=1)-200.)


# In[ ]:


def get_hedging_score(alphas, betas):
    return (alphas*betas).sum(dim=1)


# In[ ]:


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

x.dtype, y.dtype, betas.dtype, alphas.dtype
# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[ ]:


losses_epoch = []
losses_val_history = []
mean_buy_sell_expenses = []
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.33, verbose=True)

for epoch in range(1000):
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
#         print("Xs:\n", x[:2, -1, :].cpu())
#         print("ys:\n", y[:2, :].cpu())
#         print(x)
        alphas = model(x)#.type(torch.Tensor)#.to(device)        
        loss = calc_loss(alphas, betas, x, y)
        losses_epoch.append(loss.cpu().detach().numpy())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%100==99:
            print(np.mean(losses_epoch[-100:]))
            
    model.eval()
    losses_val = []
    L1s = []
    L2s = []
    profits = []
    for x_val, y_val in val_dl:
        x_val, y_val = x_val.to(device), y_val.to(device)
        alphas_val = model(x_val).type(torch.Tensor).to(device)        
        loss_val = calc_loss(alphas_val, betas, x_val, y_val)
        losses_val.append(loss_val.cpu().detach().numpy())
        L1s.append(get_dist_from_200(alphas_val).cpu().detach().numpy().mean())
        L2s.append(get_hedging_score(alphas_val, betas).cpu().detach().numpy().mean())
        profits.append(get_profit(y_val, x_val[:, -1, 1:6], alphas_val).cpu().detach().numpy().mean())
        mean_buy_sell_expenses.append(pd.DataFrame(alphas_val.detach().cpu().numpy()).diff().abs().sum(axis=1).mean()*10)
    
    mean_loss_val = np.mean(losses_val)
    losses_val_history.append(mean_loss_val)
    mean_buy_sell_expenses = np.mean(mean_buy_sell_expenses)
    
    print(mean_loss_val)
#     print("losses_epoch:", losses_epoch)
#     print("losses_val_history:", losses_val_history)
    
    mean_L1 = np.mean(L1s)
    mean_L2 = np.mean(L2s)
    mean_profit = np.mean(profits)
    mean_profit_with_expenses = mean_profit-0.001*mean_buy_sell_expenses

    mean_epoch_loss = np.mean(losses_epoch)
    print(f"epoch = {epoch}; mean_epoch_loss = {mean_epoch_loss:.3f}; val_loss = {mean_loss_val:.3f}; dist_from_200_loss = {mean_L1:.3f}; betas_loss = {mean_L2:.3f}; mean_buy_sell_expenses = {mean_buy_sell_expenses:.3f}; mean_profit = {mean_profit:.6f}; mean_profit_with_expenses = {mean_profit_with_expenses:.6f};")
    scheduler.step(mean_loss_val)

    if mean_loss_val<=min(losses_val_history):
        print("saving model.")
        save_path = f"./model_vindish_epoch_{epoch}_train_loss_{int(mean_epoch_loss)}_val_loss{int(mean_loss_val)}.pth.tar"
        torch.save(model.state_dict(), save_path)
    else:
        print("not saving model.")
    
    lr = optimizer.param_groups[0]["lr"]

    writer.add_scalar("mean_epoch_loss", mean_epoch_loss, epoch)
    writer.add_scalar("mean_loss_val", mean_loss_val, epoch)
    writer.add_scalar("dist_from_200_loss", mean_L1, epoch)
    writer.add_scalar("betas_loss", mean_L2, epoch)
    writer.add_scalar("mean_profit", mean_profit, epoch)
    writer.add_scalar("mean_profit_with_expenses", mean_profit_with_expenses, epoch)
    writer.add_scalar("mean_buy_sell_expenses", mean_buy_sell_expenses, epoch)
    writer.add_scalar("lr", lr, epoch)
    
    losses_epoch = []; L1s = [];  L2s = []; profits = []; mean_buy_sell_expenses=[]

    model.train()


# In[ ]:


(1+mean_profit)**(6*8*200)


# In[ ]:


xx = X_test[:2]
# xx
xx[:, -1, 1:6]


# In[ ]:


yy = y_test[:2]
yy


# In[ ]:


alphas = model.to("cpu")(xx)
print(alphas)


# In[ ]:


alphas.abs().cpu().detach().numpy().sum(axis=1)


# In[ ]:


alphas@betas.to("cpu")


# In[ ]:


betas


# In[ ]:


get_profit(yy, xx[:, -1, 1:6], alphas)


# Change a, b during epochs so that profit will also be tuned during initial steps of optimization
w_fc3 = list(model.fc3.parameters())w_fc3[0] = torch.zeros_like(w_fc3[0])
w_fc3[1] = torch.zeros_like(w_fc3[1])
w_fc3[1][0] = 200.list(model.fc3.parameters())model.fc3.weight.data = (torch.zeros_like(model.fc3.weight))
model.fc3.bias.data = torch.zeros_like(model.fc3.bias)
model.fc3.bias[0] = 200.x = X[:64, :, :]
y = y[:64, :]calc_loss(model(x), betas, x.to(device), y.to(device))
# In[ ]:


(1+0.08/200)**(6*9*200)


# In[ ]:


(1+0.05/200)**(6*9*200)


# In[ ]:


### testing


model = Model()
# PATH = save_path
PATH = "model_vindish_epoch_18_train_loss_-280_val_loss107.pth.tar"
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()

profits = []
costs = []
trans_cost = 0.01
alphas_test = []

x, y = next(iter(test_dl))
x, y = x.to(device), y.to(device)
alphas_test.append(model(x).type(torch.Tensor).to(device))

i = 0

for x, y in tqdm(test_dl): # batch size is 1 for testing
    i += 1
#     if i == 10:
#         break
    x, y = x.to(device), y.to(device)
    alphas_test.append(model(x).type(torch.Tensor).to(device))
#         loss_val = calc_loss(alphas_val, betas, x_val, y_val)
#         losses_val.append(loss_val.cpu().detach().numpy())
#         L1s.append(get_dist_from_200(alphas_val).cpu().detach().numpy().mean())
#         L2s.append(get_hedging_score(alphas_val, betas).cpu().detach().numpy().mean())
    profits.append(get_profit(y, x[:, -1, 1:6], alphas_test[-1]).cpu().detach().numpy().mean())
    costs.append(np.abs(alphas_test[-1].cpu().detach().numpy()-alphas_test[-2].cpu().detach().numpy()).sum()*trans_cost)

print("Total profit including costs: {:.4f}".format(sum(profits+costs)))
pd.DataFrame({"profits":profits, "costs":costs})


# In[ ]:


alpha_50 = pd.Series(alphas_test)[51]#.apply(lambda x: x.cpu().detach().numpy()[0])[50]
# alpha_50
# len(alphas_test)


# In[ ]:


x = X_test[50:51]
y = y_test[50:51]
x, y = x.to(device), y.to(device)
get_profit(y, x[:, -1, 1:6], alpha_50).cpu().detach().numpy().mean()
# len(profits)


# In[ ]:


profits[50]


# In[ ]:


alphas_test = model(x0)


# In[ ]:


6*24

