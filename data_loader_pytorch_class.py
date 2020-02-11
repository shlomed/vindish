
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import pandas as pd
import numpy as np


# In[ ]:


class VindishDataset(Dataset):
    def __init__(self, l_sample=10, df_data_file='E:\\Datasets\\vindish\\df_all.pkl'):
        self.df = pd.read_pickle(df_data_file)
        self.l_sample = l_sample
        np.random.seed(42)
        self.cols_to_extract = ['Time_To_Expiration', 
                                'UX1', 'UX2', 'UX3', 'UX4', 'UX5', 'SP500', 
                                'time_of_day', 'dow', 'dom', 'doy', 
                                'UX1_diff', 'UX2_diff', 'UX3_diff', 'UX4_diff', 'UX5_diff', 'SP500_diff']
        self.counter = 0
   
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, seed):
        flag_invalid = True
        while(flag_invalid):
            index = np.random.randint(0, self.df.shape[0])
            sample = self.df.iloc[index:index+self.l_sample+2]
            flag_invalid = not self._check_validity(sample)
    
        return self._reform_sample(sample)
        
    def _reform_sample(self, sample):
        sample = sample[self.cols_to_extract]
        x1 = Tensor(sample.iloc[:self.l_sample].values)
        x2 = Tensor(sample.iloc[1:self.l_sample+1].values)
        y = Tensor(sample.iloc[-1].values)
        
        return x1, x2, y
        
    def _check_validity(self, sample):
        rule1 = sample.iloc[0]["Time_To_Expiration"] > sample.iloc[-1]["Time_To_Expiration"]
        
        return rule1


# from data_loader_pytorch_class import VindishDataset
# dataset = VindishDataset()
# dataloader = DataLoader(dataset, batch_size=128)

# %%time
# a = next(iter(dataloader))
