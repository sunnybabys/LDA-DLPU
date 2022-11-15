import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data import  dis_sim, r_sim, dis_sim_gaussian, r_sim_gaussian

class DisDataset(Dataset):
    def __init__(self, alpha=0.5):
        sim = dis_sim * alpha + dis_sim_gaussian * (1-alpha)      
        x=[]
        y=[]
        for i in range(sim.shape[0]):
            total = 0
            count = 0
            for j in range(sim.shape[1]):
                tmp = sim[i][j]
                if i !=j and tmp != 0:
                    x.append([i,j])
                    y.append(tmp)
                    total = total + tmp
                    count = count + 1
        self.x_data = torch.tensor(x)
        self.y_data = torch.tensor(y)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len

class RDataset(Dataset):
    def __init__(self, beta=0.5):
        sim = r_sim * beta + r_sim_gaussian * (1-beta)
        x=[]
        y=[]
        for i in range(sim.shape[0]):
            total = 0 #normalization
            count = 0
            for j in range(sim.shape[1]):
                tmp = sim[i][j]
                if i != j and tmp != 0:
                    x.append([i,j])
                    y.append(tmp)
                    total = total + tmp
                    count = count + 1
        self.x_data = torch.tensor(x)
        self.y_data = torch.tensor(y)
        self.len = self.y_data.shape[0]


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len

