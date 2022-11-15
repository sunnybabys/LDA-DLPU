import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class regression_model(nn.Module):
    def __init__(self, embed, input_size=128*2):
        super(regression_model, self).__init__()
        self.embed = embed
        self.sim = nn.CosineSimilarity()
        self.dis = nn.PairwiseDistance(p=2)
    
    def forward(self, x):
        x = x.view(-1, 2)
        idx_d1, idx_d2 = x[:,0], x[:,1]
        x_embed_d1,x_embed_d2 = self.embed(idx_d1.long()), self.embed(idx_d2.long())
        x = 0.5 + 0.5*self.sim(x_embed_d1, x_embed_d2)
        return x

class dis_model(nn.Module):
    def __init__(self, embed_size=128, dis_size=383):
        super(dis_model,self).__init__()
        self.embed_dis = nn.Embedding(dis_size, embed_size) 
        self.init_weights()
        self.model = regression_model(self.embed_dis, input_size = embed_size*2)

    def init_weights(self):
        initrange = 0.1
        self.embed_dis.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        return self.model.forward(x)
        
class r_model(nn.Module):
    def __init__(self, embed_size=128, r_size=495):
        super(r_model,self).__init__()
        self.embed_r  = nn.Embedding(r_size,embed_size)
        self.init_weights()
        self.model = regression_model(self.embed_r, input_size = embed_size*2)

    def init_weights(self):
        initrange = 0.1
        self.embed_r.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        return self.model.forward(x)


