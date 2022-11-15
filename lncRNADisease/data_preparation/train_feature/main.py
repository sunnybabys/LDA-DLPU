import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import dis_model, r_model
from datahelp import DisDataset, RDataset
from data import  dis_sim, r_sim

m = r_sim.shape[0]
d = dis_sim.shape[0]

class Config(object):
    def __init__(self):
        self.embed_size = 16
        self.batch_size = 100
        self.epochs = 150
        self.log_interval = 1000
        self.alpha = 0.4
        self.beta = 0.3

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if  torch.cuda.is_available() else {}

opt = Config()

def train_reprenstation(epoch, model, optimizer, train_loader):
    model.train()
    
    train_loss = 0
    for batch_idx, (data,y) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        predict = model(data)
        loss = nn.MSELoss(reduction="sum")(predict.float(), y.float().to(device)).cpu()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx %  opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss


if __name__ == '__main__':

    #training representation of diseases
    model_dis = dis_model(opt.embed_size).to(device)
    optimizer_d = optim.Adam(model_dis.parameters(), lr=1e-4)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d,'min',factor=0.1, patience=4, verbose=True)
    train_loader_dis = DataLoader(dataset=DisDataset(opt.alpha),batch_size= opt.batch_size, shuffle=True, **kwargs)

    for epoch in range(1,  opt.epochs+1):
        train_loss = train_reprenstation(epoch,model_dis,optimizer_d,train_loader_dis)
        scheduler_d.step(train_loss)

    #training representation of RNAs
    model_r  = r_model(opt.embed_size).to(device)
    optimizer_m = optim.Adam(model_r.parameters(), lr=1e-4)
    scheduler_m = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_m, 'min',factor=0.1, patience=4, verbose=True)
    train_loader_r = DataLoader(dataset=RDataset(opt.beta), batch_size= opt.batch_size, shuffle=True, **kwargs)

    for epoch in range(1,  opt.epochs+1):
        train_loss = train_reprenstation(epoch, model_r, optimizer_m, train_loader_r)
        scheduler_m.step(train_loss)
    print("END")
    disease = model_dis.embed_dis(torch.Tensor(list(range(d))).long().cuda()).detach().cpu().numpy()
    rna = model_r.embed_r(torch.Tensor(list(range(m))).long().cuda()).detach().cpu().numpy()
    np.savetxt("../../data/disease16.txt",disease)
    np.savetxt("../../data/rna16.txt",rna)