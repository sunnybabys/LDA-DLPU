import pickle
import xlrd
import numpy as np
import torch
import pandas as pd
import pickle
import os

dir_path = '../../data/'


#1. load disease semantic similarity.
dis_sim = pd.read_excel(dir_path + 'lncRNADisease-disease semantic similarity matrix.xls',index_col=0).values
dis_sim = torch.from_numpy(dis_sim)


#2. load RNA functional similarity
r_sim = pd.read_excel(dir_path + 'lncRNADisease-lncRNA functional similarity matrix.xls',index_col=0).values
r_sim = torch.from_numpy(r_sim)

#3. load gaussian disease simlarity
dis_sim_gaussian = pd.read_csv(dir_path + 'disease_GaussianSimilarity.csv', index_col=0).values
dis_sim_gaussian = torch.from_numpy(dis_sim_gaussian)


#4. load gaussian RNA simlarity
r_sim_gaussian = pd.read_csv(dir_path + 'rna_GaussianSimilarity.csv', index_col=0).values
r_sim_gaussian = torch.from_numpy(r_sim_gaussian)

