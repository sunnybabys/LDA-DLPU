# encoding=utf-8
import random
import pickle
import numpy as np
import pandas as pd
from hyperparams import hyperparams as params


class Neg_DataLoader:
    def __init__(self,filename):

        self.interaction = pd.read_csv(filename + '/select_sample_lncRNA_disease_association_matrix.csv', index_col=0)
        self.rna_feature = np.loadtxt('../data/rna16.txt')
        self.disease_feature = np.loadtxt('../data/disease16.txt')
        dataset = []
        for i in range(self.interaction.shape[0]):
            for j in range(self.interaction.shape[1]):
                dataset.append(np.hstack((self.rna_feature[i], self.disease_feature[j],self.interaction.iloc[i, j])))
        self.dataset = pd.DataFrame(dataset).values
        self.pre_dataset = pd.DataFrame(dataset)


class Non_Neg_DataLoader:
    def __init__(self,filename):
        self.interaction = pd.read_excel(filename + '/lncRNADisease-lncRNA-disease associations matrix.xls', index_col=0)
        self.rna_feature = pd.read_csv(filename + '/rna_feature.csv', index_col=0).values
        self.disease_feature = pd.read_csv(filename + '/disease_feature.csv', index_col=0).values

        dataset = []
        for i in range(self.interaction.shape[0]):
            for j in range(self.interaction.shape[1]):
                    dataset.append(np.hstack((self.rna_feature[i],self.disease_feature[j],self.interaction.iloc[i, j])))
        self.dataset= pd.DataFrame(dataset).values
        self.pre_dataset = pd.DataFrame(dataset)
