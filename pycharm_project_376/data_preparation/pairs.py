import numpy as np
import pickle
import pandas as pd


interMatrix = pd.read_excel("../data/MNDR-lncRNA-disease_association_matrix.xls",header=0,index_col=0)
index_rna = interMatrix.index.to_list()
index_diseae = interMatrix.columns.to_list()
interMatrix = interMatrix.values
rows, cols = interMatrix.shape
print('matrix shape:', interMatrix.shape)
rd_pairs = []
for i in range(rows):
    for j in range( cols):
        rd_pairs.append([i,j,interMatrix[i,j]])

rd_pairs = np.array(rd_pairs).reshape(-1,3)
print(rd_pairs)
np.savetxt("../data/lncrna_disease_pair.txt",rd_pairs,fmt='%d')