# encoding=utf-8

import numpy as np
import pickle
import pandas as pd

interMatrix = pd.read_excel("../../data/MNDR-lncRNA-disease_association_matrix.xls",header=0,index_col=0).values

rows, cols = interMatrix.shape
print('matrix shape:', interMatrix.shape)
pos_set = []
neg_set = []
for i in range(rows):
    for j in range(cols):
        if interMatrix[i][j] != 0:
            pos_set.append((i, j, 1))
        else:
            neg_set.append((i, j, 0))

print('positive samples:', len(pos_set))
print('negative samples:', len(neg_set))

with open('data.pkl', 'wb') as file:
    pickle.dump((pos_set, neg_set), file)

np.save('../../data/matrix.npy', interMatrix)

