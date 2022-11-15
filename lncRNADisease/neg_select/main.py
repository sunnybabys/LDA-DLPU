import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
import warnings

from train import Train


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    prob,p_sample_df,unknown_associations,random_positive,rna_size = Train(directory='../data',
          epochs=200,
          aggregator='GraphSAGE',  # 'GraphSAGE'
          embedding_size=64,
          layers=1,
          dropout=0.7,
          slope=0.2,  # LeakyReLU
          lr=0.001,
          wd=1e-3,
          random_seed=1234,
          ctx=mx.gpu(0))
    prob['e2'] = prob['e2'] - rna_size
    a = p_sample_df.shape[0]
    b = unknown_associations.shape[0]
    c = random_positive.shape[0]
    pos_sample = prob.iloc[0:a,:]
    unknown_samples = prob.iloc[a:a+b,:]
    unreal_samples =  prob.iloc[a+b:,:]
    min_e1,min_e2,min_probability = unreal_samples.min()  #找到每一列的最小值
    balance_samples = unknown_samples[unknown_samples['probability']<min_probability]
    print(balance_samples)
    association_matrix = pd.read_excel('../data/lncRNADisease-lncRNA-disease associations matrix.xls', index_col=0)

    index_rna = association_matrix.index.to_list()
    index_diseae = association_matrix.columns.to_list()
    reflection = np.zeros((balance_samples.shape[0],2))
    reflection = pd.DataFrame(reflection,columns=['lncrna','disease'])

    for i in range(balance_samples.shape[0]):
        reflection.iloc[i,0] = index_rna[int(balance_samples.iloc[i,0])]
        reflection.iloc[i,1] = index_diseae[int(balance_samples.iloc[i,1])]



    for i in range(reflection.shape[0]):
        lnc_rna_name = reflection.iloc[i,0]
        disease_name = reflection.iloc[i,1]
        association_matrix.loc[lnc_rna_name, disease_name] = -1

    association_matrix.to_csv('../data/select_sample_lncRNA_disease_association_matrix.csv')

    print("End")
