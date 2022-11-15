import networkx as nx
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl
from scipy import interp
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt


def load_data(directory):
    L_FSM = np.loadtxt('../data/rna16.txt')
    D_SSM = np.loadtxt('../data/disease16.txt')
    # L_FSM = np.load("../data/r16_feature.npy")
    # D_SSM = np.load("../data/d16_feature.npy")
    # L_FSM = pd.read_csv(directory + '/rna16_feature.csv',sep=',', header=0, index_col=0).values
    # D_SSM = pd.read_csv(directory + '/disease16_feature.csv',sep=',', header=0,index_col=0).values
    ID = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))
    IL = np.zeros(shape=(L_FSM.shape[0], L_FSM.shape[1]))
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
                ID[i][j] = D_SSM[i][j]

    for i in range(L_FSM.shape[0]):
        for j in range(L_FSM.shape[1]):
                IL[i][j] = L_FSM[i][j]

    return ID, IL


def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + '/lncrna_disease_pair.txt',sep=' ', names=['lncRNA', 'disease', 'label'])

    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_positive = known_associations.sample(n=int(known_associations.shape[0]*0.15), random_state=random_seed, axis=0)  #随机挑选正样本S，待加入未知样本U中
    random_positive['label'] = 0
    p_sample_df = known_associations.drop(random_positive.index.to_list(),axis = 0)  #去除挑选的正样本，P-S作为正样本
    n_sample_df = unknown_associations.append(random_positive)   #负样本S+U
    all_samples =p_sample_df.append(n_sample_df)
    return all_samples.values,p_sample_df,unknown_associations,random_positive

def build_graph(directory, random_seed, ctx):
    ID, IL = load_data(directory)   #ID，疾病的相似性矩阵；IL，rna的相似性矩阵
    all_samples,p_sample_df,unknown_associations,random_positive = sample(directory, random_seed)

    print('Building graph ...')
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(ID.shape[0] + IL.shape[0])   #统计图中节点的个数并嵌入图中，疾病+rna
    node_type = nd.zeros(g.number_of_nodes(), dtype='float32', ctx=ctx)  #初始化节点类型设为0
    node_type[:IL.shape[0]] = 1    #将节点分为两类，一类为疾病，一类为rna，rna用1表示,疾病用0表示
    g.ndata['type'] = node_type    #将分好类的节点嵌入图中

    print('Adding lncRNA features ...')
    l_data = nd.zeros(shape=(g.number_of_nodes(), IL.shape[1]), dtype='float32', ctx=ctx)  #构建特征矩阵，行为所有节点数量，列为rna的特征数量
    l_data[: IL.shape[0], :] = nd.from_numpy(IL)    #前IL.shape[0]行数据为rna的特征，后面都有的行为0，表示没有rna特征，因为他们属于疾病节点
    g.ndata['l_features'] = l_data

    print('Adding disease features ...')
    d_data = nd.zeros(shape=(g.number_of_nodes(), ID.shape[1]), dtype='float32', ctx=ctx)
    d_data[IL.shape[0]: ID.shape[0]+IL.shape[0], :] = nd.from_numpy(ID)
    g.ndata['d_features'] = d_data

    print('Adding edges ...')
    lncrna_ids = list(range(0, IL.shape[0]))
    disease_ids = list(range(0, ID.shape[0]))

    lncrna_ids_invmap = {id_: i for i, id_ in enumerate(lncrna_ids)}
    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}

    sample_lncrna_vertices = [lncrna_ids_invmap[id_]  for id_ in all_samples[:, 0]]  # 找rna对应的编号
    sample_disease_vertices = [disease_ids_invmap[id_] + IL.shape[0] for id_ in all_samples[:, 1]]  # 找疾病样本编号

    g.add_edges(sample_lncrna_vertices, sample_disease_vertices,  # 添加边，边的起始节点列表、边的终止节点列表
                data={'inv': nd.zeros(all_samples.shape[0], dtype='int32', ctx=ctx),  # 添加边的特征(见源码)
                      'rating': nd.from_numpy(all_samples[:, 2].astype('float32')).copyto(ctx)})  # 添加边的标签
    g.add_edges(sample_disease_vertices, sample_lncrna_vertices,  # 双向边，无向图
                data={'inv': nd.zeros(all_samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(all_samples[:, 2].astype('float32')).copyto(ctx)})

    g.readonly()
    print('Successfully build graph !!')
    print(g.all_edges())
    return g,all_samples,p_sample_df,unknown_associations,random_positive,IL.shape[0]
