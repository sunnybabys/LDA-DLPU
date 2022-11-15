import numpy as np
import pandas as pd
import math

# 读取数据

association_matrix = pd.read_excel("../data/lncRNADisease-lncRNA-disease associations matrix.xls",header=0,index_col=0)
index_rna = association_matrix.index.to_list()
index_diseae = association_matrix.columns.to_list()
association_matrix = association_matrix.values
m = association_matrix.shape[0]
n = association_matrix.shape[1]
#190列，89行，每列表示一个病,每行表示一个lncrna，数值表示一个rna和每个病是否有联系

# 计算disease之间的相似度

association_matrix = association_matrix.T

# 转置之后有190行，89列，每行一个disease

disease_similarity = np.zeros([n, n])  # 100种病之间的相似度，初始化矩阵

width = 0

for c in range(n):
    width += np.sum(association_matrix[c]**2)**0.5  # 按定义用二阶范数计算width parameter



# 计算association_matrix
count = 0
for i in range(n):
    for j in range(n):
        disease_similarity[i, j] = math.exp((np.sum((association_matrix[i] - association_matrix[j])**2)**0.5
                                        * width/n) * (-1))  # 计算不同行（disease）之间的二阶范数


# 保存结果

result = pd.DataFrame(disease_similarity)
result.index = index_diseae
result.columns = index_diseae
result.to_csv('../data/disease_GaussianSimilarity.csv')
# 注意，这样保存之后会多了一行一列行号序号，需要删除


# 计算lncRNA之间的相似度

association_matrix = association_matrix.T  # 转置方便计算

rna_similarity = np.zeros([m, m])  # 89种lnccRNA之间的相似度，初始化矩阵

# 计算association_matrix
count = 0
for a in range(m):
    for b in range(m):
        rna_similarity[a, b] = math.exp((np.sum((association_matrix[a] - association_matrix[b])**2)**0.5
                                        * width/m) * (-1))


# 保存结果

result = pd.DataFrame(rna_similarity)
result.index = index_rna
result.columns = index_rna
result.to_csv('../data/rna_GaussianSimilarity.csv')





