import numpy as np
import pandas as pd
from data_input import  Neg_DataLoader, Non_Neg_DataLoader
from Net import transNet,transNet2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
from hyperparams import hyperparams as params


data =  Non_Neg_DataLoader("../data")
val_data = data.pre_dataset.iloc[data.dataset[:,-1]!=-1]
label = val_data.iloc[:,-1].values


AUC = 0
AUPR = 0

prob_y = []
pre_y = []

for i in range(label.shape[0]):
    n = list(val_data.index)[i]
    feature_test = val_data.loc[n, :].values.reshape(1,-1)[:,0:-1]
    target_test = val_data.loc[n, :].values.reshape(1,-1)[:,-1]
    feature_train = data.pre_dataset.drop(index=n, axis=0).values[:,0:-1]
    target_train  = data.pre_dataset.drop(index=n, axis=0).values[:,-1]
    print('第{}次/{}'.format(i+1,label.shape[0]))
    model =  transNet2(params.col_num, 256, 1).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = torch.nn.MSELoss().to(params.device)
    for epoch in range(params.epoch_num):
        model.train()
        feature_train = torch.FloatTensor(feature_train)
        target_train = torch.FloatTensor(target_train)
        train_x = feature_train.to(params.device)
        train_y = target_train.to(params.device)
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(loss.item())

    model.eval()
    feature_test = torch.FloatTensor(feature_test)
    target_test = torch.LongTensor(target_test)
    test_x = feature_test.to(params.device)
    test_y = target_test.to(params.device)
    pred = model(test_x)
    pred = pred.cuda().data.cpu().numpy()

    KT_y_prob_1 = np.arange(0, dtype=float)
    for k in pred:
        KT_y_prob_1 = np.append(KT_y_prob_1, k)
    light_y = []
    for k in KT_y_prob_1:  # 0 1
        if k > 0.5:
            light_y.append(1)
        else:
            light_y.append(0)
    prob_y.append(KT_y_prob_1)
    pre_y.append(light_y)


prob_y = np.array(prob_y).reshape(-1)
pre_y = np.array(pre_y).reshape(-1)

fpr, tpr, thresholds = roc_curve(label, prob_y)
prec, rec, thr = precision_recall_curve(label, prob_y)
AUC = auc(fpr, tpr)
AUPR = auc(rec, prec)

f = open("../data/data2_loocv_non_auc.csv", mode="a")
for j in range(len(fpr)):
    f.write(str(fpr[j]))
    f.write(",")
    f.write(str(tpr[j]))
    f.write(",")
    f.write(str(auc(fpr, tpr)))
    f.write("\n")
f.write("END__{}".format(i))
f.write("\n")
f.write("\n")
f.close()

f = open("../data/data2_loocv_non_aupr.csv", mode="a")
for j in range(len(prec)):
    f.write(str(rec[j]))
    f.write(",")
    f.write(str(prec[j]))
    f.write(",")
    f.write(str(auc(rec, prec)))
    f.write("\n")
f.write("END__{}".format(i))
f.write("\n")
f.write("\n")
f.close()


print('--------------------------------------结果---------------------------------------------')
print("AUC:%.4f" % AUC)
print("AUPR:%.4f" % AUPR)



