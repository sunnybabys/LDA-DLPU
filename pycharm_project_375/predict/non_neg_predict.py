import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
from Net import transNet2
from data_input import  Non_Neg_DataLoader
from hyperparams import hyperparams as params


data =  Non_Neg_DataLoader("../data")
dataset_label = data.dataset[:,-1].reshape(-1)

n_acc = []
n_precision = []
n_recall = []
n_f1 = []
n_AUC = []
n_AUPR = []

for i in range(params.number):
    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    AUC = 0
    AUPR = 0
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(data.dataset,dataset_label):
        feature_train = data.dataset[train_index][:,:-1]
        feature_test = data.dataset[test_index][:,:-1]
        target_train = dataset_label[train_index]
        target_test =  dataset_label[test_index]

        print('begin training:')
        model = transNet2(params.col_num, 256, 1).to(params.device)
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
            if epoch % 50 ==0:
               print(loss.item())

        model.eval()
        feature_test = torch.FloatTensor(feature_test)
        target_test = torch.LongTensor(target_test)
        test_x = feature_test.to(params.device)
        test_y = target_test.to(params.device)
        pred = model(test_x)
        pred = pred.cuda().data.cpu().numpy()
        KT_y_prob_1 = np.arange(0, dtype=float)
        for i in pred:
            KT_y_prob_1 = np.append(KT_y_prob_1, i)
        light_y = []
        for i in KT_y_prob_1:  # 0 1
            if i > 0.5:
                light_y.append(1)
            else:
                light_y.append(0)
        acc += accuracy_score(target_test, light_y)
        precision += precision_score(target_test, light_y)
        recall += recall_score(target_test, light_y)
        f1 += f1_score(target_test, light_y)

        fpr, tpr, thresholds = roc_curve(target_test, KT_y_prob_1)
        prec, rec, thr = precision_recall_curve(target_test, KT_y_prob_1)
        AUC += auc(fpr, tpr)
        AUPR += auc(rec, prec)

        f = open("../data/data1_cv3_non_auc.csv", mode="a")
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

        f = open("../data/data1_cv3_non_aupr.csv", mode="a")
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

    acc = acc / 5
    precision = precision / 5
    recall = recall / 5
    f1 = f1 / 5
    AUC = AUC / 5
    AUPR = AUPR / 5

    print('--------------------------------------结果---------------------------------------------')
    print("accuracy:%.4f" % acc)
    print("precision:%.4f" % precision)
    print("recall:%.4f" % recall)
    print("F1 score:%.4f" % f1)
    print("AUC:%.4f" % AUC)
    print("AUPR:%.4f" % AUPR)

    n_acc.append(acc)
    n_precision.append(precision)
    n_recall.append(recall)
    n_f1.append(f1)
    n_AUC.append(AUC)
    n_AUPR.append(AUPR)

mean_acc = np.mean(n_acc)
mean_precision = np.mean(n_precision)
mean_recall = np.mean(n_recall)
mean_f1 = np.mean(n_f1)
mean_AUC = np.mean(n_AUC)
mean_AUPR = np.mean(n_AUPR)

std_acc = np.std(n_acc)
std_precision = np.std(n_precision)
std_recall = np.std(n_recall)
std_f1 = np.std(n_f1)
std_AUC = np.std(n_AUC)
std_AUPR = np.std(n_AUPR)

print('--------------------------------------平均结果---------------------------------------------')
print("accuracy:%.4f" % mean_acc)
print("precision:%.4f" % mean_precision)
print("recall:%.4f" % mean_recall)
print("F1 score:%.4f" % mean_f1)
print("AUC:%.4f" % mean_AUC)
print("AUPR:%.4f" % mean_AUPR)

print('--------------------------------------平均std---------------------------------------------')
print("accuracy:%.4f" % std_acc)
print("precision:%.4f" % std_precision)
print("recall:%.4f" % std_recall)
print("F1 score:%.4f" % std_f1)
print("AUC:%.4f" % std_AUC)
print("AUPR:%.4f" % std_AUPR)
