import numpy as np
import pandas as pd
import math
import tqdm
#import gpytorch
# from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from seq_load import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metric(true, pred):
    confusion = confusion_matrix(true, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)

wordvec_len = 8
HIDDEN_NUM = 128
LAYER_NUM = 3
FC_DROPOUT = 0.5
RNN_DROPOUT = 0.5
CELL = 'LSTM'


def predict(model, x):
    model.eval() #evaluation mode do not use drop out
    fx = model(x)
    return fx

pos_train_fa = 'hela1_pos_51bp_1W.fasta'
neg_train_fa = 'neg_51bp_1W.fasta'

model_path = '.'
if model_path[-1] == '/':
    model_path = model_path[:-1]
checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar', map_location=torch.device('cpu'))



model = CNN51_RNN(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, RNN_DROPOUT, CELL)
model.load_state_dict(checkpoint['state_dict'])

data_pos_train = load_data_bicoding(pos_train_fa)
data_neg_train = load_data_bicoding(neg_train_fa)

data_train = np.array([_ + [1] for _ in data_pos_train] + [_ + [0] for _ in data_neg_train])
np.random.seed(42)
np.random.shuffle(data_train)

X_test = np.array([_[:-1] for _ in data_train])
y_test = np.array([_[-1] for _ in data_train])
print(y_test.shape)
X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / wordvec_len), wordvec_len)
X_test = torch.from_numpy(X_test).float()


batch_size = 256
i = 0
N = X_test.shape[0]
y_pred_test = []
y_pred_prob_test = []

while i + batch_size < N:
    x_batch = X_test[i:i + batch_size]

    fx = predict(model, x_batch)
    y_pred = fx.cpu().data.numpy().argmax(axis=1)
    prob_data = F.log_softmax(fx, dim=1).cpu().data.numpy()
    for m in range(len(prob_data)):
        y_pred_prob_test.append(np.exp(prob_data)[m][1])

    y_pred_test += list(y_pred)

    i += batch_size

x_batch = X_test[i:N]
fx = predict(model, x_batch)
y_pred = fx.cpu().data.numpy().argmax(axis=1)
prob_data = F.log_softmax(fx, dim=1).cpu().data.numpy()
for m in range(len(prob_data)):
    y_pred_prob_test.append(np.exp(prob_data)[m][1])

y_pred_test += list(y_pred)


# test metrics
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)

test_specificity = calculate_metric(y_test, y_pred_test)
# test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)

print(" acc = %.2f%%, AUROC_test = %0.3f, test_recall(sn) = %0.3f ,test_sp = %0.3f, test_precision = %0.3f, test_f1_score = %0.3f, test_mcc = %0.3f"% (100. * np.mean(y_pred_test == y_test), auc(fpr_test, tpr_test), test_recall_score, test_specificity, test_precision_score, test_f1_score,
       test_mcc))







