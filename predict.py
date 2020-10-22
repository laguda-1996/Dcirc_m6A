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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-predict_fa", "--predict_fasta", action="store", dest='predict_fa', required=True,
                        help="predict fasta file")
    parser.add_argument("-model_path", "--model_path", action="store", dest='model_path', required=True,
                        help="model_path")
    parser.add_argument("-outfile", "--outfile", action="store", dest='outfile', required=True,
                        help="outfile name")

    args = parser.parse_args()


    predict_file = args.predict_fa
    model_path = args.model_path
    if model_path[-1] == '/':
        model_path = model_path[:-1]
    checkpoint = torch.load(model_path + '/' + 'checkpoint.pth.tar', map_location=torch.device('cpu'))

    model = CNN51_RNN(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, RNN_DROPOUT, CELL)
    model.load_state_dict(checkpoint['state_dict'])

    X_test, fa_header = header_and_seqload(predict_file)
    X_test=np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / wordvec_len), wordvec_len)
    X_test = torch.from_numpy(X_test).float()


    batch_size = 256
    i = 0
    N = X_test.shape[0]
    y_pred_test = []
    y_pred_prob_test = []

    with open(args.out_fn, 'w') as fw:
        while i + batch_size < N:
            x_batch = X_test[i:i + batch_size]
            header_batch = fa_header[i:i + batch_size]

            fx = predict(model, x_batch)
            # y_pred = fx.cpu().data.numpy().argmax(axis=1)
            prob_data = F.log_softmax(fx, dim=1).cpu().data.numpy()
            for m in range(len(prob_data)):
                # y_pred_prob_test.append(np.exp(prob_data)[m][1])
                fw.write(header_batch[m] + '\t' + str(np.exp(prob_data)[m][1]) + '\n')

            y_pred_test += list(y_pred)

            i += batch_size

        x_batch = X_test[i:N]
        fx = predict(model, x_batch)
        # y_pred = fx.cpu().data.numpy().argmax(axis=1)
        prob_data = F.log_softmax(fx, dim=1).cpu().data.numpy()
        for m in range(len(prob_data)):
            # y_pred_prob_test.append(np.exp(prob_data)[m][1])
            fw.write(header_batch[m] + '\t' + str(np.exp(prob_data)[m][1]) + '\n')









