import numpy as np
import pandas as pd
import math
import argparse
# import tqdm
# import gpytorch
# from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp.autocast_mode as autocast
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from seq_load import *
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is {}'.format(device))

def save_checkpoint(state, is_best, model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'checkpoint.pth.tar')

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn, 'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i]) + '\t' + str(y_pred[i]) + '\n')


def calculate_metric(gt, pred):
    confusion = confusion_matrix(gt, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP)


if __name__ == '__main__':

    torch.manual_seed(1000)

    parser = argparse.ArgumentParser()

    # main option
    parser.add_argument("-pos_fa", "--positive_fasta", action="store", dest='pos_fa', required=True,
                        help="positive fasta file")
    parser.add_argument("-neg_fa", "--negative_fasta", action="store", dest='neg_fa', required=True,
                        help="negative fasta file")

    parser.add_argument("-outdir", "--out_dir", action="store", dest='out_dir', required=True,
                        help="output directory")

    # rnn option
    parser.add_argument("-rnntype", "--rnn_type", action="store", dest='rnn_type', default='LSTM', type=str,
                        help="[capital] LSTM(default), GRU")
    parser.add_argument("-hidnum", "--hidden_num", action="store", dest='hidden_num', default=128, type=int,
                        help="rnn size")
    parser.add_argument("-rnndrop", "--rnn_drop", action="store", dest='rnn_drop', default=0.5, type=float,
                        help="rnn size")

    # fc option
    parser.add_argument("-fcdrop", "--fc_drop", action="store", dest='fc_drop', default=0.5, type=float,
                        help="Optional: 0.5(default), 0~0.5(recommend)")

    # optimization option
    parser.add_argument("-optim", "--optimization", action="store", dest='optim', default='Adam', type=str,
                        help="Optional: Adam(default), RMSprop")
    parser.add_argument("-epochs", "--max_epochs", action="store", dest='max_epochs', default=10, type=int,
                        help="max epochs")
    parser.add_argument("-lr", "--learning_rate", action="store", dest='learning_rate', default=0.0001, type=float,
                        help="Adam: 0.0001(default), 0.0001~0.01(recommend)")
    # parser.add_argument("-lrstep", "--lr_decay_step", action="store", dest='lr_decay_step', default=10, type=int,
    #                     help="learning rate decay step")
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=8, type=int,
                        help="batch size")

    args = parser.parse_args()

    model_path = '.'

    wordvec_len = 8
    HIDDEN_NUM = args.hidden_num

    LAYER_NUM = 3

    RNN_DROPOUT = args.rnn_drop
    FC_DROPOUT = args.fc_drop
    CELL = args.rnn_type
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size

    tprs = []
    ROC_aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    PR_aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    pos_train_fa = args.pos_fa
    neg_train_fa = args.neg_fa

    X, y = load_train_val_code(pos_train_fa, neg_train_fa)
    folds = StratifiedKFold(n_splits=5).split(X, y)
    for trained, valided in folds:
        X_train, y_train = X[trained], y[trained]
        X_test, y_test = X[valided], y[valided]
        X_train, y_train, X_test, y_test = to_torch_fmt(X_train, y_train, X_test, y_test)
        X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)

        model = CNN51_RNN(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, RNN_DROPOUT, CELL)
        model = model.to(device)
        loss = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if args.optim == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

        best_acc = 0.
        best_train_accuracy_score = 0.
        patience = 0.


        def train(model, loss, optimizer, x, y):
            model.train()

            # Reset gradient
            optimizer.zero_grad()

            # Forward
            fx = model(x)

            loss = loss.forward(fx, y)

            pred_prob = F.log_softmax(fx, dim=1)

            # Backward
            loss.backward()

            # grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            # Update parameters
            optimizer.step()

            return loss.cpu().item(), pred_prob, list(np.array(y.cpu())), list(fx.data.cpu().detach().numpy().argmax(axis=1))  # cost,pred_probability and true y value


        def predict(model, x):
            model.eval()  # evaluation mode do not use drop out
            fx = model(x)
            return fx


        EPOCH = args.max_epochs
        n_classes = 2
        n_examples = len(X_train)

        for i in range(EPOCH):
            start_time = time.time()

            cost = 0.
            y_pred_prob_train = []
            y_batch_train = []
            y_batch_pred_train = []

            num_batches = n_examples // BATCH_SIZE
            for k in range(num_batches):
                start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
                output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end],
                                                                         y_train[start:end])
                cost += output_train

                prob_data = y_pred_prob.cpu().detach().numpy()

                # else:
                for m in range(len(prob_data)):
                    y_pred_prob_train.append(np.exp(prob_data)[m][1])

                y_batch_train += y_batch
                y_batch_pred_train += y_pred_train

            scheduler.step()

            # rest samples
            start, end = num_batches * BATCH_SIZE, n_examples
            output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end],
                                                                     y_train[start:end])
            cost += output_train

            prob_data = y_pred_prob.cpu().detach().numpy()


            for m in range(len(prob_data)):
                y_pred_prob_train.append(np.exp(prob_data)[m][1])


            y_batch_train += y_batch
            y_batch_pred_train += y_pred_train

            # train metrics
            fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)


            train_accuracy_score = sklearn.metrics.accuracy_score(y_batch_train, y_batch_pred_train)
            train_recall_score = sklearn.metrics.recall_score(y_batch_train, y_batch_pred_train)
            train_precision_score = sklearn.metrics.precision_score(y_batch_train, y_batch_pred_train)
            train_f1_score = sklearn.metrics.f1_score(y_batch_train, y_batch_pred_train)
            train_mcc = sklearn.metrics.matthews_corrcoef(y_batch_train, y_batch_pred_train)

            # predict val
            fx_test = predict(model, X_test)
            y_pred_prob_test = []

            y_pred_test = fx_test.cpu().data.numpy().argmax(axis=1)
            prob_data = F.log_softmax(fx_test, dim=1).data.cpu().numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(np.exp(prob_data)[m][1])

            # val metrics
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)


            test_specificity = calculate_metric(y_test, y_pred_test)
            test_accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred_test)
            test_recall_score = sklearn.metrics.recall_score(y_test, y_pred_test)
            test_precision_score = sklearn.metrics.precision_score(y_test, y_pred_test)
            test_f1_score = sklearn.metrics.f1_score(y_test, y_pred_test)
            test_mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred_test)

            end_time = time.time()
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)

            print("Epoch %d, cost = %f, AUC_train = %0.3f, acc = %.2f%%, AUC_test = %0.3f ,train_accuracy = %0.3f, train_recall(sn) = %0.3f, train_precision = %0.3f, train_f1_score = %0.3f, train_mcc = %0.3f, test_accuracy = %0.3f, test_recall(sn) = %0.3f ,test_sp = %0.3f, test_precision = %0.3f, test_f1_score = %0.3f, test_mcc = %0.3f"
                % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), 100. * np.mean(y_pred_test == y_test),
                   auc(fpr_test, tpr_test), train_accuracy_score, train_recall_score, train_precision_score, train_f1_score,
                   train_mcc, test_accuracy_score, test_recall_score, test_specificity, test_precision_score, test_f1_score, test_mcc))

            print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            cur_acc = 100. * np.mean(y_pred_test == y_test)
            is_best = bool(cur_acc > best_acc)
            best_acc = max(cur_acc, best_acc)
            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best, model_path)

            # patience
            if not is_best:
                patience += 1
                if patience >= 5:
                    break

            else:
                patience = 0

            if is_best:
                ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                                    model_path + '/' + 'predout_train.tsv')

                ytest_ypred_to_file(y_test, y_pred_prob_test,
                                    model_path + '/' + 'predout_val.tsv')


        fprArray.append(fpr_test)
        tprArray.append(tpr_test)
        thresholdsArray.append(thresholds_test)
        tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr_test, tpr_test)
        ROC_aucs.append(roc_auc)

        recall_array.append(recall_test)
        precision_array.append(precision_test)
        precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
        pr_auc = auc(recall_test ,precision_test)
        PR_aucs.append(pr_auc)


    colors = cycle(['#caffbf', '#ffc6ff' ,'#ffadad', '#ffd6a5', '#caffbf', '#9bf6ff', '#a0c4ff', '#bdb2ff'])
    ## ROC plot for CV
    fig = plt.figure(0)
    for i, color in zip(range(len(fprArray)), colors):
        plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, ROC_aucs[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#c4c7ff',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ROC_mean_auc = auc(mean_fpr, mean_tpr)
    ROC_std_auc = np.std(ROC_aucs)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('.')
    plt.close(0)

    fig = plt.figure(1)
    for i, color in zip(range(len(recall_array)), colors):
        plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                 label='PRC fold %d (AUPRC = %0.2f)' % (i + 1, PR_aucs[i]))
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    PR_mean_auc = auc(mean_recall, mean_precision)
    PR_std_auc = np.std(PR_aucs)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig('.')
    plt.close(0)

    print('> best acc:', best_acc)

