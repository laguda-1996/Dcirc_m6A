import numpy as np
import pandas as pd
import math
# import tqdm
# import gpytorch
# from matplotlib import pyplot as plt
from itertools import cycle
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp.autocast_mode as autocast
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from seq_load import *



class CNN51_RNN(nn.Module):
    def __init__(self , HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT, FC_DROPOUT, CELL):
        super(CNN51_RNN ,self).__init__()
        self.basicconv0a = torch.nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 7), stride=(1,2), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )# [B, 32, 1, 25]
        self.basicconv0b = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), stride=(2,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )# [B, 32, 1, 11]
        self.basicconv2a = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )# [B, 64, 1, 9]
        self.rnn = BiLSTM_Attention(128 ,HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT)
        # self.rnn = torch.nn.Sequential()
        # if CELL == 'LSTM':
        #     self.rnn.add_module("lstm", nn.LSTM(input_size=128, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
        #                        bidirectional=True, dropout=RNN_DROPOUT))
        # else:
        #     self.rnn.add_module("gru", nn.GRU(input_size=128, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM,
        #                                             bidirectional=True, dropout=RNN_DROPOUT))

        self.fc1 = nn.Linear(HIDDEN_NUM * 2, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(FC_DROPOUT)

    def forward(self, x):
        # x > [batch_size, sequence_len, word_vec]
        x = x.unsqueeze(3).permute(0, 2, 3, 1)
        x = self.basicconv0a(x)
        x = self.basicconv0b(x)
        x = self.basicconv2a(x) # x > [batch_size, channel(input_size), 1, seq_len]
        x = x.squeeze(2).permute(2, 0, 1)  # x > [sequence_len, batch_size, word_vec]
        out  = self.rnn(x) # out > [sequence_len, batch_size, num_directions*hidden_size]
        # out = torch.mean(out, 0)
        out = self.dropout(self.fc1(out))
        out = F.relu(out)
        out = self.fc2(out)
        return out


class BiLSTM_Attention(nn.Module):
    def __init__(self ,input_size, HIDDEN_NUM, LAYER_NUM, RNN_DROPOUT):
        super(BiLSTM_Attention, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size, hidden_size=HIDDEN_NUM, num_layers=LAYER_NUM, bidirectional=True, dropout=RNN_DROPOUT)
        # self.fc1 = nn.Linear(HIDDEN_NUM * 2, 10)
        # self.fc2 = nn.Linear(10, 2)
        # self.out = nn.Linear(HIDDEN_NUM * 2, num_classes)

    # lstm_output : [batch_size, n_step, HIDDEN_NUM * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state ):
        HIDDEN_NUM = 128
        hidden = final_state.view(-1, HIDDEN_NUM * 2, 3) # hidden : [batch_size, HIDDEN_NUM * num_directions(=2), 3(=n_layer)]
        hidden = torch.mean(hidden, 2).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, HIDDEN_NUM * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, HIDDEN_NUM * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #return context, soft_attn_weights.cpu().data.numpy() # context : [batch_size, HIDDEN_NUM * num_directions(=2)]
        return context # context : [batch_size, HIDDEN_NU

    def forward(self, x):
        # input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        # input = x.permute(1, 0, 2)  input : [len_seq, batch_size, embedding_dim]
        input = x

        # hidden_state = Variable(torch.zeros(1*2, len(X), HIDDEN_NUM)) # [num_layers(=1) * num_directions(=2), batch_size, HIDDEN_NUM]
        # cell_state = Variable(torch.zeros(1*2, len(X), HIDDEN_NUM)) # [num_layers(=1) * num_directions(=2), batch_size, HIDDEN_NUM]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, HIDDEN_NUM]
        # output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, HIDDEN_NUM]
        #attn_output, attention = self.attention_net(output, final_hidden_state)
        attn_output= self.attention_net(output, final_hidden_state)
        #return self.fc2(self.fc1(attn_output)), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]
        # return self.fc2(F.relu(self.fc1(attn_output))) # model : [batch_size, num_classes], att
        return attn_output

