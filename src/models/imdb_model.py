from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import constants


# 定义lstm模型用于文本分类
class RNNModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(RNNModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=True)   # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size*2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)
    
    def forward(self, x):
        """
        input : [bs, maxlen]
        output: [bs, 2] 
        """
        x = self.Embedding(x)  # [bs, ml, emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))   # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)    # [bs, 2]
        return out  # [bs, 2]
    
def LSTMNet():
    model = RNNModel(constants.imdb_max_words, constants.imdb_embedding_size, constants.imdb_hidden_size, constants.imdb_dropout)
    return model