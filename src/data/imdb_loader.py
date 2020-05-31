'''
描述：读取imdb数据集
'''

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import * 
from keras.preprocessing.sequence import pad_sequences 
from keras.datasets import imdb
import constants


def load_data():
    '''
    描述：读取imdb数据集
    参数：无
    返回：train_loader, test_loader
    '''
    
    # 借助Keras加载imdb数据集
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=constants.imdb_max_words)
    x_train = pad_sequences(x_train, maxlen=constants.imdb_max_len, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=constants.imdb_max_len, padding="post", truncating="post")

    # 转化为TensorDataset
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

    # 转化为 DataLoader
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=constants.imdb_batch_size)

    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=constants.imdb_batch_size)
    
    return train_loader, test_loader