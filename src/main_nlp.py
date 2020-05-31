'''
训练测试nlp模型语言建模PTB的函数
'''

# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn

import os
import os.path as osp
import time

import constants
import models.nlp_model
import data.corpus_loader
import algorithm.lookahead
import algorithm.RAdam

def init_args():
    '''
    描述：加载命令行选项
    参数：无
    返回：args全局参数， result_dir结果存储位置
    '''
    
    lookahead_options = [1, 0]
    
    #lookahead步数2种：5,10
    lookahead_step_options = [5, 10]
    
    #lookahead lr 2种：0.8, 0.5
    lookahead_lr_options = [0.5, 0.8]
    
    parser = argparse.ArgumentParser(description='RNN')
    parser.add_argument('--lookahead', type=int, default=1, help='Whether use lookahead or not', choices=lookahead_options)
    parser.add_argument('--lookahead_lr', '-l', type=float, default=0.8, help='The inner learning rate of lookahead(0.8 default)', 
                        choices = lookahead_lr_options)
    parser.add_argument('--lookahead_steps' , type=int, default=5, help='The step k of lookahead(5 default)', \
        choices = lookahead_step_options)
    args = parser.parse_args()
    print(args)
    print("Please specify the save name of result (in ../results as txt form)")
    file_name = input()
    result_dir = constants.result_dir + file_name + constants.result_back
    print("The result will be saved at", result_dir)
    return args, result_dir


def init_components(args):
    '''
    描述：加载训练的各种东西--device，model，data_loader,criterion ,optimizer, epochs
    参数：args参数
    返回：device，model，data_loader,criterion ,optimizer, epochs
    '''
    torch.manual_seed(1234)
    device = torch.device(constants.gpu_id)

    # load data
    train_batch_size = constants.nlp_train_batch_size
    test_batch_size = constants.nlp_test_batch_size
    batch_size = {'train': train_batch_size,'valid': test_batch_size}
    data_loader = data.corpus_loader.Corpus(constants.nlp_data_dir, batch_size, 35)
    
    
    voc_size = data_loader.voc_size
    model = models.nlp_model.LMModel(voc_size, 50, 50, 2)
    model = model.to(device)

    current_lr = constants.nlp_learning_rate
    criterion = nn.CrossEntropyLoss()


    optimizer = algorithm.RAdam.RAdam(model.parameters(), lr=current_lr)
    if(args.lookahead == 1):
        optimizer = algorithm.lookahead.Lookahead(optimizer, args.lookahead_steps, args.lookahead_lr)
            
    epochs = constants.nlp_epochs
    
    return device, model, data_loader, criterion, optimizer, epochs

    


def evaluate(device, model, data_loader, criterion):
    '''
    描述：验证cnn模型一个epoch
    参数：device, model, test_loader, optimizer, criterion
    返回：ppl，accuracy
    '''
    model.train(False)
    EndFlag = False
    data_loader.set_valid()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    while EndFlag == False:
        valid, target, EndFlag = data_loader.get_batch()
        if(EndFlag == True):
            continue
        total_num += valid.size(0) * valid.size(1)
        valid = valid.to(device)
        target = target.to(device)
        outputs, _ = model(valid)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, target)
        total_loss += loss.item() * valid.size(0) * valid.size(1)
        total_correct += torch.sum(predictions == target.data)
    epoch_pp = math.exp(total_loss / total_num)
    epoch_acc = total_correct.double() / total_num
    return epoch_pp, epoch_acc.item()


def train(device, model, data_loader, optimizer, criterion):
    '''
    描述：训练rnn模型一个epoch
    参数：device, model, train_loader, optimizer, criterion
    返回：pp，accuracy
    '''
    model.train(True)
    EndFlag = False
    data_loader.set_train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    while EndFlag == False:
        train, target, EndFlag = data_loader.get_batch()
        if(EndFlag == True):
            continue
        total_num += train.size(0) * train.size(1)
        train = train.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs, _ = model(train)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)
        _, predictions = torch.max(outputs, 1)   
        loss = criterion(outputs, target)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item() * train.size(0) * train.size(1)
        total_correct += torch.sum(predictions == target.data)
    epoch_pp = math.exp(total_loss / total_num)
    epoch_acc = total_correct.double() / total_num
    return epoch_pp, epoch_acc.item()

args, log_filename = init_args()
device, model, data_loader, criterion, optimizer, epochs = init_components(args)


acc_list = []
for epoch in range(1, epochs+1):
    print('epoch:{:d}/{:d}'.format(epoch, epochs))
    print('*' * 100)

    train_start = time.time()
    train_pp, train_acc = train(device, model, data_loader, optimizer, criterion)
    train_end = time.time()
        
    print("training: {:.4f}, {:.4f}".format(train_pp, train_acc))
        
    valid_start = time.time()
    valid_pp, valid_acc = evaluate(device, model, data_loader, criterion)
    valid_end = time.time()

    print("validation: {:.4f}, {:.4f}".format(valid_pp, valid_acc))
    print("train time: {:.4f} s".format(train_end - train_start))
    print("valid time: {:.4f} s".format(valid_end - valid_start))

    acc_list.append({"train": train_acc, "valid": valid_acc, "train_pp": train_pp, "valid_pp":valid_pp})
    
#输出信息
with open(log_filename,"w") as f:
    for item in acc_list:
        f.write("training: {:.4f}\n".format(item["train"]))
        f.write("validation: {:.4f}\n".format(item["valid"]))
        f.write("training_pp: {:.4f}\n".format(item["train_pp"]))
        f.write("validation_pp: {:.4f}\n".format(item["valid_pp"]))
