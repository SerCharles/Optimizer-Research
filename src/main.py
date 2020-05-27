import pdb
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from model.model_cnn import ResNet18

import constants
import data.cifar10_loader, data.cifar100_loader
import algorithm.lookahead



def init_args():
    '''
    描述：加载命令行选项
    参数：无
    返回：args，全局参数
    '''
    dataset_options = ['cifar10', 'cifar100']

    #优化算法组合：RAdam + lookahead
    algorithm_options = ['SGD', 'Adam', 'RAdam']

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
    parser.add_argument('--algorithm', '-a', default='Adam',
                    choices=algorithm_options)
    parser.add_argument('--lookahead', type=bool, default=True, help='Whether use lookahead or not')
    parser.add_argument('--lookahead_steps', '-s', type=int, default=5, help='The step k of lookahead')
    parser.add_argument('--lookahead_lr', '-l', type=float, default=0.8, help='The inner learning rate of lookahead')

    #TODO 更多超参数
    
    args = parser.parse_args()
    return args

def init_components_cnn(args):
    '''
    描述：加载训练的各种东西--device，model，data，criterion，scheduler，optimizer
    参数：args参数
    返回：device，model，train_loader,test_loader,criterion,scheduler,optimizer
    '''
    #device
    torch.cuda.set_device(constants.gpu_id)
    device = torch.device(constants.gpu_id)
    cudnn.benchmark = True  # Should make training should go faster for large models

    #写死seed
    torch.manual_seed(constants.seed)

    #data
    if(args.dataset == 'cifar10'):
        train_loader, test_loader = data.cifar10_loader.load_data(data_dir = constants.data_dir, batch_size = constants.cnn_batch_size)
        num_classes = 10
    elif(args.dataset == 'cifar100'):
        train_loader, test_loader = data.cifar100_loader.load_data(data_dir = constants.data_dir, batch_size = constants.cnn_batch_size)
        num_classes = 100
    #model
    model = ResNet18(num_classes = num_classes)

    model.to(device)
    
    #criterion
    criterion = nn.CrossEntropyLoss()
    
    #optimizer
    if(args.algorithm == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=constants.cnn_learning_rate,
                                weight_decay=5e-4)
    elif(args.algorithm == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=constants.cnn_learning_rate,
                                weight_decay=5e-4)
    
    if(args.lookahead == True):
        optimizer = algorithm.lookahead.Lookahead(optimizer, args.lookahead_steps, args.lookahead_lr)
        
    #scheduler
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    return device, model, train_loader, test_loader, criterion, optimizer, scheduler



def train_cnn(device, model, train_loader, optimizer, criterion):
    '''
    描述：训练cnn模型一个epoch
    参数：device, model, train_loader, optimizer, criterion
    返回：loss，accuracy
    '''
    model.train(True)
    total_loss = 0.0
    total_correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)

    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()

def test_cnn(device, model, test_loader, criterion):
    '''
    描述：验证cnn模型一个epoch
    参数：device, model, test_loader, optimizer, criterion
    返回：loss，accuracy
    '''
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
            
    epoch_loss = total_loss / len(test_loader.dataset)
    epoch_acc = total_correct.double() / len(test_loader.dataset)
    return epoch_loss, epoch_acc.item()





def main_cnn(device, model, train_loader, test_loader, criterion, optimizer, scheduler):
    '''
    描述：cnn训练-测试主函数
    参数：device, model, train_loader, test_loader, criterion, optimizer, scheduler
    返回：train_loss, train_accuracy, test_loss, test_accuracy的list； best_accuracy
    '''
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    best_accuracy = 0.0
    for epoch in range(constants.cnn_epochs):
        print('epoch:{:d}/{:d}'.format(epoch + 1, constants.cnn_epochs))
        print('*' * 100)

        
        train_start = time.time()
        train_loss, train_accuracy = train_cnn(device, model, train_loader, optimizer, criterion)
        train_end = time.time()
        
        print("training: {:.4f}, {:.4f}".format(train_loss, train_accuracy))
        
        test_start = time.time()
        test_loss, test_accuracy = test_cnn(device, model, test_loader, criterion)
        test_end = time.time()
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        print("validation: {:.4f}, {:.4f}".format(test_loss, test_accuracy))
        print("train time: {:.4f} s".format(train_end - train_start))
        print("valid time: {:.4f} s".format(test_end - test_start))

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        
        scheduler.step(epoch)
    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, best_accuracy


if __name__ == '__main__':
    args = init_args()
    if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        device, model, train_loader, test_loader, criterion, optimizer, scheduler = init_components_cnn(args)
        train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, best_accuracy = \
            main_cnn(device, model, train_loader, test_loader, criterion, optimizer, scheduler)