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
from models import ResNet18, LSTMNet

import constants
import data.cifar10_loader, data.cifar100_loader, data.imdb_loader
import algorithm.lookahead



def init_args():
    '''
    描述：加载命令行选项
    参数：无
    返回：args全局参数， result_dir结果存储位置
    '''
    

    lookahead_options = [1, 0]
    
    dataset_options = ['cifar10', 'cifar100', 'imdb']

    #优化算法组合：RAdam + lookahead
    algorithm_options = ['SGD', 'Adam', 'RAdam']
    
    #学习率两种：0.1,0.3
    lr_options = [0.1, 0.3]
    
    #lookahead步数2种：5,10
    lookahead_step_options = [5, 10]
    
    #lookahead lr 2种：0.8, 0.5
    lookahead_lr_options = [0.5, 0.8]
    
    #衰减两种:快速，慢速
    decay_options = ['slow', 'fast']

    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
    parser.add_argument('--algorithm', '-a', default='Adam',
                    choices=algorithm_options)
    parser.add_argument('--lookahead', type=int, default=1, help='Whether use lookahead or not', choices=lookahead_options)
    parser.add_argument('--lookahead_lr', '-l', type=float, default=0.8, help='The inner learning rate of lookahead(0.8 default)', 
                        choices = lookahead_lr_options)
    parser.add_argument('--lookahead_steps' , type=int, default=5, help='The step k of lookahead(5 default)', \
        choices = lookahead_step_options)
    parser.add_argument('--learning_rate','-lr', type=float, default=0.1, help='The learning rate (0.1 default)', 
                        choices = lr_options)
    parser.add_argument('--lr_decay', default='slow', help='The learning rate decay strategy, slow(default) or fast', 
                        choices = decay_options)
    
    args = parser.parse_args()
    print(args)
    print("Please specify the save name of result (in ../results as txt form)")
    file_name = input()
    result_dir = constants.result_dir + file_name + constants.result_back
    print("The result will be saved at", result_dir)
    
    return args, result_dir

def init_components(args):
    '''
    描述：加载训练的各种东西--device，model，data，criterion，scheduler，optimizer, epochs
    参数：args参数
    返回：device，model，train_loader,test_loader,criterion,scheduler,optimizer, epochs
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
    elif(args.dataset == 'imdb'):
        train_loader, test_loader = data.imdb_loader.load_data()
        num_classes = 2
        
    #model
    if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        model = ResNet18(num_classes = num_classes)
    elif(args.dataset == 'imdb'):
        model = LSTMNet()
    model.to(device)
    
    #criterion
    criterion = nn.CrossEntropyLoss()
    
    #optimizer
    learning_rate = args.learning_rate
    if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        if(args.algorithm == 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=5e-4)
        elif(args.algorithm == 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                weight_decay=5e-4)
    elif(args.dataset == 'imdb'):
        if(args.algorithm == 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 100)
        elif(args.algorithm == 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate / 100)
    if(args.lookahead == 1):
        optimizer = algorithm.lookahead.Lookahead(optimizer, args.lookahead_steps, args.lookahead_lr)
        
    #scheduler
    if(args.lr_decay == 'slow'):
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif(args.lr_decay == 'fast'):
        scheduler = MultiStepLR(optimizer, milestones=[30, 48, 58], gamma=0.2)


    #epochs
    if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        epochs = constants.cnn_epochs
    elif(args.dataset == 'imdb'):
        epochs = constants.imdb_epochs
    return device, model, train_loader, test_loader, criterion, optimizer, scheduler, epochs



def train(device, model, train_loader, optimizer, criterion):
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

def test(device, model, test_loader, criterion):
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



def main(device, model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    '''
    描述：cnn训练-测试主函数
    参数：device, model, train_loader, test_loader, criterion, optimizer, scheduler, epochs
    返回：train_loss, train_accuracy, test_loss, test_accuracy的list； best_accuracy
    '''
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    best_accuracy = 0.0
    for epoch in range(epochs):
        print('epoch:{:d}/{:d}'.format(epoch + 1, epochs))
        print('*' * 100)

        
        train_start = time.time()
        train_loss, train_accuracy = train(device, model, train_loader, optimizer, criterion)
        train_end = time.time()
        
        print("train: {:.4f}, {:.4f}".format(train_loss, train_accuracy))
        
        test_start = time.time()

        test_loss, test_accuracy = test(device, model, test_loader, criterion)
        test_end = time.time()
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        print("test: {:.4f}, {:.4f}".format(test_loss, test_accuracy))
        print("train time: {:.4f} s".format(train_end - train_start))
        print("test time: {:.4f} s".format(test_end - test_start))

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

        if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
            scheduler.step(epoch)
    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, best_accuracy



if __name__ == '__main__': 
    
    args, result_dir = init_args()
    device, model, train_loader, test_loader, criterion, optimizer, scheduler, epochs = init_components(args)
    train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list, best_accuracy = \
        main(device, model, train_loader, test_loader, criterion, optimizer, scheduler, epochs)
    with open(result_dir,"w") as f:
        for i in range(epochs):
            f.write("train_loss: {:.4f}\n".format(train_loss_list[i]))
            f.write("train_accuracy: {:.4f}\n".format(train_accuracy_list[i]))
            f.write("test_loss: {:.4f}\n".format(test_loss_list[i]))
            f.write("test_accuracy: {:.4f}\n".format(test_accuracy_list[i]))