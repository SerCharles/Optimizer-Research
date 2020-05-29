'''
绘制一个模型的训练，测试模型
'''

import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import numpy as np
import constants

class PlotCurves:
    def GetAccuracyAndLoss(self, file):
        '''
        描述：读取记录文件，获取训练测试准确率
        '''
        train_loss_list = []
        train_accuracy_list = []
        test_loss_list = []
        test_accuracy_list = []
        
        with open(file) as f:
            lines = f.readlines()
            for item in lines:
                item_split = item.split()
                if(len(item_split) <= 0):
                    continue
                if item_split[0] == "train_loss:":
                    train_loss_list.append(float(item_split[-1]))
                elif item_split[0] == "train_accuracy:":
                    train_accuracy_list.append(float(item_split[-1]))
                elif item_split[0] == "test_loss:":
                    test_loss_list.append(float(item_split[-1]))
                elif item_split[0] == "test_accuracy:":
                    test_accuracy_list.append(float(item_split[-1]))
        return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list


        
    def PlotContrast(self, ScalarList, length):
        writer = SummaryWriter()
        for epoch in range(length):
            TheScalar = {}
            for name in ScalarList.keys():
                TheScalar[name] = ScalarList[name][epoch]
            
            writer.add_scalars('scalar/test', TheScalar, epoch)
        writer.close()
        
    def __init__(self, model_list):
        super(PlotCurves, self).__init__()
        ScalarList = {}
        length = -1
        for item in model_list:
            model_name = item["model_name"]
            model_dir = item["model_dir"]
            log_place = constants.result_dir + model_dir + ".txt"
            train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = self.GetAccuracyAndLoss(log_place)
            if length < 0:
                length = len(train_loss_list)
            ScalarList[model_name] = train_loss_list
        #self.PlotModel(y_train, y_valid, model_name, save_place)
        self.PlotContrast(ScalarList, length)
        
if __name__ == '__main__':
    PlotBase = []
    Baseline = {"model_name":"The baseline with RAdam on cifar-10", "model_dir":"radam_baseline"}
    Ranger = {"model_name":"Ranger: RAdam + lookahead on cifar-10", "model_dir":"ranger"}
    PlotBase.append(Baseline)
    PlotBase.append(Ranger)


    
    
    print("please input the type you want")
    print("0:Basic")

    num = int(input())
    if num == 0:
        PlotCurves(PlotBase)