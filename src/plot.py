'''
绘制一个模型的训练，测试曲线
'''

import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import numpy as np
import constants


class PlotCurvesNLP:
    '''
    读取NLP的pp和其他不同
    '''
    def GetAccuracyAndLoss(self, file):
        '''
        描述：读取记录文件，获取训练测试准确率
        '''
        train_pp_list = []
        train_accuracy_list = []
        test_pp_list = []
        test_accuracy_list = []
        
        with open(file) as f:
            lines = f.readlines()
            for item in lines:
                item_split = item.split()
                if(len(item_split) <= 0):
                    continue
                if item_split[0] == "training_pp:":
                    train_pp_list.append(float(item_split[-1]))
                elif item_split[0] == "training:":
                    train_accuracy_list.append(float(item_split[-1]))
                elif item_split[0] == "validation_pp:":
                    test_pp_list.append(float(item_split[-1]))
                elif item_split[0] == "validation:":
                    test_accuracy_list.append(float(item_split[-1]))
        return train_pp_list, train_accuracy_list, test_pp_list, test_accuracy_list

        
    def PlotContrast(self, ScalarList, length, the_name):
        writer = SummaryWriter()
        writer.add_text('title',the_name)
        for epoch in range(length):
            TheScalar = {}
            for name in ScalarList.keys():
                TheScalar[name] = ScalarList[name][epoch]
            writer.add_scalars(the_name, TheScalar, epoch)
        writer.close()
        
    def __init__(self, model_list, type, name):
        '''
        model_list:要可视化的不同模型信息
        type=loss:train loss曲线
        type=accuracy：test accuracy曲线
        name:标题
        '''
        super(PlotCurvesNLP, self).__init__()
        ScalarList = {}
        length = -1
        for item in model_list:
            model_name = item["model_name"]
            model_dir = item["model_dir"]
            log_place = constants.result_dir + model_dir + ".txt"
            train_pp_list, train_accuracy_list, test_pp_list, test_accuracy_list = self.GetAccuracyAndLoss(log_place)
            if length < 0:
                length = len(train_pp_list)
            if(type == 'loss'):
                ScalarList[model_name] = train_pp_list
            else:
                ScalarList[model_name] = test_pp_list
        self.PlotContrast(ScalarList, length, name)

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


        
    def PlotContrast(self, ScalarList, length, the_name):
        writer = SummaryWriter()
        writer.add_text('title',the_name)
        for epoch in range(length):
            TheScalar = {}
            for name in ScalarList.keys():
                TheScalar[name] = ScalarList[name][epoch]
            
            writer.add_scalars(the_name, TheScalar, epoch)

        writer.close()
        
    def __init__(self, model_list, type, name):
        '''
        model_list:要可视化的不同模型信息
        type=loss:train loss曲线
        type=accuracy：test accuracy曲线
        name:标题
        '''
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
            if(type == 'loss'):
                ScalarList[model_name] = train_loss_list
            else:
                ScalarList[model_name] = test_accuracy_list
        self.PlotContrast(ScalarList, length, name)
        
if __name__ == '__main__':
    PlotBase = []
    Baseline = {"model_name":"RAdam", "model_dir":"radam_no"}
    Ranger = {"model_name":"Ranger", "model_dir":"radam_yes"}
    PlotBase.append(Baseline)
    PlotBase.append(Ranger)

    Plot001 = []
    RAdam001 = {"model_name":"RAdam", "model_dir":"radam_no_001"}
    Ranger001 = {"model_name":"Ranger", "model_dir":"radam_yes_001"}
    Plot001.append(RAdam001)
    Plot001.append(Ranger001)
    
    PlotFast = []
    RAdamFast = {"model_name":"RAdam", "model_dir":"radam_no_fast"}
    RangerFast = {"model_name":"Ranger", "model_dir":"radam_yes_fast"}
    PlotFast.append(RAdamFast)
    PlotFast.append(RangerFast)
    
    PlotParams = []
    RAdam = {"model_name":"RAdam Baseline", "model_dir":"radam_no"}
    Ranger = {"model_name":"Ranger Baseline", "model_dir":"radam_yes"}
    Ranger05 = {"model_name":"Ranger When inner learning rate alpha = 0.5", "model_dir":"radam_yes_05"}
    Ranger10 = {"model_name":"Ranger When inner steps k = 10", "model_dir":"radam_yes_10"}
    PlotParams.append(RAdam)
    PlotParams.append(Ranger)
    PlotParams.append(Ranger05)
    PlotParams.append(Ranger10)
    
    PlotAblation = []
    SGD = {"model_name":"SGD", "model_dir":"sgd_no"}
    Adam = {"model_name":"Adam", "model_dir":"adam_no"}
    RAdam = {"model_name":"RAdam", "model_dir":"radam_no"}
    SGDLookahead = {"model_name":"SGD + lookahead", "model_dir":"sgd_yes"}
    AdamLookahead = {"model_name":"Adam + lookahead", "model_dir":"adam_yes"}
    Ranger = {"model_name":"Ranger(RAdam + lookahead)", "model_dir":"radam_yes"}
    PlotAblation.append(SGD)
    PlotAblation.append(Adam)
    PlotAblation.append(RAdam)
    PlotAblation.append(SGDLookahead)
    PlotAblation.append(AdamLookahead)
    PlotAblation.append(Ranger)
    
    PlotBatch = []
    Batch2 = {"model_name":"2 steps, 320 batchs", "model_dir":"adam_yes_2_320"}
    Batch5 = {"model_name":"5 steps, 128 batchs", "model_dir":"adam_yes"}
    Batch10 = {"model_name":"10 steps, 64 batchs", "model_dir":"adam_yes_10_64"}
    PlotBatch.append(Batch2)
    PlotBatch.append(Batch5)
    PlotBatch.append(Batch10)


    PlotBase100 = []
    Baseline100 = {"model_name":"RAdam", "model_dir":"radam_no_cifar100"}
    Ranger100 = {"model_name":"Ranger", "model_dir":"radam_yes_cifar100"}
    PlotBase100.append(Baseline100)
    PlotBase100.append(Ranger100)

    Plot001100 = []
    RAdam001100 = {"model_name":"RAdam", "model_dir":"radam_no_001_cifar100"}
    Ranger001100 = {"model_name":"Ranger", "model_dir":"radam_yes_03_cifar100"}
    Plot001100.append(RAdam001100)
    Plot001100.append(Ranger001100)
    
    PlotFast100 = []
    RAdamFast100 = {"model_name":"RAdam", "model_dir":"radam_no_fast_cifar100"}
    RangerFast100 = {"model_name":"Ranger", "model_dir":"radam_yes_fast_cifar100"}
    PlotFast100.append(RAdamFast100)
    PlotFast100.append(RangerFast100)
    
    PlotParams100 = []
    RAdam100 = {"model_name":"RAdam Baseline", "model_dir":"radam_no_cifar100"}
    Ranger100 = {"model_name":"Ranger Baseline", "model_dir":"radam_yes_cifar100"}
    Ranger05100 = {"model_name":"Ranger When inner learning rate alpha = 0.5", "model_dir":"radam_yes_05_cifar100"}
    Ranger10100 = {"model_name":"Ranger When inner steps k = 10", "model_dir":"radam_yes_10_cifar100"}
    PlotParams100.append(RAdam100)
    PlotParams100.append(Ranger100)
    PlotParams100.append(Ranger05100)
    PlotParams100.append(Ranger10100)
    
    PlotAblation100 = []
    SGD100 = {"model_name":"SGD", "model_dir":"sgd_no_cifar100"}
    Adam100 = {"model_name":"Adam", "model_dir":"adam_no_cifar100"}
    RAdam100 = {"model_name":"RAdam", "model_dir":"radam_no_cifar100"}
    SGDLookahead100 = {"model_name":"SGD + lookahead", "model_dir":"sgd_yes_cifar100"}
    AdamLookahead100 = {"model_name":"Adam + lookahead", "model_dir":"adam_yes_cifar100"}
    Ranger100 = {"model_name":"Ranger(RAdam + lookahead)", "model_dir":"radam_yes_cifar100"}
    PlotAblation100.append(SGD100)
    PlotAblation100.append(Adam100)
    PlotAblation100.append(RAdam100)
    PlotAblation100.append(SGDLookahead100)
    PlotAblation100.append(AdamLookahead100)
    PlotAblation100.append(Ranger100)

    PlotBatch100 = []
    Batch2100 = {"model_name":"2 steps, 320 batchs", "model_dir":"adam_yes_2_320_cifar100"}
    Batch5100 = {"model_name":"5 steps, 128 batchs", "model_dir":"adam_yes_cifar100"}
    Batch10100 = {"model_name":"10 steps, 64 batchs", "model_dir":"adam_yes_10_64_cifar100"}
    PlotBatch100.append(Batch2100)
    PlotBatch100.append(Batch5100)
    PlotBatch100.append(Batch10100)


    PlotPTB = []
    RAdamPTB = {"model_name":"RAdam", "model_dir":"nlp_no"}
    RangerPTB = {"model_name":"Ranger", "model_dir":"nlp_yes"}
    PlotPTB.append(RAdamPTB)
    PlotPTB.append(RangerPTB)

    PlotIMDB = []
    RAdamIMDB = {"model_name":"RAdam", "model_dir":"imdb_no"}
    RangerIMDB = {"model_name":"Ranger", "model_dir":"imdb_yes"}
    PlotIMDB.append(RAdamIMDB)
    PlotIMDB.append(RangerIMDB)

    PlotCNN = []
    RAdamCNN = {"model_name":"RAdam", "model_dir":"cnn_no"}
    RangerCNN = {"model_name":"Ranger", "model_dir":"cnn_yes"}
    PlotCNN.append(RAdamCNN)
    PlotCNN.append(RangerCNN)


    print("please input the type you want")
    print("000:Contrast between Ranger and RAdam on cifar10 -- train loss")
    print("001:Contrast between Ranger and RAdam on cifar10 -- test accuracy")
    print("010:Contrast between Ranger and RAdam on cifar10 when lr = 0.01 -- train loss")
    print("011:Contrast between Ranger and RAdam on cifar10 when lr = 0.01 -- test accuracy")
    print("020:Contrast between Ranger and RAdam on cifar10 when lr decays faster -- train loss")
    print("021:Contrast between Ranger and RAdam on cifar10 when lr decays faster -- test accuracy")
    print("030:Contrast between Ranger and RAdam on cifar10 when Ranger's lookahead uses different parameters -- train loss")
    print("031:Contrast between Ranger and RAdam on cifar10 when Ranger's lookahead uses different parameters -- test accuracy")
    print("040:Ranger's ablation study on cifar10 -- train loss")
    print("041:Ranger's ablation study on cifar10 -- test accuracy")
    print("050:Contrast between different combinations of inner steps and batches on cifar10 -- train loss")
    print("051:Contrast between different combinations of inner steps and batches on cifar10 -- test accuracy")   
    
    print("100:Contrast between Ranger and RAdam on cifar100 -- train loss")
    print("101:Contrast between Ranger and RAdam on cifar100 -- test accuracy")
    print("110:Contrast between Ranger and RAdam on cifar100 when lr = 0.01 -- train loss")
    print("111:Contrast between Ranger and RAdam on cifar100 when lr = 0.01 -- test accuracy")
    print("120:Contrast between Ranger and RAdam on cifar100 when lr decays faster -- train loss")
    print("121:Contrast between Ranger and RAdam on cifar100 when lr decays faster -- test accuracy")
    print("130:Contrast between Ranger and RAdam on cifar100 when Ranger's lookahead uses different parameters -- train loss")
    print("131:Contrast between Ranger and RAdam on cifar100 when Ranger's lookahead uses different parameters -- test accuracy")
    print("140:Ranger's ablation study on cifar100 -- train loss")
    print("141:Ranger's ablation study on cifar100 -- test accuracy")
    print("150:Contrast between different combinations of inner steps and batches on cifar100 -- train loss")
    print("151:Contrast between different combinations of inner steps and batches on cifar100 -- test accuracy") 
    
    print("200:Contrast between Ranger and RAdam on ptb -- train perplexity")
    print("201:Contrast between Ranger and RAdam on ptb -- test perplexity")
    print("300:Contrast between Ranger and RAdam on imdb sentiment analyze -- train loss")
    print("301:Contrast between Ranger and RAdam on imdb sentiment analyze -- test accuracy")
    print("400:Contrast between Ranger and RAdam on hw2 dataset and model -- train loss")
    print("401:Contrast between Ranger and RAdam on hw2 dataset and model -- test accuracy")
    
    
    num = str(input())
    if num == "000":
        PlotCurves(PlotBase, 'loss', "The train loss contrast on cifar10")
    elif num == "001":
        PlotCurves(PlotBase, 'accuracy', "The test accuracy contrast on cifar10")
    elif num == "010":
        PlotCurves(Plot001, 'loss', "The train loss contrast on cifar10 when lr = 0.01")
    elif num == "011":
        PlotCurves(Plot001, 'accuracy', "The test accuracy contrast on cifar10 when lr = 0.01")
    elif num == "020":
        PlotCurves(PlotFast, 'loss', "The train loss contrast on cifar10 when lr decays faster")
    elif num == "021":
        PlotCurves(PlotFast, 'accuracy', "The test accuracy contrast on cifar10 when lr decays faster")
    elif num == "030":
        PlotCurves(PlotParams, 'loss', "The train loss contrast on cifar10 when Ranger's lookahead uses different parameters")
    elif num == "031":
        PlotCurves(PlotParams, 'accuracy', "The test accuracy contrast on cifar10 when Ranger's lookahead uses different parameters")
    elif num == "040":
        PlotCurves(PlotAblation, 'loss', "The train loss contrast of Ranger's ablation study on cifar10")
    elif num == "041":
        PlotCurves(PlotAblation, 'accuracy', "The test accuracy contrast of Ranger's ablation study on cifar10")
    elif num == "050":
        PlotCurves(PlotBatch, 'loss', "The train loss contrast between different combinations of inner steps and batches on cifar10")
    elif num == "051":
        PlotCurves(PlotBatch, 'accuracy', "The test accuracy contrast between different combinations of inner steps and batches on cifar10")


    elif num == "100":
        PlotCurves(PlotBase100, 'loss', "The train loss contrast on cifar100")
    elif num == "101":
        PlotCurves(PlotBase100, 'accuracy', "The test accuracy contrast on cifar100")
    elif num == "110":
        PlotCurves(Plot001100, 'loss', "The train loss contrast on cifar100 when lr = 0.01")
    elif num == "111":
        PlotCurves(Plot001100, 'accuracy', "The test accuracy contrast on cifar100 when lr = 0.01")
    elif num == "120":
        PlotCurves(PlotFast100, 'loss', "The train loss contrast on cifar100 when lr decays faster")
    elif num == "121":
        PlotCurves(PlotFast100, 'accuracy', "The test accuracy contrast on cifar100 when lr decays faster")
    elif num == "130":
        PlotCurves(PlotParams100, 'loss', "The train loss contrast on cifar100 when Ranger's lookahead uses different parameters")
    elif num == "131":
        PlotCurves(PlotParams100, 'accuracy', "The test accuracy contrast on cifar100 when Ranger's lookahead uses different parameters")
    elif num == "140":
        PlotCurves(PlotAblation100, 'loss', "The train loss contrast of Ranger's ablation study on cifar100")
    elif num == "141":
        PlotCurves(PlotAblation100, 'accuracy', "The test accuracy contrast of Ranger's ablation study on cifar100")
    elif num == "150":
        PlotCurves(PlotBatch100, 'loss', "The train loss contrast between different combinations of inner steps and batches on cifar100")
    elif num == "151":
        PlotCurves(PlotBatch100, 'accuracy', "The test accuracy contrast between different combinations of inner steps and batches on cifar100")
      
    elif num == '200':
        PlotCurvesNLP(PlotPTB, 'loss', 'The train perplexity contrast on PTB.')
    elif num == '201':
        PlotCurvesNLP(PlotPTB, 'accuracy', 'The test perplexity contrast on PTB.')   
    elif num == '300':
        PlotCurves(PlotIMDB, 'loss', 'The train loss contrast on IMDB sentiment analyze.')
    elif num == '301':
        PlotCurves(PlotIMDB, 'accuracy', 'The test accuracy contrast on IMDB sentiment analyze.')  
    elif num == '400':
        PlotCurves(PlotCNN, 'loss', 'The train loss contrast on hw2 Image set and model.')
    elif num == '401':
        PlotCurves(PlotCNN, 'accuracy', 'The test accuracy contrast on hw2 Image set and model.')