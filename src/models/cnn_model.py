'''
描述：加载resnet18模型进行cifar10/100分类
'''


from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import constants


def ResNet18(num_classes):
    '''
    描述：加载resnet18模型接口
    参数：分类数目
    返回：模型
    '''
    model_resnet = models.resnet18(pretrained = False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

