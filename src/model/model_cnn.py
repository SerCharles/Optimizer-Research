from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


def ResNet18(num_classes):
    model_resnet = models.resnet18(pretrained = False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet
