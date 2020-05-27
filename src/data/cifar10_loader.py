from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os


def load_data(data_dir = "../../data/", batch_size = 128):
    '''
    描述：加载cifar-10数据集，并且进行transformation
    参数：数据集位置，batch size
    返回：train loader，test loader
    '''
    #data augmentation
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    num_classes = 10
    train_dataset = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     transform=train_transform,
                                     download=False)

    test_dataset = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    transform=test_transform,
                                    download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)
    return train_loader, test_loader