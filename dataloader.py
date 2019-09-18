import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import subset_sampler


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0, path="./data/mnist") :
    if data_aug :
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            ])
    else :
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
            datasets.MNIST(root=path, train=True, download=True, transform=transform_train),
            batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
            )
    train_eval_loader = DataLoader(
            datasets.MNIST(root=path, train=True, download=True, transform=transform_test),
            batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
            )

    test_loader = DataLoader(
            datasets.MNIST(root=path, train=False, download=True, transform=transform_test),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
            )

    return train_loader, test_loader, train_eval_loader

def get_cifar10_loaders(data_aug=True, batch_size=128, test_batch_size=500, perc=1.0, path="./data/cifar10") :
    mean = (0.4914, 0.4822, 0.2265)
    std = (0.2023, 0.1994, 0.2010)
    if data_aug :
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    else :
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

    train_loader = DataLoader(
            datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train),
            batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
            )
    eval_loader = DataLoader(
            datasets.CIFAR10(root=path, train=True, download=True, transform=transform_test),
            batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
            )

    test_loader = DataLoader(
            datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
            )

    return train_loader, test_loader, eval_loader


def get_mnist_subset(sample_size=10, train_data=False, path="./data/mnist") :
    source = datasets.MNIST(root=path, train=train_data, download=True, transform=transforms.ToTensor())
    return subset_sampler(source, num_image=sample_size)

def get_cifar10_subset(sample_size=10, train_data=False, path="./data/cifar10") :
    source = datasets.CIFAR10(root=path, train=train_data, download=True, transform=transforms.ToTensor())
    return subset_sampler(source, num_image=sample_size)

def normalize(image) :
    image = image.clone()
    mean = (0.4914, 0.4822, 0.2265)
    std = (0.2023, 0.1994, 0.2010)
    if len(image.shape) == 3 :
        image = image.reshape(1, *image.size())
    image[:,0,:,:] = (image[:,0,:,:] - mean[0]) / std[0]
    image[:,1,:,:] = (image[:,1,:,:] - mean[1]) / std[1]
    image[:,2,:,:] = (image[:,2,:,:] - mean[2]) / std[2]
    if image.size(0) == 1 :
        return image.reshape(*image.size()[1:])
    return image

def inverse_normalize(image) :
    image = image.clone()
    mean = (0.4914, 0.4822, 0.2265)
    std = (0.2023, 0.1994, 0.2010)
    if len(image.shape) == 3 :
        image = image.reshape(1, *image.size())
    image[:,0,:,:] = image[:,0,:,:] * std[0] + mean[0]
    image[:,1,:,:] = image[:,1,:,:] * std[1] + mean[1]
    image[:,2,:,:] = image[:,2,:,:] * std[2] + mean[2]
    if image.size(0) == 1 :
        return image.reshape(*image.size()[1:])
    return image

def inf_generator(iterable) :
    iterator = iterable.__iter__()
    while True :
        try :
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
