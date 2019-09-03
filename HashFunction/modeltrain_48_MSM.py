import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
from torchvision.models import ResNet
import numpy as np
import os
from torch.autograd import Variable
from get_data import *
import logging

logging.basicConfig(level=logging.INFO, filename='resnet18_MSMloss_48.log', format="%(levelname)s:%(asctime)s:%(message)s")


def get_dataloader(dataset,batch_size,shuffle):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader

mysize=[3,224,224]
trainset_path = '../data/cifar10/train/whole'
trainset_id = 'id_cifar10.txt'
trainset = 'cifar10_data.h5'
hash_label_path = '../data/cifar10/hashlabel/whole/48'
hash_label = 'cifar10_whole_hashlabel_48.h5'
hash_len = 48
batch_size = 32
num_epochs = 800

trainset_data = get_function_train_dataset(mysize, trainset_path, trainset_id, trainset)
input_dataset = get_dataloader(trainset_data, batch_size, False)

label_data = get_hash_labels(hash_label_path, hash_label)
input_labels = get_dataloader(label_data, batch_size, False)

# -------------------------模型选择，优化方法， 学习率策略----------------------
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('resnet18-5c106cde.pth'))

for parma in model.parameters():
    parma.requires_grad = False

num_fc_in = model.fc.in_features

model.fc = torch.nn.Sequential(nn.Linear(num_fc_in, 256),
			     nn.LeakyReLU(negative_slope=0.2,inplace=True),
			     nn.BatchNorm1d(256),
			     nn.Linear(256, 64),
			     nn.LeakyReLU(negative_slope=0.2,inplace=True),
			     nn.BatchNorm1d(64),
			     nn.Linear(64, hash_len),
			     nn.BatchNorm1d(hash_len),
                 nn.Sigmoid())


if torch.cuda.is_available():
    model.cuda()
loss_fc = nn.MultiLabelSoftMarginLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
model.train()

for epoch in range(num_epochs):


    for data, label in zip(input_dataset, input_labels):
        if torch.cuda.is_available():
            data = Variable(data).float().cuda()
            label = Variable(label).float().cuda()

        else:
            data = Variable(data).float()
            label = Variable(label).float()

        optimizer.zero_grad()
        hash_value = model(data)
        loss = loss_fc(hash_value, label)
        logging.info(loss)
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        print(epoch)
        logging.info(epoch)
        print(loss)
        torch.save(model.state_dict(), '../data/cifar10/hashmodel/whole/48/cifar10_whole_hashmodel_48_MSM.pth')

print('training finish !')
torch.save(model.state_dict(), '../data/cifar10/hashmodel/whole/48/cifar10_whole_hashmodel_48_MSM.pth')
