# coding: utf-8
import numpy as np
import h5py
import os.path
import torch
import torch.utils.data
from torch.autograd import Variable
from torch import nn
from autoencoder import StackedAutoEncoder
import logging

__PATH__GRAPH = '../data/graph/cifar10_imagenet'

logging.basicConfig(level=logging.INFO, filename='euc_autoencoder_loss_48.log', format="%(levelname)s:%(asctime)s:%(message)s")

def get_relations():
    h5file_relations = h5py.File(os.path.join(__PATH__GRAPH, 'euc_relations_cifar10_imagenet.h5'), 'r')
    np_relations = h5file_relations['euc_relations_cifar10_imagenet']
    relations = []
    for np_relation in np_relations:
        relation = torch.from_numpy(np_relation)
        relation = relation.reshape(10, 100, 100)
        relations.append(relation)

    logging.info(len(relations))
    h5file_relations.close()
    return relations

def get_dataloader(trainset, batch_size, shuffle):
    dataloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader


if __name__ == "__main__":

    batch_size_train = 32
    num_epochs = 8000
    hash_len = 48
    relations = get_relations()
    pretrain = True

    model = StackedAutoEncoder(hash_len)
    if pretrain:
        model.load_state_dict(torch.load('./euc_autoencoder_cifar10_48.pth'))
    if torch.cuda.is_available():
        model = model.cuda()


    criterion = nn.MSELoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    model.train()
    dataloader_train = get_dataloader(relations, batch_size_train, True)

    for epoch in range(num_epochs):
        train_count = 0
        for data in dataloader_train:
            data = Variable(data).float().cuda()
            H, data_reconstruct, loss_layer_sum = model(data, batch_size_train)
            loss_stacked = criterion(data_reconstruct, Variable(data, requires_grad=False))
            loss_reconstructed = 0.7 * loss_stacked + 0.3 * loss_layer_sum
            train_count = train_count+1
            adj_batch = data.reshape(batch_size_train, 100000)[:, (train_count-1)*batch_size_train:train_count*batch_size_train]
            D = torch.diag(torch.sum(adj_batch, dim=0))
            L = D - adj_batch
            loss_hidden = 2 * torch.trace(torch.matmul(torch.matmul(torch.transpose(H, 0, 1), L), H))
            loss_sum = 0.9*loss_reconstructed+0.099*loss_hidden
            logging.info(loss_sum.item())
        if epoch <= 6000:
            optimizer1.zero_grad()
            loss_sum.backward()
            optimizer1.step()
        else:
            optimizer2.zero_grad()
            loss_sum.backward()
            optimizer2.step()
        if epoch % 10 == 0:
            logging.info("epoch:")
            logging.info(epoch)
            logging.info("loss_sum:")
            logging.info(loss_sum.item())
            print(epoch)
            torch.save(model.state_dict(), './euc_autoencoder_cifar10_48.pth')

