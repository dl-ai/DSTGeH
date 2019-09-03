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
__PATH__HASHLABEL = '../data/cifar10/hashlabel/whole/48'

logging.basicConfig(level=logging.INFO, filename='cifar10_whole_hashlabel_48.log', format="%(levelname)s:%(asctime)s:%(message)s")

def get_relations():
    h5file_relations = h5py.File(os.path.join(__PATH__GRAPH, 'relations_cifar10_imagenet.h5'), 'r')
    np_relations = h5file_relations['relations_cifar10_imagenet']
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

    batch_size_test = 16
    hash_len = 48
    relations = get_relations()

    model = StackedAutoEncoder(hash_len)
    model.load_state_dict(torch.load('./autoencoder_cifar10_48.pth'))
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    dataloader_test = get_dataloader(relations, batch_size_test, False)

    hashtags_file = h5py.File(os.path.join(__PATH__HASHLABEL, 'cifar10_whole_hashlabel_48.h5'), 'w')
    count = 0
    hashtags = []
    for relation in dataloader_test:
        relation = Variable(relation).float().cuda()
        tag = model(relation, batch_size_test).cpu()
        tag = tag.detach().numpy()
        hashbool = tag > 0
        hashtag = hashbool.astype(np.int32)
        logging.info(hashtag)
        if count < (50000/batch_size_test):
            hashtags.append(hashtag)
            count = count + 1
        else:
            break
    hashtags = np.reshape(hashtags, (-1, hash_len))
    logging.info(hashtags.shape)
    hashtags_file.create_dataset("hashcode", data=hashtags)
    hashtags_file.close()