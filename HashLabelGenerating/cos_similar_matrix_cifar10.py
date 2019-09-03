# coding: utf-8
import datetime
import numpy as np
import h5py
import os.path
import logging

__PATH__CIFAR10 = '../data/cifar10'
__PATH__IMAGENET = '../data/imagenet'
__PATH__GRAPH = '../data/graph/cifar10_imagenet'

logging.basicConfig(level=logging.INFO,filename='cos_similar_matrix_imagenet_cifar10.log',format="%(levelname)s:%(asctime)s:%(message)s")

def cosine_distance(vector_a, vector_b):
    cosdist = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    return cosdist

def cos_similar_matrix(feature_set, feature_num):
    relations = np.zeros((feature_num, feature_num))
    for i in range(0,feature_num):
        for j in range(i,feature_num):
            cosdist = cosine_distance(feature_set[i], feature_set[j])
            if i < j:
                relations[i][j] = cosdist
                relations[j][i] = cosdist
            elif i == j:
                relations[i][j] = 1
    return relations


if __name__ == "__main__":
    h5file_cifar10 = h5py.File(os.path.join(__PATH__CIFAR10, 'cifar10_features.h5'), 'r')
    h5file_imagenet = h5py.File(os.path.join(__PATH__IMAGENET, 'imagenet_features.h5'), 'r')
    h5file_relations = h5py.File(os.path.join(__PATH__GRAPH, 'relations_cifar10_imagenet.h5'), 'w')

    cifar_features = h5file_cifar10['cifar10_features'].value
    imagenet_features = h5file_imagenet['imagenet_features'].value
    print(len(cifar_features))
    print(len(imagenet_features))
    print(cifar_features.shape)
    print(imagenet_features.shape)
    feature_set = np.concatenate((cifar_features, imagenet_features), 0)
    feature_num = len(feature_set)


    starttime = datetime.datetime.now()

    relations = cos_similar_matrix(feature_set, feature_num)
    logging.info(relations)
    logging.info(relations.shape)

    endtime = datetime.datetime.now()
    logging.info('usetime:')
    logging.info((endtime - starttime).seconds)


    h5file_relations.create_dataset("relations_cifar10_imagenet", data=relations)

    h5file_cifar10.close()
    h5file_imagenet.close()
    h5file_relations.close()
