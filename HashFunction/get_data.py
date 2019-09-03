import os.path
import h5py
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


def get_cifar10_testdata(mysize,download_flag,label_path,label_name):
    tsf = transforms.Compose(
        [transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='../data/cifar10/',train=False,download=download_flag,transform=tsf)
    imageset = []
    labelset = []
    testlabelh5 = h5py.File(os.path.join(label_path,label_name),'w')
    for data in testset:
        images, labels = data
        np_images=images.detach().numpy()
        np_images_resize = np.resize(np_images, mysize)
        images = torch.from_numpy(np_images_resize)
        imageset.append(images)
        labelset.append(labels)
    testlabelh5.create_dataset("testlabel",data=labelset)
    testlabelh5.close()
    return imageset


def get_h5_data(path,filename):
    file = os.path.join(path, filename)
    try:
        data = h5py.File(file, 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    return data


def get_train_id(path,id_filename):

    id_txt = os.path.join(path, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    return _ids

def get_function_train_dataset(mysize,path = '../data/cifar10/train/whole' ,id_filename = 'id_cifar10.txt',filename = 'cifar10_data.h5'):
    ids = get_train_id(path, id_filename)
    data = get_h5_data(path, filename)
    trainset = []
    for i in range(len(ids)):
        np_images = data[ids[i]]['image']
        np_images_resize = np.resize(np_images, mysize)
        images = torch.from_numpy(np_images_resize)
        trainset.append(images)

    return trainset

def get_hash_labels(path,filename):
    data = get_h5_data(path,filename)
    labels = []
    np_labels = data['hashcode']
    for np_label in np_labels:
        np_label = np_label.astype(np.int64)
        labels.append(np_label)
    labels = np.array(labels)
    return labels
