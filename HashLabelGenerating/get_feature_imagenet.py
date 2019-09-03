import torch
import pretrainedmodels
import h5py
import os.path
import numpy as np
import logging

__PATH__ = '../data/imagenet'

logging.basicConfig(level=logging.INFO,filename='imagenet_data_test.log',format="%(levelname)s:%(asctime)s:%(message)s")

def getAllids():
    id_filename = 'id_imagenet.txt'
    id_txt = os.path.join(__PATH__, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    return _ids


def getData():
    filename = 'imagenet_data.h5'
    file = os.path.join(__PATH__, filename)
    try:
        data = h5py.File(file, 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    return data

def get_data_imagenet_from_file():
    ids = getAllids()
    data = getData()
    testset = []
    for i in range(len(ids)):
        np_images = data[ids[i]]['image'].value
        images = torch.from_numpy(np_images)
        testset.append(images)

    return testset


def get_dataloader(testset):
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return testloader

def get_net():
    model = pretrainedmodels.models.inceptionv4(num_classes=1000, pretrained='imagenet')
    model.eval()
    return model

def get_features():
    testset_imagenet = get_data_imagenet_from_file()
    testloader_imagenet = get_dataloader(testset_imagenet)
    output_features = []

    if torch.cuda.is_available():
        model = get_net()
        model = model.cuda()

        for data in testloader_imagenet:
            images = data
            logging.info(images)
            images = images.cuda()
            output_features1 = model(images)
            output_features1 = output_features1.cpu()
            output_features1 = output_features1.detach().numpy()
            output_features.append(output_features1)

    else:
        model = get_net()

        for data in testloader_imagenet:
            images = data
            output_features1 = model(images)
            output_features1 = output_features1.detach().numpy()
            output_features.append(output_features1)

    return output_features


if __name__ == '__main__':
    np_features = get_features()
    np_features_reshape = np.reshape(np_features, (-1, 1536))
    h5f = h5py.File(os.path.join(__PATH__, 'imagenet_features.h5'), 'w')
    h5f.create_dataset("imagenet_features", data=np_features_reshape)
    h5f.close()
