import torch
import torch.nn as nn
from torch.autograd import Variable
import h5py
import os.path
from get_data import *
import logging
from torchvision import models

logging.basicConfig(level=logging.INFO, filename='test_hashcode.log', format="%(levelname)s:%(asctime)s:%(message)s")

model_path_name = '../data/cifar10/hashmodel/whole/48/cifar10_whole_hashmodel_48.pth'
hash_path = '../data/cifar10/hashcode/whole/48'
hash_name = 'cifar10_whole_hashcode_48.h5'
num_bit = 48
download_flag = True
label_path = '../data/cifar10/test'
label_name = 'cifar10_test_label.h5'
mysize=[3,224,224]

def get_testdataloader(dataset, batch_size, shuffle):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader

model = models.resnet18(pretrained=False)

for parma in model.parameters():
    parma.requires_grad = False


num_fc_in = model.fc.in_features


model.fc = torch.nn.Sequential(nn.Linear(num_fc_in, 256),
			     nn.LeakyReLU(negative_slope=0.2,inplace=True),
			     nn.BatchNorm1d(256),
			     nn.Linear(256, 64),
			     nn.LeakyReLU(negative_slope=0.2,inplace=True),
			     nn.BatchNorm1d(64),
			     nn.Linear(64, num_bit),
			     nn.BatchNorm1d(num_bit),
                 nn.Sigmoid())


model.load_state_dict(torch.load(model_path_name))


if torch.cuda.is_available():
    model.cuda()

testset = get_cifar10_testdata(mysize,download_flag, label_path, label_name)
testset = get_testdataloader(testset, 16, False)
print("test already")

model.eval()

f = open('test3.txt','a')


testhashh5 = h5py.File(os.path.join(hash_path, hash_name), 'w')

count = 0
test_hash = []

for data in testset:

    if torch.cuda.is_available():
        data = Variable(data).float().cuda()
    else:
        data = Variable(data).float()

    if torch.cuda.is_available():
        hash_value = model(data)
        hash_value = hash_value.cpu().detach().numpy()
    else:
        hash_value = model(data)
        hash_value = hash_value.detach().numpy()

    logging.info(hash_value)

    hashbool = hash_value > 0.5
    hashcode = hashbool.astype(np.int32)
    logging.info(hashcode)

    hashtest3 = hash_value > 0.3
    hashcode3 = hashtest3.astype(np.int32)
    f.writelines(str(hashcode3))

    if count < (10000/16):
        test_hash.append(hashcode)

        count = count+1
    else:
        break

test_hash = np.reshape(test_hash, (-1, num_bit))
print(len(test_hash))
testhashh5.create_dataset("testhash", data=test_hash)
testhashh5.close()
f.close()









