import torchvision
import torchvision.transforms as transforms
import h5py
import os.path
import logging

__PATH__ = '../data/cifar10'
logging.basicConfig(level=logging.INFO,filename='cifar10_data.log',format="%(levelname)s:%(asctime)s:%(message)s")

def get_data_cifar10(download_flag):
    transform = transforms.Compose(
        [transforms.Resize(299),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True,
                                            download=download_flag, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False,
                                           download=download_flag, transform=transform)

    return trainset, testset


def data_cifar10():
    trainset_cifar10, testset_cifar10 = get_data_cifar10(False)
    data_id = open(os.path.join(__PATH__, 'id_cifar10.txt'), 'w')
    h5f = h5py.File(os.path.join(__PATH__, 'cifar10_data.h5'), 'w')
    count = 0

    for data in trainset_cifar10:
        grp = h5f.create_group(str(count))
        data_id.write(str(count) + '\n')
        images, labels = data
        logging.info(images)
        grp['image']=images.detach().numpy()
        count=count+1

    h5f.close()
    data_id.close()
    return


if __name__ == '__main__':
    data_cifar10()