import torchvision
import torchvision.transforms as transforms
import h5py
import os.path
import logging

__PATH__ = '../data/imagenet'

logging.basicConfig(level=logging.INFO,filename='imagenet_data.log',format="%(levelname)s:%(asctime)s:%(message)s")

def get_data_imagenet():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    testset = torchvision.datasets.ImageFolder(root='../data/imagenet/imagenet_test_50000', transform=transform)

    return testset

def data_imagenet():
    testset_imagenet = get_data_imagenet()
    data_id = open(os.path.join(__PATH__, 'id_imagenet.txt'), 'w')
    h5f = h5py.File(os.path.join(__PATH__, 'imagenet_data.h5'), 'w')
    count = 0

    for data in testset_imagenet:
        grp = h5f.create_group(str(count))
        data_id.write(str(count) + '\n')
        images, labels = data
        logging.info(images)
        grp['image'] = images.detach().numpy()
        count = count+1


    h5f.close()
    data_id.close()
    return


if __name__ == '__main__':
    data_imagenet()