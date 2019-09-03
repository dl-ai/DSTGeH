import os.path
import h5py
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

import random


def convert_vec2str(a):
    return ''.join([str(i) for i in a])

def hashism(a):
    hashstrs = []
    for vec in a:
        hashstrs.append([convert_vec2str(vec)])
    return hashstrs


def get_h5_data(path,filename):
    file = os.path.join(path, filename)
    try:
        data = h5py.File(file, 'r')
    except:
        raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
    return data


def get_test_hashcode(path,filename):
    file = get_h5_data(path,filename)
    datas = file['testhash'].value
    return hashism(datas)

def get_test_label(path,filename):
    test_labels=get_h5_data(path,filename)['testlabel']
    test_labelset=[]
    for test_label in test_labels:
        test_labelset.append([test_label])
    return test_labelset


class Create_query():
    def __init__(self, hashs, labels):
        self.hashs = hashs
        self.labels = labels
        self.samplesum = len(labels)

        self.hash_goal = []
        self.label_goal = []

    def __add_goal(self, cursor):
        self.hash_goal.append(self.hashs[cursor])
        self.label_goal.append(self.labels[cursor])


    def get_rand_query_data(self,snum):
        self.hash_goal = []
        self.label_goal = []
        cursor = random.randint(0, 100)
        csnum = 0
        while csnum < snum:
            seed = 1 + random.randint(0, 100) % 2
            cursor = (cursor + seed) % self.samplesum
            csnum = csnum + 1
            self.__add_goal(cursor)
        return self.hash_goal, self.label_goal



    def get_certain_query_data(self,cls):
        self.hash_goal = []
        self.label_goal = []
        for cng in cls:
            clsid = cng[0]
            clsnum = cng[1]
            cursor = random.randint(0,100)
            cclsnum = 0
            while cclsnum<clsnum:
                seed = 1+random.randint(0,100)%2
                cursor = (cursor+seed)%self.samplesum
                if self.labels[cursor][0] == clsid:
                    cclsnum = cclsnum + 1
                    self.__add_goal(cursor)
        return self.hash_goal,self.label_goal

    def get_hash_list(self):
        return self.hash_goal

    def get_label_list(self):
        return self.label_goal
