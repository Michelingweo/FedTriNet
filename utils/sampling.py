#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import math

def mnist_iid(dataset, num_users,label_rate):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]#定义dict_users为空字典，all_idxs是所有数据的序号
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))#从all_indxs中随机不放回抽num_items个data
    #     all_idxs = list(set(all_idxs) - dict_users[i])#更新all_idxs
    #     # dict_users = {user_idx : set(data_idx) }
    # return dict_users

    num_items = int(len(dataset) / num_users)
    dict_users, dict_users_labeled = {}, {}
    dict_users_unlabeled = {idxs: set() for idxs in range(num_users)}
    # pseduo_label, all_idxs = [i for i in range(len(dataset))], [i for i in range(len(dataset))]
    all_idxs = [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users_labeled[i] = set(np.random.choice(list(dict_users[i]), int(num_items * label_rate), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        for idxs in list(dict_users[i] - dict_users_labeled[i]):
            dict_users_unlabeled[i].add(idxs)

    # return dict_users, dict_users_labeled, pseduo_label
    return dict_users, dict_users_labeled, dict_users_unlabeled



def mnist_noniid(dataset, num_users, label_rate):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:dic_users
    """
    num_shards, num_imgs = 200, 300
    num_items = int(len(dataset) / num_users)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_labeled = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.label_list)
    print(np.size(labels))
    print(np.size(idxs))
    # sort labels
    idxs_labels = np.vstack((idxs, labels)) # 拼接index与label
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # 按照 label 顺序排列
    idxs = idxs_labels[0,:] # 取index

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) # 所有shard里面随机选两个
        idx_shard = list(set(idx_shard) - rand_set) # 更新全部shard
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            dict_users_labeled[i] = set(np.random.choice(list(dict_users[i]), int(num_items * label_rate), replace=False))
    return dict_users, dict_users_labeled


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def iid_sample(dataset, label_rate,num_users=10):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]#定义dict_users为空字典，all_idxs是所有数据的序号
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))#从all_indxs中随机不放回抽num_items个data
    #     all_idxs = list(set(all_idxs) - dict_users[i])#更新all_idxs
    #     # dict_users = {user_idx : set(data_idx) }
    # return dict_users

    num_items = int(len(dataset) / num_users)
    num_users = int(num_users)
    dict_users, dict_users_labeled = {}, {}
    dict_users_unlabeled = {idxs: set() for idxs in range(num_users)}
    pseduo_label, all_idxs = [i for i in range(len(dataset))], [i for i in range(len(dataset))]
#     all_idxs = [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users_labeled[i] = set(np.random.choice(list(dict_users[i]), int(num_items * label_rate), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        for idxs in list(dict_users[i] - dict_users_labeled[i]):
            dict_users_unlabeled[i].add(idxs)
            pseduo_label[idxs] = -1

    # return dict_users, dict_users_labeled, pseduo_label
    return dict_users, dict_users_labeled, pseduo_label

def noniid_sample(dataset, label_rate, num_users=10, ppt = 0.5, Distrib = 'fixed'):
    '''
    :param dataset: dataset param
    :param label_rate: percentage of labeled data
    :param num_users: only works for 10 clients setting
    :param ppt: proportion for data dsitribution
    :param Distrib: fixed or random
    :return:
    '''

    num_items = int(len(dataset) / num_users)
    num_users = int(num_users)
    dict_users, dict_users_labeled = {}, {}
    pseduo_label, all_idxs = [i for i in range(len(dataset))], [i for i in range(len(dataset))]
    dict_users_unlabeled = {idxs: set() for idxs in range(int(num_users))}

    dict_idxs_class = {i: set() for i in range(10)}
    dict_idxs_shards = {i: set() for i in range(10)}

    for idxs in range(len(dataset)):
        label_class = dataset[idxs][1]
        dict_idxs_class[label_class] = dict_idxs_class[label_class] | set([idxs])
        pseduo_label[idxs] = dataset[idxs][1]

    for i in range(10):
        dict_idxs_class[i] = list(dict_idxs_class[i])
        # divid into shards
        for idx in range(math.floor(len(dict_idxs_class[i])*ppt)):
            dict_idxs_shards[i].add(dict_idxs_class[i][idx])
            dict_idxs_class[i].remove(dict_idxs_class[i][idx])


    if Distrib == 'fixed':
        # fixed shards combine
        for j in range(10,20):
            dict_idxs_shards[j] = dict_idxs_class[j-10]
        for i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
            dict_idxs_shards[i - 1] = list(dict_idxs_shards[i]) + list(dict_idxs_shards[i - 1])
        for id in range(num_users):
            dict_users[id] = dict_idxs_shards[2*id]
    else:
        # random shards combine
        a_list = list(dict_idxs_shards.keys())
        b_list = list(dict_idxs_class.keys())
        for i in range(num_users):
            a = b = 0
            while (a == b):
                a = int(np.random.choice(a_list, 1))
                b = int(np.random.choice(b_list, 1))
            print("a",a)
            print("b",b,'\n')
            a_list.remove(a)
            b_list.remove(b)
            dict_users[i] = list(dict_idxs_shards[a]) + list(dict_idxs_class[b])

    for i in range(num_users):
        dict_users_labeled[i] = set(np.random.choice(list(dict_users[i]), int(len(dict_users[i]) * label_rate), replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        for idxs in list(set(dict_users[i]) - set(dict_users_labeled[i])):
            dict_users_unlabeled[i].add(idxs)
            pseduo_label[idxs] = -1
    for i in range(10):
        dict_users[i] = set(dict_users[i])

        dict_users_labeled[i] = set(dict_users_labeled[i])
    return dict_users, dict_users_labeled, pseduo_label
#     return dict_users, dict_users_labeled, dict_users_unlabeled

def noniid_ii_sample(dataset, label_rate, num_users=10, ppt = 0.5, Distrib = 'fixed'):#_ii
    '''
    :param dataset: dataset param
    :param label_rate: percentage of labeled data
    :param num_users: only works for 10 clients setting
    :param ppt: proportion for data dsitribution
    :param Distrib: fixed or random
    :return:
    '''

    num_items = int(len(dataset) / num_users)
    num_users = int(num_users)
    dict_users, dict_users_labeled = {}, {}
    pseduo_label, all_idxs = [i for i in range(len(dataset))], [i for i in range(len(dataset))]
    dict_users_unlabeled = {idxs: set() for idxs in range(int(num_users))}

    dict_idxs_class = {i: set() for i in range(10)}
    dict_idxs_shards = {i: set() for i in range(10)}

    for idxs in range(len(dataset)):
        label_class = dataset[idxs][1]
        dict_idxs_class[label_class] = dict_idxs_class[label_class] | set([idxs])
        pseduo_label[idxs] = dataset[idxs][1]

    len_unlabeled={}
    for idxs in range(10):
        len_unlabeled[idxs] = len(dict_idxs_class[idxs]) / num_users * (1-label_rate)
        
    for i in range(num_users):
        for idxs in range(10):
            set_selected = set(np.random.choice(list(dict_idxs_class[idxs]), int(len_unlabeled[idxs]), replace=False))
            dict_users_unlabeled[i] = dict_users_unlabeled[i] | set_selected 
            dict_idxs_class[idxs] = dict_idxs_class[idxs] - set_selected

    for i in range(10):
        dict_idxs_class[i] = list(dict_idxs_class[i])
        # divid into shards
        for idx in range(math.floor(len(dict_idxs_class[i])*ppt)):
            dict_idxs_shards[i].add(dict_idxs_class[i][idx])
            dict_idxs_class[i].remove(dict_idxs_class[i][idx])


    if Distrib == 'fixed':
        # fixed shards combine
        for j in range(10,20):
            dict_idxs_shards[j] = dict_idxs_class[j-10]
        for i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
            dict_idxs_shards[i - 1] = list(dict_idxs_shards[i]) + list(dict_idxs_shards[i - 1])
        for id in range(num_users):
            dict_users_labeled[id] = dict_idxs_shards[2*id]
    else:
        # random shards combine
        a_list = list(dict_idxs_shards.keys())
        b_list = list(dict_idxs_class.keys())
        for i in range(num_users):
            a = b = 0
            while (a == b):
                a = int(np.random.choice(a_list, 1))
                b = int(np.random.choice(b_list, 1))
            print("a",a)
            print("b",b,'\n')
            a_list.remove(a)
            b_list.remove(b)
            dict_users_labeled[i] = list(dict_idxs_shards[a]) + list(dict_idxs_class[b])

    for i in range(10):
        dict_users_labeled[i] = set(dict_users_labeled[i])
        dict_users[i] = dict_users_unlabeled[i] | dict_users_labeled[i]
#         print(len(dict_users[i]),len(dict_users_unlabeled[i]),len(dict_users_labeled[i]))

    for i in range(num_users): 
        for idxs in list(set(dict_users[i]) - set(dict_users_labeled[i])):
            pseduo_label[idxs] = -1   
    return dict_users, dict_users_labeled, pseduo_label
#     return dict_users, dict_users_labeled, dict_users_unlabeled


def noniid_iii_sample(dataset,label_rate, num_users):#iii
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]#定义dict_users为空字典，all_idxs是所有数据的序号
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))#从all_indxs中随机不放回抽num_items个data
    #     all_idxs = list(set(all_idxs) - dict_users[i])#更新all_idxs
    #     # dict_users = {user_idx : set(data_idx) }
    # return dict_users

    num_items = int(len(dataset) / num_users)
    dict_users, dict_users_labeled = {}, {}
    dict_users_unlabeled = {idxs: set() for idxs in range(num_users)}
    dict_label_rate = {}
    pseduo_label, all_idxs = [i for i in range(len(dataset))], [i for i in range(len(dataset))]
#     all_idxs = [i for i in range(len(dataset))]
    for idxs in range(len(dataset)):
        pseduo_label[idxs] = dataset[idxs][1]
        
    for i in range(num_users):
        # you could change these parameters.
        if i<5 :
            dict_label_rate[i] = label_rate * 0.5
        else:
            dict_label_rate[i] = label_rate * 1.5


    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        dict_users_labeled[i] = set(np.random.choice(list(dict_users[i]), int(num_items * dict_label_rate[i]), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        for idxs in list(dict_users[i] - dict_users_labeled[i]):
            dict_users_unlabeled[i].add(idxs)
            pseduo_label[idxs] = -1 

    return dict_users, dict_users_labeled, pseduo_label
#     return dict_users, dict_users_labeled, dict_users_unlabeled




if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

