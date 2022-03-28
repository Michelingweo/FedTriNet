#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7
import os
from skimage import io, transform
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
from collections import OrderedDict

from list_txt.make_list import make_list, relabel
from models.Update import DatasetSplit

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update1 import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, discriminator
from models.Fed import FedAvg
from models.test import test_img, test_img_client,test_img_client2
from list_txt.make_list import maskvector_init

class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image







if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print('user_num:',args.num_users)
    print('label_rate:',args.label_rate)
    print('p1epochs:',args.p1epochs)
    print('p2epochs:',args.p2epochs)
    # Set random seed for reproducibility
    manualSeed = 999
#     manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # '''----------------------------------------------选数据集--------------------------------------------'''

    # load dataset and split users

    # dataset[0]: set of (img, label)  dataset[i][0]: No.i img  dataset[i][1]: No.i label
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
#         dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transform)
        # sample users
        if args.iid:
            dict_users, dict_users_label, dict_users_unlabel = mnist_iid(dataset_train, args.num_users, args.label_rate)
        else:
            dict_users, dict_users_label = mnist_noniid(dataset_train, args.num_users, args.label_rate)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users, dict_users_label = mnist_iid(dataset_train, args.num_users, args.label_rate)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'svhn':
        transform_svhn = transforms.Compose([
                        RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_svhn_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.SVHN('./data/svhn',  split='train', download=True, transform=transform_svhn)
        dataset_test = datasets.SVHN('./data/svhn',  split='test', download=True, transform=transform_svhn_test)

        if args.iid:
            dict_users, dict_users_label = mnist_iid(dataset_train, args.num_users, args.label_rate)
        else:
            exit('Error: only consider IID setting in svhn')


    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # '''--------------------------------------------选网络--------------------------------------------------------'''
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_C = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_C = CNNMnist(args=args).to(args.device)
        net_D = discriminator().to(args.device)
        
        net_D.weight_init()
    elif args.dataset == 'svhn':
        net_C = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_C = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_C)
    print(net_D)
    net_C.train()

    # copy weights
    Cw_glob = net_C.state_dict()

    # mask_vector initialization
    # 0: 1-labeled 0:pseudo-label -1-unlabeled 1:user's belonging 2: label/pseudo-label  3: data idx
    mask_vector, label_refer = maskvector_init(dataset_train, args, dict_users, dict_users_label)
    # print('label_vector length:', len(label_vector))
    # print('unlabel_vector length:', len(unlabel_vector))
    
    img_size = dataset_train[0][0].shape
    img_size = img_size[1]
    # label preprocess
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
    fill = torch.zeros([10, 10, img_size, img_size])
    for i in range(10):
        fill[i, i, :, :] = 1
    
    # training
    Closs_train = []
    Dloss_train = []
#     cv_loss, cv_acc = [], []
#     val_loss_pre, counter = 0, 0
#     net_best = None
#     best_loss = None
#     val_acc_list, net_list = [], []

    localNet_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}#存储local net所有layer的para
    localNet_last2_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}#存储后两层layer的para
    m = max(int(args.frac * args.num_users), 1)  # max(0.1*100,1)=10


    # -----------------------------------------initialization-------------------------------------------

    for iter in range(int(args.num_users/m)):

        
        w_fist2_collection, Closs_locals= [], []
        Cw_glob_avg = []
        
        if args.num_users > 10:
            idxs_users = np.arange(iter*10,(iter+1)*10,1)
        else:
            idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            # first epoch
            # train using only labeled data
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx],stage=0)  # define
            w, Closs = local.init_train(net=copy.deepcopy(net_C).to(args.device))
            localNet_dic[idx] = w # w :ordereddic ---> localNet_dic[idx]: ordereddic
            # update后两层网络 localNet_last2_dic[idx]:dict

            if args.dataset == 'mnist':
                localNet_last2_dic[idx] = {
                    'fc1.weight': w['fc1.weight'], 'fc1.bias': w['fc1.bias'],
                'fc2.weight': w['fc2.weight'], 'fc2.bias': w['fc2.bias']
                                           }
            elif args.dataset == 'svhn' or args.dataset == 'cifar' :
                localNet_last2_dic[idx] = {"conv_layer.0.weight":w["conv_layer.0.weight"],
                                           "conv_layer.0.bias":w["conv_layer.0.bias"],
                                           'conv_layer.1.weight': w['conv_layer.1.weight'],'conv_layer.1.bias': w['conv_layer.1.bias'],
                                           'conv_layer.1.running_mean':w[ 'conv_layer.1.running_mean'],
                                           'conv_layer.1.running_var':w['conv_layer.1.running_var'],
                                           'conv_layer.1.num_batches_tracked':w[ 'conv_layer.1.num_batches_tracked'],
                                           'conv_layer.3.weight':w['conv_layer.3.weight'], 'conv_layer.3.bias':w['conv_layer.3.bias'],
                                           'conv_layer.7.weight':w['conv_layer.7.weight'], 'conv_layer.7.bias':w['conv_layer.7.bias'],
                                           'conv_layer.8.weight': w['conv_layer.8.weight'],
                                           'conv_layer.8.bias': w['conv_layer.8.bias'],
                                           'conv_layer.8.running_mean': w['conv_layer.8.running_mean'],
                                           'conv_layer.8.running_var': w['conv_layer.8.running_var'],
                                           'conv_layer.8.num_batches_tracked': w['conv_layer.8.num_batches_tracked'],
                                           'conv_layer.10.weight': w['conv_layer.10.weight'],
                                           'conv_layer.10.bias': w['conv_layer.10.bias'],
                                          'fc_layer.1.weight': w['fc_layer.1.weight'], 'fc_layer.1.bias': w['fc_layer.1.bias'],
                                           'fc_layer.3.weight': w['fc_layer.3.weight'], 'fc_layer.3.bias': w['fc_layer.3.bias'],
                                           'fc_layer.6.weight': w['fc_layer.6.weight'], 'fc_layer.6.bias': w['fc_layer.6.bias']}

            Cw_glob_avg.append(copy.deepcopy(w)) # w :ordereddic ---> w_glob_avg: list of ordereddic
            # 前两层网络 w_fisrt2: dict
            if args.dataset == 'mnist' :
                w_first2 = {'conv1.weight': w['conv1.weight'], 'conv1.bias': w['conv1.bias'],
                           'conv2.weight': w['conv2.weight'], 'conv2.bias': w['conv2.bias']}
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked'],
                            'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias']}
            # 将待average的前两层网络加入w_locals中
            a_ = OrderedDict(w_first2) # w_first2: dict ---> a_:ordereddic
            w_fist2_collection.append(copy.deepcopy(a_)) # a_:ordereddic ---> w_fist2_clct: list of ordereddic
            Closs_locals.append(copy.deepcopy(Closs))
        # update global weights---first 2 layers' parameters
        Cw_glob = FedAvg(w_fist2_collection) # a_:ordereddic ---> w_glob : ordereddic

        # update net_glob
        Cw_glob_avg_dic = FedAvg(Cw_glob_avg) # w_glob_avg_dic:ordereddic
        net_C.load_state_dict(Cw_glob_avg_dic)

        # print loss
        Closs_avg = sum(Closs_locals) / len(Closs_locals)
        print('Initial: Round {:3d}, Average loss {:.3f}'.format(iter, Closs_avg))
        Closs_train.append(Closs_avg)

    test_data = DataLoader(dataset_test, batch_size=1, shuffle=False)
    max_prob = []
    for (img, label) in test_data:
        img, label = img.to(args.device), label.to(args.device).long
        img.cuda()
        log_probs_init = F.softmax(net_C(img), dim=1)
        pseudo_label = log_probs_init.data.max(1, keepdim=True)[1]
        max_prob.append(log_probs_init.data.max(1, keepdim=True)[0])

    max_prob = torch.tensor(max_prob)
    print('Initial Mean Max_prob:',torch.mean(max_prob))
    #Initialization result
    # plot loss curve
    plt.figure()
    plt.plot(range(len(Closs_train)), Closs_train,color='red')
    
    plt.ylabel('train_loss')
    plt.savefig('./save/init_{}_user{}_ep{}+{}_lbr{}_iid{}.png'.format(args.dataset, args.num_users, args.p1epochs,args.p2epochs, args.label_rate, args.iid))
    # # testing
    net_C.eval()
    acc_train, loss_train = test_img(net_C, dataset_train, args)
    acc_test, loss_test = test_img(net_C, dataset_test, args)
    print("Initial Global Training accuracy: {:.2f}".format(acc_train))
    print("Initial Global Testing accuracy: {:.2f}".format(acc_test))
    net_C.train()
    


    #--------------------------------------------pre-train---------------------------------------------------------

    Closs_train = []
    Dloss_train = []
    for iter in range(args.p1epochs):

        w_fist2_collection, Closs_locals, Dloss_locals = [], [],[]
        Cw_glob_avg,Dw_glob_avg = [],[]
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = np.arange(10)
        for idx in idxs_users:
            # first epoch
            # train using only labeled data
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx],idxs2=dict_users_unlabel[idx],stage=1)  # define
            w,Dw,Closs, Dloss = local.Phase1_train(C=copy.deepcopy(net_C).to(args.device),D=copy.deepcopy(net_D).to(args.device),img_size=img_size)
            localNet_dic[idx] = w # w :ordereddic ---> localNet_dic[idx]: ordereddic
            # update后两层网络 localNet_last2_dic[idx]:dict
            if args.dataset == 'mnist':
                localNet_last2_dic[idx] = {
                    'fc1.weight': w['fc1.weight'], 'fc1.bias': w['fc1.bias'],
                    'fc2.weight': w['fc2.weight'], 'fc2.bias': w['fc2.bias']
                }
            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                localNet_last2_dic[idx] = {"conv_layer.0.weight":w["conv_layer.0.weight"],
                                           "conv_layer.0.bias":w["conv_layer.0.bias"],
                                            'conv_layer.1.weight': w['conv_layer.1.weight'],
                                           'conv_layer.1.bias': w['conv_layer.1.bias'],
                                           'conv_layer.1.running_mean': w['conv_layer.1.running_mean'],
                                           'conv_layer.1.running_var': w['conv_layer.1.running_var'],
                                           'conv_layer.1.num_batches_tracked': w['conv_layer.1.num_batches_tracked'],
                                           'conv_layer.3.weight': w['conv_layer.3.weight'],
                                           'conv_layer.3.bias': w['conv_layer.3.bias'],
                                           'conv_layer.7.weight': w['conv_layer.7.weight'],
                                           'conv_layer.7.bias': w['conv_layer.7.bias'],
                                           'conv_layer.8.weight': w['conv_layer.8.weight'],
                                           'conv_layer.8.bias': w['conv_layer.8.bias'],
                                           'conv_layer.8.running_mean': w['conv_layer.8.running_mean'],
                                           'conv_layer.8.running_var': w['conv_layer.8.running_var'],
                                           'conv_layer.8.num_batches_tracked': w['conv_layer.8.num_batches_tracked'],
                                           'conv_layer.10.weight': w['conv_layer.10.weight'],
                                           'conv_layer.10.bias': w['conv_layer.10.bias'],
                                           'fc_layer.1.weight': w['fc_layer.1.weight'],
                                           'fc_layer.1.bias': w['fc_layer.1.bias'],
                                           'fc_layer.3.weight': w['fc_layer.3.weight'],
                                           'fc_layer.3.bias': w['fc_layer.3.bias'],
                                           'fc_layer.6.weight': w['fc_layer.6.weight'],
                                           'fc_layer.6.bias': w['fc_layer.6.bias']}

            Cw_glob_avg.append(copy.deepcopy(w)) # w :ordereddic ---> w_glob_avg: list of ordereddic
            Dw_glob_avg.append(copy.deepcopy(Dw))
            # 前两层网络 w_fisrt2: dict
            if args.dataset == 'mnist' :
                w_first2 = {'conv1.weight': w['conv1.weight'], 'conv1.bias': w['conv1.bias'],
                            'conv2.weight': w['conv2.weight'], 'conv2.bias': w['conv2.bias']}
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked'],
                            'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias']}
            # 将待average的前两层网络加入w_locals中
            a_ = OrderedDict(w_first2) # w_first2: dict ---> a_:ordereddic
            w_fist2_collection.append(copy.deepcopy(a_)) # a_:ordereddic ---> w_fist2_clct: list of ordereddic
            Closs_locals.append(copy.deepcopy(Closs))
            Dloss_locals.append(copy.deepcopy(Dloss))
        # update global weights---first 2 layers' parameters
        Cw_glob = FedAvg(w_fist2_collection) # a_:ordereddic ---> w_glob : ordereddic

        # update net_glob
        Cw_glob_avg_dic = FedAvg(Cw_glob_avg) # w_glob_avg_dic:ordereddic
        net_C.load_state_dict(Cw_glob_avg_dic)
        Dw_glob_avg_dic = FedAvg(Dw_glob_avg) 
        net_D.load_state_dict(Dw_glob_avg_dic)

        # print loss
        Closs_avg = sum(Closs_locals) / len(Closs_locals)
        Dloss_avg = sum(Dloss_locals) / len(Dloss_locals)
        print('Phase1: Round {:3d}, Average C loss {:.3f}, Average D loss {:.3f}'.format(iter, Closs_avg,Dloss_avg))
        Closs_train.append(Closs_avg)
        Dloss_train.append(Dloss_avg)

    test_data = DataLoader(dataset_test, batch_size=1, shuffle=False)
    max_prob = []
    for (img, label) in test_data:
        img, label = img.to(args.device), label.to(args.device).long
        img.cuda()
        log_probs_init = F.softmax(net_C(img), dim=1)
        pseudo_label = log_probs_init.data.max(1, keepdim=True)[1]
        max_prob.append(log_probs_init.data.max(1, keepdim=True)[0])

    max_prob = torch.tensor(max_prob)
    print('pre-train Mean Max_prob:',torch.mean(max_prob))
    #Initialization result
    # plot loss curve
    plt.figure()
    plt.plot(range(len(Closs_train)), Closs_train,color='red')
    plt.plot(range(len(Dloss_train)), Dloss_train,color='blue')
    plt.ylabel('train_loss')
    plt.savefig('./save/p1_{}_user{}_ep{}+{}_lbr{}_iid{}.png'.format(args.dataset, args.num_users, args.p1epochs,args.p2epochs, args.label_rate, args.iid))
    # # testing
    net_C.eval()
    acc_train, loss_train = test_img(net_C, dataset_train, args)
    acc_test, loss_test = test_img(net_C, dataset_test, args)
    print("Pre train Global Training accuracy: {:.2f}".format(acc_train))
    print("Pre train Global Testing accuracy: {:.2f}".format(acc_test))
    net_C.train()
    
    # client testing
    acc_train_list, loss_train_list = [], []
    acc_test_list, loss_test_list = [], []
    for i in range(args.num_users):
        if args.dataset == 'mnist' :
                net = CNNMnist(args=args).to(args.device)
        elif args.dataset == 'svhn' or args.dataset == 'cifar':
                net = CNNCifar(args=args).to(args.device)
        net.load_state_dict(localNet_dic[i])
        net.eval()
        acc_train, loss_train = test_img_client(net_C, net, dataset_train, args, dict_users[i])
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        acc_test, loss_test = test_img_client2(net_C,net, dataset_test, args)
        acc_test_list.append(acc_test)
        loss_test_list.append(loss_test)
    acc_train_avg = np.max(acc_train_list)
    acc_test_avg = np.max(acc_test_list)
    print("Phase1 User's Training accuracy: {:.2f}".format(acc_train_avg))
    print("Phase1 Use's Testing accuracy: {:.2f}".format(acc_test_avg))
    
    

    # training
    Closs_train = []
    Dloss_train = []
#     cv_loss, cv_acc = [], []
#     val_loss_pre, counter = 0, 0
#     net_best = None
#     best_loss = None
#     val_acc_list, net_list = [], []
    
    localifpseudo_dic = {idxs:0 for idxs in range(args.num_users)}
            # '''-----------------------------------------Semi-train-------------------------------------------'''
    # train dict:  key-user_idx : set(labeldata_idx + pseudolabel_idx)
    dict_train = dict_users_label
    dict_pseudo = {idxs: set() for idxs in range(args.num_users)}

    for iter in range(int(args.p2epochs)):
        
        w_fist2_clct, Closs_locals, Dloss_locals = [], [],[]
        Cw_glob_avg,Dw_glob_avg = [],[]

        updated_pseudo_num = []
        pseudo_correctRate = []
        epoch_dis_avg = []
        


        m = max(int(args.frac * args.num_users), 1)  # max(0.1*10,1)=10
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if iter < 10:
            threshold = 2.8 * float(torch.mean(max_prob))
        elif iter < 35:
            threshold = (100 - iter * 2) / 100 * 2.8 * float(torch.mean(max_prob))
        else:threshold = 2.8 * float(torch.mean(max_prob)) / 2

        print('Round:',iter,'threshold:',threshold)

        for idx in idxs_users:
            
            client_dis_avg = []
            updated_pseudo_count = 0

            # net_2tune
            if args.dataset == 'mnist' :
                net_2tune = CNNMnist(args=args).to(args.device)
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                net_2tune = CNNCifar(args=args).to(args.device)

            # 对临时变量_赋值前两层经过fedavg的结果
            _ = dict(Cw_glob)   # w_glob : ordereddic ---> _ : dic
            # 拼接后两层参数
            _.update(localNet_last2_dic[idx])
            # 转换为ordereddic类型
            new_para = OrderedDict(_)  # _ : dic ---> new_para : ordereddic

            net_2tune.load_state_dict(new_para)
            # freeze
            if args.dataset == 'mnist':
                net_2tune.conv1.weight.requires_grad = False
                net_2tune.conv1.bias.requires_grad = False

                net_2tune.conv2.weight.requires_grad = False
                net_2tune.conv2.bias.requires_grad = False
            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                net_2tune.conv1.requires_grad = False
                net_2tune.conv2.requires_grad = False


            # '''*********************************fine-tune**********************************************************'''
            local_tune = LocalUpdate(args=args, maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx],stage=0)
            w_tuned, loss_tuned = local_tune.fine_tune(net=copy.deepcopy(net_2tune).to(args.device))

            # Net-new&old
            if args.dataset == 'mnist':
                net_Cnew = CNNMnist(args=args).to(args.device)
                net_Cold = CNNMnist(args=args).to(args.device)
            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                net_Cnew = CNNCifar(args=args).to(args.device)
                net_Cold = CNNCifar(args=args).to(args.device)
                # 拼接所得新网络
            # net_new.load_state_dict(w_tuned)
            net_Cnew.load_state_dict(w_tuned)
            old_para = localNet_dic[idx]
            net_Cold.load_state_dict(old_para)


            net_C.eval()
            acc_test, loss_test = test_img(net_C, dataset_test, args)
            print("Round:{:3d}, user:{:3d},Glob Testing accuracy: {:.2f}".format(iter,idx, acc_test))
            net_C.train()

            net_Cold.eval()
            old_test, old_loss_test = test_img(net_Cold, dataset_test, args)
            print("\t\t\tnet_old Testing accuracy: {:.2f}".format(old_test))
            net_Cold.train()

            net_Cnew.eval()
            new_test, new_loss_test = test_img(net_Cnew, dataset_test, args)
            print("\t\t\tnet_new Testing accuracy: {:.2f}".format(new_test))
            net_Cnew.train()
            # '''*********************************Relabel**********************************************************'''
            # Begin Pseduo Label
            net_D.eval()
            try_data = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=1, shuffle=False)
            total_pseudo_num = 0
            label_correct_count = 0
            # throughout all the data
            for i, (img, label) in zip(dict_users[idx],try_data):
                # if data is unlabeled
                if mask_vector[i][0] == -1:
                    img, label = img.to(args.device), label.to(args.device).long
                    log_probs1 = F.softmax(net_Cold(img), dim=1)
                    log_probs2 = F.softmax(net_Cnew(img), dim=1)
                    log_probs3 = F.softmax(net_C(img), dim=1)
                    

                    log_probs = log_probs1 + log_probs2 + log_probs3
                    pseudo_label = log_probs.data.max(1, keepdim=True)[1]
                    max_prob_pseudo = log_probs.data.max(1, keepdim=True)[0]
                    
                    _ = torch.argmax(log_probs, dim=1).long()
                    label_d = fill[_].cuda()
                    ifmatch = torch.mean(net_D(img,label_d))
                    client_dis_avg.append(ifmatch)
    

                    if float(max_prob_pseudo) >= threshold and ifmatch > 0.5:
                        mask_vector[i][2] = int(pseudo_label)
                        mask_vector[i][0] = 0 # twice relabeling is not allowed

                        userid = int(idx)
                        dict_train[userid].add(i)
                        dict_pseudo[userid].add(i)

                        total_pseudo_num += 1
                        updated_pseudo_count+=1
                        if pseudo_label == label_refer[i]:
                            label_correct_count += 1
                   
                        
                    total_pseudo_num = max(total_pseudo_num,1)
                    localifpseudo_dic[idx] += total_pseudo_num
                    if total_pseudo_num == 0 :
                        continue
                    else:
                        pseudo_correctRate.append(label_correct_count/total_pseudo_num)
            
            updated_pseudo_num.append(updated_pseudo_count)
            epoch_dis_avg.append(sum(client_dis_avg)/len(client_dis_avg))
            net_D.train()
            # *********************************local update**********************************************************
            # local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx])
#             local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx],idxs2=dict_pseudo[idx])
            local1 = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx],idxs2=dict_users_unlabel[idx],idxs3 = dict_pseudo[idx],stage=1)
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx],idxs2=dict_users_unlabel[idx],idxs3 = dict_pseudo[idx])
            if localifpseudo_dic[idx] < 30:
                w,Dw,Closs, Dloss = local1.Phase1_train(C=copy.deepcopy(net_C).to(args.device),D=copy.deepcopy(net_D).to(args.device),img_size=img_size)
            else:
                w,Dw,Closs, Dloss = local.Phase2_train(C=copy.deepcopy(net_C).to(args.device),D=copy.deepcopy(net_D).to(args.device),img_size=img_size, sever_round=iter)
            # w, loss = local.semitrain(net=copy.deepcopy(net_new).to(args.device), sever_round=iter, args=args)
            Closs_locals.append(copy.deepcopy(Closs))
            Cw_glob_avg.append(copy.deepcopy(w))
            Dloss_locals.append(copy.deepcopy(Dloss))
            Dw_glob_avg.append(copy.deepcopy(Dw))
            # 更新local net参数
            localNet_dic[idx] = w
            # 前两层网络
            if args.dataset == 'mnist' :
                w_first2 = {'conv1.weight': w['conv1.weight'], 'conv1.bias': w['conv1.bias'],
                           'conv2.weight': w['conv2.weight'], 'conv2.bias': w['conv2.bias']}
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked'],
                            'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias']}
            # 将待average的前两层网络加入w_locals中
            a_ = OrderedDict(w_first2)  # w_first2: dict ---> a_:ordereddic
            w_fist2_clct.append(copy.deepcopy(a_))  # a_:ordereddic ---> w_fist2_clct: list of ordereddic
            # loss_locals.append(copy.deepcopy(loss))
            # update后两层网络
            if args.dataset == 'mnist':
                localNet_last2_dic[idx] = {
                    'fc1.weight': w['fc1.weight'], 'fc1.bias': w['fc1.bias'],
                    'fc2.weight': w['fc2.weight'], 'fc2.bias': w['fc2.bias']
                }

            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                localNet_last2_dic[idx] = {"conv_layer.0.weight":w["conv_layer.0.weight"],
                                           "conv_layer.0.bias":w["conv_layer.0.bias"],
                                           'conv_layer.1.weight': w['conv_layer.1.weight'],
                                           'conv_layer.1.bias': w['conv_layer.1.bias'],
                                           'conv_layer.1.running_mean': w['conv_layer.1.running_mean'],
                                           'conv_layer.1.running_var': w['conv_layer.1.running_var'],
                                           'conv_layer.1.num_batches_tracked': w['conv_layer.1.num_batches_tracked'],
                                           'conv_layer.3.weight': w['conv_layer.3.weight'],
                                           'conv_layer.3.bias': w['conv_layer.3.bias'],
                                           'conv_layer.7.weight': w['conv_layer.7.weight'],
                                           'conv_layer.7.bias': w['conv_layer.7.bias'],
                                           'conv_layer.8.weight': w['conv_layer.8.weight'],
                                           'conv_layer.8.bias': w['conv_layer.8.bias'],
                                           'conv_layer.8.running_mean': w['conv_layer.8.running_mean'],
                                           'conv_layer.8.running_var': w['conv_layer.8.running_var'],
                                           'conv_layer.8.num_batches_tracked': w['conv_layer.8.num_batches_tracked'],
                                           'conv_layer.10.weight': w['conv_layer.10.weight'],
                                           'conv_layer.10.bias': w['conv_layer.10.bias'],
                                           'fc_layer.1.weight': w['fc_layer.1.weight'],
                                           'fc_layer.1.bias': w['fc_layer.1.bias'],
                                           'fc_layer.3.weight': w['fc_layer.3.weight'],
                                           'fc_layer.3.bias': w['fc_layer.3.bias'],
                                           'fc_layer.6.weight': w['fc_layer.6.weight'],
                                           'fc_layer.6.bias': w['fc_layer.6.bias']}

        # update global weights---first 2 layers' parameters
        Cw_glob = FedAvg(w_fist2_clct)
        # update net_glob
        Cw_glob_avg_dic = FedAvg(Cw_glob_avg)
        net_C.load_state_dict(Cw_glob_avg_dic)
        Dw_glob_avg_dic = FedAvg(Dw_glob_avg)
        net_D.load_state_dict(Dw_glob_avg_dic)

        # print loss
        Closs_avg = sum(Closs_locals) / len(Closs_locals)
        print('Phase2: Round {:3d}, Average C loss {:.3f}, Average D loss {:.3f}'.format(iter, Closs_avg,Dloss_avg))
        Closs_train.append(Closs_avg)

        pseudo_correctRate = torch.tensor(pseudo_correctRate)
        print('Relabel accuracy:', torch.mean(pseudo_correctRate))

        updated_pseudo_num = torch.tensor(updated_pseudo_num).float()
        print('Mean updated data num:', torch.mean(updated_pseudo_num))
        epoch_dis_avg = torch.tensor(epoch_dis_avg).float()
        print('Mean discriminator judge rate:', torch.mean(epoch_dis_avg))


    # result
    # plot loss curve
    plt.figure()
    plt.plot(range(len(Closs_train)), Closs_train,color='red')
    plt.plot(range(len(Dloss_train)), Dloss_train,color='blue')
    plt.ylabel('train_loss')
    plt.savefig('./save/p2_{}_user{}_ep{}+{}_lbr{}_iid{}.png'.format(args.dataset, args.num_users, args.p1epochs,args.p2epochs, args.label_rate, args.iid))
    # # testing
    net_C.eval()
    acc_train, loss_train = test_img(net_C, dataset_train, args)
    acc_test, loss_test = test_img(net_C, dataset_test, args)
    print("Phase2 Global Training accuracy: {:.2f}".format(acc_train))
    print("Phase2 Global Testing accuracy: {:.2f}".format(acc_test))


    # client testing
    acc_train_list, loss_train_list = [], []
    acc_test_list, loss_test_list = [], []
    for i in range(args.num_users):
        if args.dataset == 'mnist' :
                net = CNNMnist(args=args).to(args.device)
        elif args.dataset == 'svhn' or args.dataset == 'cifar':
                net = CNNCifar(args=args).to(args.device)
        net.load_state_dict(localNet_dic[i])
        net.eval()
        acc_train, loss_train = test_img_client(net_C, net, dataset_train, args, dict_users[i])
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        acc_test, loss_test = test_img_client2(net_C,net, dataset_test, args)
        acc_test_list.append(acc_test)
        loss_test_list.append(loss_test)
    acc_train_avg = np.max(acc_train_list)
    acc_test_avg = np.max(acc_test_list)
    print("Final User's Training accuracy: {:.2f}".format(acc_train_avg))
    print("Final Use's Testing accuracy: {:.2f}".format(acc_test_avg))
