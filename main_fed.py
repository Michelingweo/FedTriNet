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
import xlwt,xlrd
import time

from list_txt.make_list import make_list, relabel
from models.Update import DatasetSplit

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.sampling import iid_sample,noniid_sample,noniid_ii_sample,noniid_iii_sample
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
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
    print('localepochs:',args.local_ep)
    print('dataset:',args.dataset)
    print('frac:',args.frac)
    print('fine tune',args.finetune)
    print('threshold protection',args.tp)
    print('pseudo_select',args.pseudo_select)
    print('compare_key',args.compare_key)
    print('swa', args.swa)
    # Set random seed for reproducibility
#     manualSeed = 999
#     #     manualSeed = random.randint(1, 10000) # use if you want new results
#     print("Random Seed: ", manualSeed)
#     random.seed(manualSeed)
#     torch.manual_seed(manualSeed)
#     np.random.seed(manualSeed)
    # '''----------------------------------------------选数据集--------------------------------------------'''

    # load dataset and split users

    # dataset[0]: set of (img, label)  dataset[i][0]: No.i img  dataset[i][1]: No.i label
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
#         if args.iid:
#             dict_users, dict_users_label = mnist_iid(dataset_train, args.num_users,args.label_rate)
#         else:
#             dict_users, dict_users_label = mnist_noniid(dataset_train, args.num_users, args.label_rate)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
#         if args.iid:
#             dict_users, dict_users_label = mnist_iid(dataset_train, args.num_users, args.label_rate)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')

    elif args.dataset == 'svhn':
        transform_svhn = transforms.Compose([
                        RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_svhn_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.SVHN('../data/svhn',  split='train', download=True, transform=transform_svhn)
        dataset_test = datasets.SVHN('../data/svhn',  split='test', download=True, transform=transform_svhn_test)

#         if args.iid:
#             dict_users, dict_users_label = mnist_iid(dataset_train, sargs.num_users, args.label_rate)
#         else:
#             exit('Error: only consider IID setting in svhn')
    elif args.dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset_train = datasets.FashionMNIST('../data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.FashionMNIST('../data/mnist/', train=False, download=True, transform=transform)


    else:
        exit('Error: unrecognized dataset')
        
    print(args.iid)
    if args.iid == 'iid':
        print('iid')
        dict_users, dict_users_label, dict_users_unlabel = iid_sample(dataset_train, args.label_rate,args.num_users)
    elif args.iid == 'noniid1':
        print('noniid1')
        dict_users, dict_users_label, dict_users_unlabel = noniid_sample(dataset_train, args.label_rate,args.num_users)
    elif args.iid == 'noniid2':
        print('noniid2')
        dict_users, dict_users_label, dict_users_unlabel = noniid_ii_sample(dataset_train, args.label_rate,args.num_users)
    elif args.iid == 'noniid3':
        print('noniid3')
        dict_users, dict_users_label, dict_users_unlabel = noniid_iii_sample(dataset_train, args.label_rate,args.num_users)
        
    img_size = dataset_train[0][0].shape
    
    
    Time = time.asctime( time.localtime(time.time()) )
    workbook = xlwt.Workbook(encoding='utf-8')       #新建工作簿
    sheet1 = workbook.add_sheet('Sheet1')          #新建sheet
    sheet1.write(0,0,"Phase")
    sheet1.write(0,1,"round")
    sheet1.write(0,2,"loss")
    sheet1.write(0,3,"Test Acc")


    # '''--------------------------------------------选网络--------------------------------------------------------'''
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'svhn':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist'or'fmnist':
        net_glob = CNNMnist(args=args).to(args.device)
    
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # mask_vector initialization
    # 0: 1-labeled 0:pseudo-label -1-unlabeled 1:user's belonging 2: label/pseudo-label  3: data idx
    mask_vector, label_refer = maskvector_init(dataset_train, args, dict_users, dict_users_label)
    # print('label_vector length:', len(label_vector))
    # print('unlabel_vector length:', len(unlabel_vector))

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
#     Time = time.asctime( time.localtime(time.time()) )
#     workbook = xlwt.Workbook(encoding='utf-8')       #新建工作簿
#     sheet1 = workbook.add_sheet('Sheet1')          #新建sheet
#     sheet1.write(0,0,"round")
#     sheet1.write(0,1,"loss")
#     sheet1.write(0,2,"Test Acc")

    localNet_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}#存储local net所有layer的para
    localNet_last2_dic = {idxs: np.array([], dtype='float') for idxs in range(args.num_users)}#存储后两层layer的para
    m = max(int(args.frac * args.num_users), 1)  # max(0.1*100,1)=10


    # -----------------------------------------initialization-------------------------------------------
    initial_round = int(args.num_users/m)
    for iter in range(initial_round):

        w_fist2_collection, loss_locals = [], []
        w_glob_avg = []
        if args.num_users > 10:
            idxs_users = np.arange(iter*10,(iter+1)*10,1)
        else:
            idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            # first epoch
            # train using only labeled data
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx])  # define
            w, loss = local.init_train(net=copy.deepcopy(net_glob).to(args.device))
            localNet_dic[idx] = w # w :ordereddic ---> localNet_dic[idx]: ordereddic
            # update后两层网络 localNet_last2_dic[idx]:dict

            if args.dataset == 'mnist' or args.dataset == 'fmnist':
                localNet_last2_dic[idx] = {
                    'fc1.weight': w['fc1.weight'], 'fc1.bias': w['fc1.bias'],
                'fc2.weight': w['fc2.weight'], 'fc2.bias': w['fc2.bias']
                                           }
            elif args.dataset == 'svhn' or args.dataset == 'cifar' :
                if args.layer_select == 1:
#                 first
                    localNet_last2_dic[idx] = {
                                          
                                           "conv_layer.0.weight":w["conv_layer.0.weight"],
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
                elif args.layer_select == 2 :
#                 first 2
                     localNet_last2_dic[idx] = {
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
                elif args.layer_select == 3 :
                    localNet_last2_dic[idx] = {
                                           
                                          'fc_layer.1.weight': w['fc_layer.1.weight'], 'fc_layer.1.bias': w['fc_layer.1.bias'],
                                           'fc_layer.3.weight': w['fc_layer.3.weight'], 'fc_layer.3.bias': w['fc_layer.3.bias'],
                                           'fc_layer.6.weight': w['fc_layer.6.weight'], 'fc_layer.6.bias': w['fc_layer.6.bias']}
            

            w_glob_avg.append(copy.deepcopy(w)) # w :ordereddic ---> w_glob_avg: list of ordereddic
            # 前两层网络 w_fisrt2: dict
            if args.dataset == 'mnist' or args.dataset =='fmnist':
                w_first2 = {'conv1.weight': w['conv1.weight'], 'conv1.bias': w['conv1.bias'],
                           'conv2.weight': w['conv2.weight'], 'conv2.bias': w['conv2.bias']}
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                if args.layer_select == 1 :
#                 first
                    w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked'], 
                                'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias']}
                elif args.layer_select == 2 :
#                 first2
                    w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked'], 
                            'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias'],
                                "conv_layer.0.weight":w["conv_layer.0.weight"],
                                           "conv_layer.0.bias":w["conv_layer.0.bias"],
                                           'conv_layer.1.weight': w['conv_layer.1.weight'],'conv_layer.1.bias': w['conv_layer.1.bias'],
                                           'conv_layer.1.running_mean':w[ 'conv_layer.1.running_mean'],
                                           'conv_layer.1.running_var':w['conv_layer.1.running_var'],
                                           'conv_layer.1.num_batches_tracked':w[ 'conv_layer.1.num_batches_tracked'],
                                           'conv_layer.3.weight':w['conv_layer.3.weight'], 'conv_layer.3.bias':w['conv_layer.3.bias'],
                                           'conv_layer.7.weight':w['conv_layer.7.weight'], 'conv_layer.7.bias':w['conv_layer.7.bias']
                               }
                elif args.layer_select == 3 :
#                 first2
                    w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked'], 
                            'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias'],
                               "conv_layer.0.weight":w["conv_layer.0.weight"],
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
                                           'conv_layer.10.bias': w['conv_layer.10.bias']}
                
               
            # 将待average的前两层网络加入w_locals中
            a_ = OrderedDict(w_first2) # w_first2: dict ---> a_:ordereddic
            w_fist2_collection.append(copy.deepcopy(a_)) # a_:ordereddic ---> w_fist2_clct: list of ordereddic
            loss_locals.append(copy.deepcopy(loss))
        # update global weights---first 2 layers' parameters
        w_glob = FedAvg(w_fist2_collection) # a_:ordereddic ---> w_glob : ordereddic

        # update net_glob
        w_glob_avg_dic = FedAvg(w_glob_avg) # w_glob_avg_dic:ordereddic
        net_glob.load_state_dict(w_glob_avg_dic)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Initial: Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        
        #initial recording
        sheet1.write(iter+1,0,'Initialization')
        sheet1.write(iter+1,1,iter)
        sheet1.write(iter+1,2,loss_avg)
        sheet1.write(iter+1,3,float(acc_test))
    
        
        

    test_data = DataLoader(dataset_test, batch_size=1, shuffle=False)
    max_prob = []
    for (img, label) in test_data:
        img, label = img.to(args.device), label.to(args.device).long
        img.cuda()
        log_probs_init = F.softmax(net_glob(img), dim=1)
        pseudo_label = log_probs_init.data.max(1, keepdim=True)[1]
        max_prob.append(log_probs_init.data.max(1, keepdim=True)[0])

    max_prob = torch.tensor(max_prob)
    print('Initial Mean Max_prob:',torch.mean(max_prob))
    #Initialization result
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/phase1_rate{}_{}_{}_p1e{}_p2e{}_C{}_iid{}.png'.format
                (args.label_rate, args.dataset, args.model, args.p1epochs, args.p2epochs, args.frac, args.iid))

    # # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Initial Global Training accuracy: {:.2f}".format(acc_train))
    print("Initial Global Testing accuracy: {:.2f}".format(acc_test))
    net_glob.train()


    #--------------------------------------------pre-train---------------------------------------------------------

    loss_train = []
    for iter in range(args.p1epochs):

        w_fist2_collection, loss_locals = [], []
        w_glob_avg = []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = np.arange(10)
        for idx in idxs_users:
            # first epoch
            # train using only labeled data
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx])  # define
            w, loss = local.init_train(net=copy.deepcopy(net_glob).to(args.device))
            localNet_dic[idx] = w # w :ordereddic ---> localNet_dic[idx]: ordereddic
            # update后两层网络 localNet_last2_dic[idx]:dict
            if args.dataset == 'mnist' or args.dataset =='fmnist':
                localNet_last2_dic[idx] = {
                    'fc1.weight': w['fc1.weight'], 'fc1.bias': w['fc1.bias'],
                    'fc2.weight': w['fc2.weight'], 'fc2.bias': w['fc2.bias']
                }
            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                localNet_last2_dic[idx] = {
                                            'conv2.0.weight':w['conv2.0.weight'], 
                                            'conv2.0.bias':w['conv2.0.bias'],
                                            "conv_layer.0.weight":w["conv_layer.0.weight"],
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

            w_glob_avg.append(copy.deepcopy(w)) # w :ordereddic ---> w_glob_avg: list of ordereddic
            # 前两层网络 w_fisrt2: dict
            if args.dataset == 'mnist' or args.dataset =='fmnist':
                w_first2 = {'conv1.weight': w['conv1.weight'], 'conv1.bias': w['conv1.bias'],
                            'conv2.weight': w['conv2.weight'], 'conv2.bias': w['conv2.bias']}
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked']}
            # 将待average的前两层网络加入w_locals中
            a_ = OrderedDict(w_first2) # w_first2: dict ---> a_:ordereddic
            w_fist2_collection.append(copy.deepcopy(a_)) # a_:ordereddic ---> w_fist2_clct: list of ordereddic
            loss_locals.append(copy.deepcopy(loss))
        # update global weights---first 2 layers' parameters
        w_glob = FedAvg(w_fist2_collection) # a_:ordereddic ---> w_glob : ordereddic

        # update net_glob
        w_glob_avg_dic = FedAvg(w_glob_avg) # w_glob_avg_dic:ordereddic
        net_glob.load_state_dict(w_glob_avg_dic)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Phase1: Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        
        #initial recording
        sheet1.write(iter+1+initial_round,0,'Phase1')
        sheet1.write(iter+1+initial_round,1,iter)
        sheet1.write(iter+1+initial_round,2,loss_avg)
        sheet1.write(iter+1+initial_round,3,float(acc_test))

    test_data = DataLoader(dataset_test, batch_size=1, shuffle=False)
    max_prob = []
    for (img, label) in test_data:
        img, label = img.to(args.device), label.to(args.device).long
        img.cuda()
        log_probs_init = F.softmax(net_glob(img), dim=1)
        pseudo_label = log_probs_init.data.max(1, keepdim=True)[1]
        max_prob.append(log_probs_init.data.max(1, keepdim=True)[0])

    max_prob = torch.tensor(max_prob)
    print('pre-train Mean Max_prob:',torch.mean(max_prob))
    #Initialization result
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/pretrain_rate{}_{}_{}_p1e{}_p2e{}_C{}_iid{}.png'.format
                (args.label_rate, args.dataset, args.model, args.p1epochs, args.p2epochs, args.frac, args.iid))

    # # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Pre train Global Training accuracy: {:.2f}".format(acc_train))
    print("Pre train Global Testing accuracy: {:.2f}".format(acc_test))
    net_glob.train()
   
    net_glob.train()
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
    localifpseudo_dic = {idxs:0 for idxs in range(args.num_users)}#存储local net所有layer的para
            # '''-----------------------------------------Semi-train-------------------------------------------'''
    # train dict:  key-user_idx : set(labeldata_idx + pseudolabel_idx)
    dict_train = dict_users_label
    dict_pseudo = {idxs: set() for idxs in range(args.num_users)}

    for iter in range(int(args.p2epochs)):
        w_fist2_clct, loss_locals, w_glob_avg= [], [], []

        updated_pseudo_num = []
        pseudo_correctRate = []


        m = max(int(args.frac * args.num_users), 1)  # max(0.1*10,1)=10
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if iter < 10:
            threshold = 2.8 * float(torch.mean(max_prob))
        elif iter < 35 and iter >= 10:
            threshold = (100 - iter * 2) / 100 * 2.8 * float(torch.mean(max_prob))
        else:
            if args.tp:
                threshold = 2.8 * float(torch.mean(max_prob)) / 2
            else:
                threshold = (100 - iter * 2) / 100 * 2.8 * float(torch.mean(max_prob))

        print('Round:',iter,'threshold:',threshold)

        for idx in idxs_users:

            updated_pseudo_count = 0

            # net_2tune
            if args.dataset == 'mnist' or args.dataset =='fmnist':
                net_2tune = CNNMnist(args=args).to(args.device)
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                net_2tune = CNNCifar(args=args).to(args.device)

            # 对临时变量_赋值前两层经过fedavg的结果
            _ = dict(w_glob)   # w_glob : ordereddic ---> _ : dic
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
            # Net-new&old
            if args.dataset == 'mnist' or args.dataset =='fmnist':
                net_new = CNNMnist(args=args).to(args.device)
                net_old = CNNMnist(args=args).to(args.device)
            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                net_new = CNNCifar(args=args).to(args.device)
                net_old = CNNCifar(args=args).to(args.device)
                
            if args.finetune:
                local_tune = LocalUpdate(args=args, maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx])
                w_tuned, loss_tuned = local_tune.fine_tune(net=copy.deepcopy(net_2tune).to(args.device))

                # 拼接所得新网络
            
                net_new.load_state_dict(w_tuned)
                old_para = localNet_dic[idx]
                net_old.load_state_dict(old_para)
            else:
#             no fine-tune
            # Net-new
            # 对临时变量_赋值前两层经过fedavg的结果
                _ = dict(w_glob)  # w_glob : ordereddic ---> _ : dic
            # 拼接后两层参数
                _.update(localNet_last2_dic[idx])
            # 转换为ordereddic类型
                new_para = OrderedDict(_)  # _ : dic ---> new_para : ordereddic
            # model = nn.DataParallel(net_new)
                net_new.load_state_dict(new_para)  # 载入新网络参数

#             net_glob.eval()
#             acc_test, loss_test = test_img(net_glob, dataset_test, args)
#             print("Round:{:3d}, user:{:3d},Glob Testing accuracy: {:.2f}".format(iter,idx, acc_test))
#             net_glob.train()

#             net_old.eval()
#             old_test, old_loss_test = test_img(net_old, dataset_test, args)
#             print("\t\t\tnet_old Testing accuracy: {:.2f}".format(old_test))
#             net_old.train()

#             net_new.eval()
#             new_test, new_loss_test = test_img(net_new, dataset_test, args)
#             print("\t\t\tnet_new Testing accuracy: {:.2f}".format(new_test))
#             net_new.train()
            
            # '''*********************************Relabel**********************************************************'''
            # Begin Pseduo Label
#             Accs = [old_test,new_test,acc_test]
            Accs = [1,1,1]
            try_data = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=1, shuffle=False)
            total_pseudo_num = 0
            label_correct_count = 0
            # throughout all the data
            for i, (img, label) in zip(dict_users[idx],try_data):
                # if data is unlabeled
                if mask_vector[i][0] == -1:
                    img, label = img.to(args.device), label.to(args.device).long
                    log_probs1 = F.softmax(net_old(img), dim=1)
                    log_probs2 = F.softmax(net_new(img), dim=1)
                    log_probs3 = F.softmax(net_glob(img), dim=1)
                    
                    if args.pseudo_select:
                        drop_sign = Accs.index(min(Accs))
                        if drop_sign == 0:
                            log_probs = log_probs2 + log_probs3
                        elif drop_sign == 1:
                            log_probs = log_probs1 + log_probs3
                        elif drop_sign == 2:
                            log_probs = log_probs2 + log_probs3
                    else:
                        log_probs = log_probs1 + log_probs2 + log_probs3
                    pseudo_label = log_probs.data.max(1, keepdim=True)[1]
                    max_prob_pseudo = log_probs.data.max(1, keepdim=True)[0]


                    if float(max_prob_pseudo) >= threshold:
                        mask_vector[i][2] = int(pseudo_label)
                        mask_vector[i][0] = 0 # twice relabeling is not allowed

                        userid = int(idx)
                        dict_train[userid].add(i)
                        dict_pseudo[userid].add(i)

                        total_pseudo_num += 1
                        updated_pseudo_count+=1
                        if pseudo_label == label_refer[i]:
                            label_correct_count += 1
                    # Majority voting

#                     y_pred1 = log_probs1.data.max(1, keepdim=True)[1] # tensor: [[label]]
#                     y_pred2 = log_probs2.data.max(1, keepdim=True)[1]
#                     y_pred3 = log_probs3.data.max(1, keepdim=True)[1]
#                     pseduo_label1 = y_pred1.cpu().detach().numpy() # numpy.int32
#                     pseduo_label2 = y_pred2.cpu().detach().numpy()
#                     pseduo_label3 = y_pred3.cpu().detach().numpy()
                    
#                     voting_list = [int(pseduo_label1),int(pseduo_label2), int(pseduo_label3)]
                    
#                     counts = np.bincount(voting_list)
#                     # relabel
#                     if counts.max() > 1:
#                         # return the mode
#                         mask_vector[i][2] = np.argmax(counts)
#                         mask_vector[i][0] = 0
#                         userid = int(idx)
#                         dict_train[userid].add(i)
#                         dict_pseudo[userid].add(i)

#                         total_pseudo_num += 1
#                         updated_pseudo_count+=1
#                         if pseudo_label == label_refer[i]:
#                             label_correct_count += 1
#                     else:continue
                        
                    total_pseudo_num = max(total_pseudo_num,1)
                    if total_pseudo_num == 0 :
                        continue
                    else:
                        pseudo_correctRate.append(label_correct_count/total_pseudo_num)
            if total_pseudo_num >= 10:
                localifpseudo_dic[idx]+=1
            updated_pseudo_num.append(updated_pseudo_count)

            # *********************************local update**********************************************************
            # local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx])
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx],idxs2=dict_pseudo[idx])

            if localifpseudo_dic[idx]<1:
                w, loss = local.phase2_train(net=copy.deepcopy(net_new).to(args.device))
            else:
                w, loss = local.semitrain_Ratio(net=copy.deepcopy(net_new).to(args.device), sever_round=iter)
#                 w, loss = local.phase2_train(net=copy.deepcopy(net_new).to(args.device))
#                 w, loss = local.semitrain(net=copy.deepcopy(net_new).to(args.device), sever_round=iter, args=args)
            
            
# =============================TEST============================================            

#             if args.dataset == 'mnist' or args.dataset =='fmnist':
#                 net_t = CNNMnist(args=args).to(args.device)

#             elif args.dataset == 'svhn' or args.dataset == 'cifar':
#                 net_t = CNNCifar(args=args).to(args.device)

#             net_t.load_state_dict(w)

#             net_t.eval()
#             t_test, loss_test = test_img(net_t, dataset_test, args)
#             print("T Testing accuracy: {:.2f}".format(t_test))
#             if args.compare_key == True:
#                 if t_test <= new_test:
#                     print('w tuned reserved')
#                     w = w_tuned
#             del net_t
# =============================TEST============================================            

            
            
            loss_locals.append(copy.deepcopy(loss))
            w_glob_avg.append(copy.deepcopy(w))
            # 更新local net参数
            localNet_dic[idx] = w
            # 前两层网络
            if args.dataset == 'mnist'or args.dataset =='fmnist' :
                w_first2 = {'conv1.weight': w['conv1.weight'], 'conv1.bias': w['conv1.bias'],
                           'conv2.weight': w['conv2.weight'], 'conv2.bias': w['conv2.bias']}
            elif args.dataset == 'svhn'or args.dataset == 'cifar':
                w_first2 = {'conv1.0.weight':w['conv1.0.weight'],'conv1.0.bias':w['conv1.0.bias'],
                            'conv1.1.weight':w['conv1.1.weight'], 'conv1.1.bias':w['conv1.1.bias'],
                            'conv1.1.running_mean':w['conv1.1.running_mean'], 'conv1.1.running_var':w['conv1.1.running_var'],
                            'conv1.1.num_batches_tracked':w['conv1.1.num_batches_tracked']}
            # 将待average的前两层网络加入w_locals中
            a_ = OrderedDict(w_first2)  # w_first2: dict ---> a_:ordereddic
            w_fist2_clct.append(copy.deepcopy(a_))  # a_:ordereddic ---> w_fist2_clct: list of ordereddic
            # loss_locals.append(copy.deepcopy(loss))
            # update后两层网络
            if args.dataset == 'mnist'or args.dataset =='fmnist':
                localNet_last2_dic[idx] = {
                    'fc1.weight': w['fc1.weight'], 'fc1.bias': w['fc1.bias'],
                    'fc2.weight': w['fc2.weight'], 'fc2.bias': w['fc2.bias']
                }

            elif args.dataset == 'svhn' or args.dataset == 'cifar':
                localNet_last2_dic[idx] = {
                           'conv2.0.weight':w['conv2.0.weight'], 'conv2.0.bias':w['conv2.0.bias'],
                                           "conv_layer.0.weight":w["conv_layer.0.weight"],
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
        w_glob = FedAvg(w_fist2_clct)
        # update net_glob
        w_glob_avg_dic = FedAvg(w_glob_avg)
        net_glob.load_state_dict(w_glob_avg_dic)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Phase2: Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        pseudo_correctRate = torch.tensor(pseudo_correctRate)
        print('Relabel accuracy:', torch.mean(pseudo_correctRate))

        updated_pseudo_num = torch.tensor(updated_pseudo_num).float()
        print('Mean updated data num:', torch.mean(updated_pseudo_num))
        
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        
        #initial recording
        sheet1.write(iter+1+initial_round+int(args.p1epochs),0,'Phase2')
        sheet1.write(iter+1+initial_round+int(args.p1epochs),1,iter)
        sheet1.write(iter+1+initial_round+int(args.p1epochs),2,loss_avg)
        sheet1.write(iter+1+initial_round+int(args.p1epochs),3,float(acc_test))
    

    # result
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/phase2_rate{}_{}_{}_p1e{}_p2e{}_C{}_iid{}.png'.format
                (args.label_rate, args.dataset, args.model, args.p1epochs, args.p2epochs, args.frac, args.iid))

    # # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Phase2 Global Training accuracy: {:.2f}".format(acc_train))
    print("Phase2 Global Testing accuracy: {:.2f}".format(acc_test))


    # testing
    acc_train_list, loss_train_list = [], []
    acc_test_list, loss_test_list = [], []
    for i in range(args.num_users):
        if args.dataset == 'mnist' :
            net = CNNMnist(args=args).to(args.device)
        elif args.dataset == 'svhn':
            net = CNNCifar(args=args).to(args.device)
        elif args.dataset == 'cifar':
            net = CNNCifar(args=args).to(args.device)
        elif args.dataset == 'fmnist' :
            net = CNNMnist(args=args).to(args.device)
                
        net.load_state_dict(localNet_dic[i])
        net.eval()
        acc_train, loss_train = test_img_client(net_glob, net, dataset_train, args, dict_users[i])
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        acc_test, loss_test = test_img_client2(net_glob,net, dataset_test, args)
        acc_test_list.append(acc_test)
        loss_test_list.append(loss_test)
    acc_train_avg = np.max(acc_train_list)
    acc_test_avg = np.max(acc_test_list)
    print("Final User's Training accuracy: {:.2f}".format(acc_train_avg))
    print("Final User's Testing accuracy: {:.2f}".format(acc_test_avg))
    
    sheet1.write(iter+2+initial_round+int(args.p1epochs),0,'Best Glob Acc')
    sheet1.write(iter+3+initial_round+int(args.p1epochs),0,'Best Local Acc')
    sheet1.write(iter+2+initial_round+int(args.p1epochs),3,float(acc_test))
    sheet1.write(iter+3+initial_round+int(args.p1epochs),3,float(acc_test_avg))
    sheet1.write(0,6,'usernum_{}_labelrate_{}_p1epochs_{}_p2epochs_{}dataset_{}_frac_{}'\
                 .format(args.num_users,args.label_rate,args.p1epochs,args.p2epochs,args.dataset,args.frac))
    workbook.save(r'./fedtrinet_{}_{}_{}_{}.xls'.format(args.dataset,args.iid,args.label_rate,Time))
