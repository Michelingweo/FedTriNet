#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from PIL import Image
import xlwt,xlrd
import time
from utils.sampling import iid_sample,noniid_sample,noniid_ii_sample,noniid_iii_sample
from utils.options import args_parser
from models.Update import LocalUpdate,DatasetSplit_fedsem,DatasetSplit
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

from list_txt.make_list import maskvector_init

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

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
    print('epoch1:',args.p1epochs)
    print('epoch2:',args.p2epochs)
    print('dataset:',args.dataset)
    print('frac:',args.frac)
    
    if args.iid == 'noniid':
        torch.cuda.set_device(0)
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    if args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
#         dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transform)
#         dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transform)
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':

        trans_cifar = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trans_cifar_test = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar_test)

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
#     else:
#         exit('Error: unrecognized dataset')

    dataset_valid = DatasetSplit_fedsem(dataset_test, [i for i in range(1000)], None)

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
    # build model
    if args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'mnist'or args.dataset =='fmnist':
        net_glob = CNNMnist(args=args).to(args.device)
#         net_glob = CDiscriminator(args=args).to(args.device)
#         net_glob.weight_init()
    elif args.dataset == 'svhn':
        net_glob = CNNCifar(args=args).to(args.device)
        
    else:
        exit('Error: unrecognized model')
        
        
    mask_vector, label_refer = maskvector_init(dataset_train, args, dict_users, dict_users_label)
        
    print(net_glob)
    Time = time.asctime( time.localtime(time.time()) )
    workbook = xlwt.Workbook(encoding='utf-8')       #新建工作簿
    sheet1 = workbook.add_sheet('Sheet1')          #新建sheet
    sheet1.write(0,0,"Round")
    sheet1.write(0,1,"Loss")
    sheet1.write(0,2,"Acc")

    #'''-----------------------------------------global net train----------------------------------------------'''
    print("\n Begin Phase 1")

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    w_best = copy.deepcopy(w_glob)
    best_loss_valid = 100000000000

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.p1epochs):

        net_glob.train()

        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)#choice trained users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
#             local = LocalUpdate_fedsemp1(args=args, dataset=dataset_train, idxs=dict_users[idx], idxs_labeled=dict_users_labeled[idx], pseudo_label=None)
#             w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx])  # define
            w, loss = local.init_train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w)) 
            loss_locals.append(copy.deepcopy(loss))
            
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        acc_valid, loss_valid = test_img(net_glob, dataset_valid, args)
        if loss_valid <= best_loss_valid:
            best_loss_valid = loss_valid
            w_best = copy.deepcopy(w_glob)
        
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Phase 1: Round {:3d}, Average loss {:.3f} Testing accuracy: {:.2f}'.format(iter, loss_avg,acc_test))
        loss_train.append(loss_avg)
        sheet1.write(iter+1,0,iter)
        sheet1.write(iter+1,1,loss_avg)
        sheet1.write(iter+1,2,float(acc_test))
#'''**********************************改标签**********************************************************'''  
    print('\n Begin Pseduo Label')
    for idx in range(args.num_users):
        try_data = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=1, shuffle=False)
        for i, (img, label) in zip(dict_users[idx],try_data):
#             if i %10000==0:
#                 print("Pseduo Label: {:3d}".format(i))
            if mask_vector[i][0] == -1:
                mask_vector[i][0] = 0 # twice relabeling is not allowed
                img, label = img.to(args.device), label.to(args.device)
                log_probs = net_glob(img)        
                y_pred = log_probs.data.max(1, keepdim=True)[1]    
                mask_vector[i][2] = int(y_pred.cpu().detach())
    print('\n Finished Pseduo Label')

#'''**********************************mid test**********************************************************''' 
#     print(len(pseudo_label))
#     print(type(pseudo_label))
#     if -1 in pseudo_label:
#         print('have -1')
#     elif 10 in pseudo_label:
#         print('have10')
    
#     print('max',pseudo_label[max(pseudo_label)])
#     print('min',pseudo_label[min(pseudo_label)])
#     pseudo_label.sort(reverse = True)
#     print(pseudo_label[:10])
    
#     print("\n Begin mid test")

#     # testing
#     net_glob.eval()


#     users_labeled=set()
#     for i in range(len(dict_users_labeled)) :
#         users_labeled = users_labeled | dict_users_labeled[i]

#     users_unlabeled=set()
#     for i in range(len(dict_users_labeled)) :
#         users_unlabeled = users_unlabeled | (dict_users[i] - dict_users_labeled[i])

#     # dataset_train_labeled = DatasetSplit_fedsem(dataset_train, users_labeled, pseudo_label)
#     # dataset_train_unlabeled = DatasetSplit_fedsem(dataset_train, users_unlabeled, pseudo_label)
#     #
#     # acc_train_labeled, loss_train_test_labeled = test_img(net_glob, dataset_train_labeled, args)
#     # print("Phase 1 labeled Training accuracy: {:.2f}".format(acc_train_labeled))
#     #
#     # if args.label_rate != 1.0 :
#     #     acc_train_unlabeled, loss_train_test_unlabled = test_img(net_glob, dataset_train_unlabeled, args)
#     #     print("Phase 1 unlabeled Training accuracy: {:.2f}".format(acc_train_unlabeled))

#     acc_test, loss_test = test_img(net_glob, dataset_test, args)
#     print("Phase 1 Testing accuracy: {:.2f} \n\n".format(acc_test))
    
#******************************************Phase 2********************************************************
    
    print("\n Begin Phase 2")

    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
#    for iter in range(1):
    for iter in range(args.p2epochs):

        net_glob.train()

        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)#choice trained users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
#             local = LocalUpdate_fedavg(args=args, dataset=dataset_train, idxs=dict_users[idx],pseudo_label=pseudo_label)#define
#             w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users[idx])
#             local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx],idxs2=dict_pseudo[idx])
            w, loss = local.init_train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        acc_valid, loss_valid = test_img(net_glob, dataset_valid, args)
        if loss_valid <= best_loss_valid:
            best_loss_valid = loss_valid
            w_best = copy.deepcopy(w_glob)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Phase 2: Round {:3d}, Average loss {:.3f} Testing accuracy: {:.2f}'.format(iter, loss_avg,acc_test))
        loss_train.append(loss_avg)
    
        sheet1.write(args.p1epochs+iter+1,0,iter)
        sheet1.write(args.p1epochs+iter+1,1,loss_avg)
        sheet1.write(args.p1epochs+iter+1,2,float(acc_test))
    
    
#*****************************************final test********************************************************

    print("\n Begin final test")

    net_glob.load_state_dict(w_best)
    
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Phase 2 Testing accuracy: {:.2f}% \n\n".format(acc_test))
#     dataset_train_labeled = DatasetSplit_fedsem(dataset_train, users_labeled, pseudo_label)
#     dataset_train_unlabeled = DatasetSplit_fedsem(dataset_train, users_unlabeled, pseudo_label)
    
    sheet1.write(args.p1epochs+args.p2epochs+1,2,float(acc_test))
    
#     acc_train_labeled, loss_train_test_labeled = test_img(net_glob, dataset_train_labeled, args)
#     print("Phase 2 labeled Training accuracy: {:.2f}".format(acc_train_labeled))    

#     if args.label_rate != 1.0 :
#         acc_train_unlabeled, loss_train_test_unlabled = test_img(net_glob, dataset_train_unlabeled, args)
#         print("Phase 2 unlabeled Training accuracy: {:.2f}".format(acc_train_unlabeled))

    sheet1.write(0,5,'user_num_{}label_rate_{}epochs_{}dataset_{}frac_{}'\
                 .format(args.num_users,args.label_rate,args.epochs,args.dataset,args.frac))
    workbook.save(r'./fedsem_{}_{}_{}_{}.xlsx'.format(args.dataset,args.iid,args.label_rate,Time))

#    dataset_train_fix = DatasetSplit(dataset_train, np.arange(len(dataset_train)), pseudo_label)
#    acc_train, loss_train_test = test_img(net_glob, dataset_train_fix, args)
#    acc_test, loss_test = test_img(net_glob, dataset_test, args)
#    print("Phase 2 Training accuracy: {:.2f}".format(acc_train))
#    print("Phase 2 Testing accuracy: {:.2f}".format(acc_test))    
    
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fedsem_label{}_{}_p{}+{}_C{}_iid{}.png'.format
                (args.label_rate, args.dataset, args.p1epochs, args.p2epochs, args.frac, args.iid))

    # testing


