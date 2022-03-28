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
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img, test_img_client
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



def read_data_path(file_name):
    img_list = []
    label_list = []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            img = line[0]
            label = int(line[1])

            img_list.append(img)
            label_list.append(label)
        # end_for
    # end_with

    # print('the number of sample: ', len(img_list))
    # print(len(label_list));
    # print(img_list[0], label_list[0]);

    # print('Trainset updated');

    return img_list, label_list


class MnistDataset(Dataset):
    """Mnist dataset."""

    def __init__(self, list_file, root_dir, transform=None):
        """
        Args:
            list_file (string): labeled list or unlabeled list
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list, self.label_list = read_data_path(list_file)
        self.root_dir = root_dir
        self.transform = transform

    # end_func

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.img_list[idx])
        image = io.imread(img_name)
        image = Image.fromarray(image, mode='L')  # image is a 'Image' type
        label = torch.LongTensor(1)
        label = self.label_list[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            image = self.transform(image)

        return image, label
    # end_func
# end_class


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # '''----------------------------------------------选数据集--------------------------------------------'''

    # load dataset and split users

    # dataset[0]: set of (img, label)  dataset[i][0]: No.i img  dataset[i][1]: No.i label
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users, dict_users_label = mnist_iid(dataset_train, args.num_users,args.label_rate)
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
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'svhn':
        net_glob = CNNCifar(args=args).to(args.device)
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

    m = max(int(args.frac * args.num_users), 1)  # max(0.1*100,1)=10



    #--------------------------------------------pre-train---------------------------------------------------------

    loss_train = []
    for iter in range(args.p1epochs):

        w_globc, loss_locals = [], []
   
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = np.arange(10)
        for idx in idxs_users:
            # first epoch
            # train using only labeled data
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_users_label[idx])  # define
            w, loss = local.init_train(net=copy.deepcopy(net_glob).to(args.device))
            
            
            w_globc.append(w) # w :ordereddic ---> w_glob_avg: list of ordereddic

            
            loss_locals.append(copy.deepcopy(loss))


        # update net_glob
        w_glob_avg = FedAvg(w_globc) # w_glob_avg_dic:ordereddic
        net_glob.load_state_dict(w_glob_avg)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Phase1: Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

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


    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
            # '''-----------------------------------------Semi-train-------------------------------------------'''
    # train dict:  key-user_idx : set(labeldata_idx + pseudolabel_idx)
    dict_train = dict_users_label
    dict_pseudo = {idxs: set() for idxs in range(args.num_users)}

    for iter in range(int(args.p2epochs)):
        w_globc, loss_locals= [], []

        updated_pseudo_num = []
        pseudo_correctRate = []


        m = max(int(args.frac * args.num_users), 1)  # max(0.1*10,1)=10
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if iter < 10:
            threshold = 1.1 * float(torch.mean(max_prob))
        elif iter < 35:
            threshold = (100 - iter * 2) / 100 * 1.1 * float(torch.mean(max_prob))
        else:threshold = 1.1 * float(torch.mean(max_prob)) / 2



        for idx in idxs_users:

            updated_pseudo_count = 0

            
            # '''*********************************Relabel**********************************************************'''
            # Begin Pseduo Label

            try_data = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=1, shuffle=False)
            total_pseudo_num = 0
            label_correct_count = 0
            # throughout all the data
            for i, (img, label) in zip(dict_users[idx],try_data):
                # if data is unlabeled
                if mask_vector[i][0] == -1:
                    img, label = img.to(args.device), label.to(args.device).long
                    
                    log_probs = F.softmax(net_glob(img), dim=1)

                    pseudo_label = log_probs.data.max(1, keepdim=True)[1]
                    max_prob = log_probs.data.max(1, keepdim=True)[0]


                    if float(max_prob) >= threshold:
                        mask_vector[i][2] = int(pseudo_label)
                        mask_vector[i][0] = 0 # twice relabeling is not allowed

                        userid = int(idx)
                        dict_train[userid].add(i)
                        dict_pseudo[userid].add(i)
                        
                        total_pseudo_num +=1
                        updated_pseudo_count+=1
                        if pseudo_label == label_refer[i]:
                            label_correct_count+=1
                   
                    total_pseudo_num = max(total_pseudo_num,1)
                    if total_pseudo_num == 0 :
                        pseudo_correctRate.append(np.mean(pseudo_correctRate))
                    else:
                        pseudo_correctRate.append(label_correct_count/total_pseudo_num)

            updated_pseudo_num.append(updated_pseudo_count)

            # *********************************local update**********************************************************
            # local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx])
            local = LocalUpdate(args=args,maskv=mask_vector, dataset=dataset_train, idxs=dict_train[idx],idxs2=dict_pseudo[idx])

            
            if iter < 10:
                w, loss = local.semitrain(net=copy.deepcopy(net_glob).to(args.device), sever_round=iter)
            else:
                w, loss = local.semitrain_Ratio(net=copy.deepcopy(net_glob).to(args.device), sever_round=iter)
            
            loss_locals.append(copy.deepcopy(loss))
            w_globc.append(copy.deepcopy(w))
            
        # update net_glob
        w_glob_avg = FedAvg(w_globc)
        net_glob.load_state_dict(w_glob_avg)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Phase2: Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        pseudo_correctRate = torch.tensor(pseudo_correctRate)
        print('Relabel accuracy:', torch.mean(pseudo_correctRate))

        updated_pseudo_num = torch.tensor(updated_pseudo_num).float()
        print('Mean updated data num:', torch.mean(updated_pseudo_num))


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
    # acc_train_list, loss_train_list = [], []
    # acc_test_list, loss_test_list = [], []
    # for i in range(args.num_users):
    #     net = net_glob
    #     net.load_state_dict(localNet_dic[i])
    #     net.eval()
    #     acc_train, loss_train = test_img_client(net, trainset, args, dict_users[i])
    #     acc_train_list.append(acc_train)
    #     loss_train_list.append(loss_train)
    #     acc_test, loss_test = test_img(net, testset, args)
    #     acc_test_list.append(acc_test)
    #     loss_test_list.append(loss_test)
    # acc_train_avg = np.mean(acc_train_list)
    # acc_test_avg = np.mean(acc_test_list)
    # print("Final Training accuracy: {:.2f}".format(acc_train_avg))
    # print("Final Testing accuracy: {:.2f}".format(acc_test_avg))
