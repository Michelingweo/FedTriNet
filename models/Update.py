#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.functional as F
from sklearn import metrics


class DatasetSplit_fedsem(Dataset):#dataset relative
    def __init__(self, dataset, idxs, pseudo_label = None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.pseudo_label=pseudo_label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[self.idxs[item]]
        if self.pseudo_label != None:
            label = int(self.pseudo_label[self.idxs[item]]) 
        return image, label

def read_data_path(file_name):
    img_list = []
    label_list = []
    unlabeled_list = {}
    with open(file_name) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            img = line[0]
            label = int(line[1])

            img_list.append(img)
            label_list.append(label)
        # end_for
        # end_with
    for i in range(len(label_list)):
        if label_list[i] == '-1':
            unlabeled_list.setdefault(i,img_list[i])
    '''
    index-i dir-img_list[i]   label-label_list[i]
    0       training/0/1.jpg  0
    1
    2
    unlabeled_list[n] = ('index','dir') = (index, img_list[i])
    '''
    print('the number of sample: ', len(img_list))
    # print(len(label_list));
    # print(img_list[0], label_list[0]);

    print('Done.');

    return img_list, label_list, unlabeled_list

def get_train_idxs(idxs, idxs_labeled, args, with_unlabel_if):

    idxs_unlabeled = idxs - idxs_labeled
    if args.data_argument == 'True' and args.iid != 'noniid2':
        if args.label_rate == 0.001 or args.label_rate == 0.0137:
            idxs_train = list(idxs_labeled) + list(idxs_labeled) + list(idxs_labeled)
        elif args.label_rate == 0.01:
            idxs_train = list(idxs_labeled) + list(idxs_labeled)
        elif args.label_rate == 0.05 or args.label_rate == 0.0137*3:
            idxs_train = list(idxs_labeled) + list(idxs_labeled)
        elif args.label_rate == 0.1 or args.label_rate == 0.0137*6:
            idxs_train = list(idxs_labeled)
        else: 
            print("error")
        print('11111')

    elif args.data_argument == 'True' and args.iid == 'noniid2' and len(list(idxs_labeled))/len(list(idxs)) < 0.4:
        idxs_train = list(idxs_labeled) + list(idxs_labeled) + list(idxs_labeled)+ list(idxs_labeled) + list(idxs_labeled) + list(idxs_labeled)
        print('2222222')
    else:
        idxs_train = list(idxs_labeled)
        print('33333')
            
    if with_unlabel_if == 'with_unlabel':
        idxs_train = idxs_train + list(idxs_unlabeled)
    idxs_train = np.random.permutation(idxs_train)

    return idxs_train


class DatasetSplit_labelid(Dataset):
    def __init__(self, dataset, idxs, mask_vector):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.mv = mask_vector
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image = self.dataset[self.idxs[item]][0]
        label = int(self.mv[self.idxs[item]][2])
        l_or_un = self.mv[self.idxs[item]][0]
        return image, (label, l_or_un)

class DatasetSplit_mask(Dataset):
    def __init__(self, dataset, idxs, mask_vector):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.mv = mask_vector
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = int(self.mv[self.idxs[item]][2])

        return image, label

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class DatasetSplit_match(Dataset):
    def __init__(self, dataset, idxs, dataset_ema = None, pseudo_label = None):
        self.dataset = dataset
        self.dataset_ema = dataset_ema
        self.idxs = list(idxs)
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[self.idxs[item]]

        if self.pseudo_label != None:
            label = int(self.pseudo_label[self.idxs[item]]) 

        if self.dataset_ema != None:
            image_ema = self.dataset_ema[self.idxs[item]][0]
            return (image, image_ema), label
        else: 
            return image, label

class LocalUpdate(object):
    def __init__(self, args, maskv, dataset=None, idxs=None, idxs2=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.loss_semifunc = nn.CrossEntropyLoss(ignore_index=-1,reduction='none').cuda()
        self.selected_clients = []
        self.mask = maskv
        self.ds = dataset
        self.id2 = idxs2
        self.ldr_masktrain = DataLoader(DatasetSplit_mask(dataset, idxs, mask_vector=maskv), batch_size=self.args.local_bs, shuffle=True)
        
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_semitrain = DataLoader(DatasetSplit_labelid(dataset, idxs, mask_vector=maskv), batch_size=self.args.local_bs, shuffle=True)

    def semitrain(self, net, server_round,args):
        T1 = 50
        T2 = 150

        af = 0.5

        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=False)
        criterian = self.loss_func
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (image, (labels, LorUn)) in enumerate(self.ldr_semitrain):
                image, label = image.to(self.args.device), labels.to(self.args.device).long()
                LorUn = LorUn.to(self.args.device)
                net.zero_grad()
                # labeled forward pass
                optimizer.zero_grad()
                output = net(image)  # labeled data 预测
                output = output.to(self.args.device)


                labeled_num = torch.sum(LorUn.eq(1).float())
                labeled_num = labeled_num.to(self.args.device)
                print('labeled_num:',labeled_num)
                pseudo_labeled_num = torch.sum(LorUn.eq(0).float())
                pseudo_labeled_num = pseudo_labeled_num.to(self.args.device)
                print('pseudo_labeled_num:',pseudo_labeled_num)
                
                # unlabeled lossfunc weight
                Iteration = int(iter+5*(server_round))
                if Iteration < T1:
                    alpha = 0.
                elif Iteration > T2:
                    alpha = af
                else:
                    alpha = (Iteration - T1) / (T2 - T1) * af
                # end_if
                
                
                
                
                # labeled forward pass
                if labeled_num == 0:
                    loss1 = 0.
                else:
                    loss1 = torch.sum(LorUn.eq(1).float() * self.loss_semifunc(output, label)) / (labeled_num)
#                 loss1 = loss1.to(self.args.device)
                # Pseudo labeled forward pass
                if pseudo_labeled_num == 0:
                    loss2 = 0.
                else:
                    loss2 = torch.sum(LorUn.eq(0).float() * self.loss_semifunc(output, label)) / (pseudo_labeled_num)
#                 loss2 = loss2.to(self.args.device)
#                 print('loss1:', loss1)
#                 print('self.loss_semifunc(output, label):',self.loss_semifunc(output, label))
#                 print('LorUn.eq(1).float() * self.loss_semifunc(output, label):',LorUn.eq(1).float() * self.loss_semifunc(output, label))
#                 print('loss2:', loss2)
#                 print('self.loss_semifunc(output, label):',self.loss_semifunc(output, label))
#                 print('LorUn.eq(1).float() * self.loss_semifunc(output, label):',LorUn.eq(0).float() * self.loss_semifunc(output, pseudo_label))
                loss = loss1 + alpha * loss2
                loss = loss.to(self.args.device)
#                 print('loss:',loss)
                
#                 if self.args.swa:
#                     if Iteration > self.args.swa_start and Iteration%self.args.swa_freq == 0 :
#                         optimizer.swap_swa_sgd()
#                         if pseudo_labeled_num > 0:
#                             optimizer.bn_update(self.ldr_semitrain,net, torch.device("cuda"))
#                         else:
#                             optimizer.bn_update(self.ldr_train, net, torch.device("cuda"))
#                         optimizer.swap_swa_sgd()
                
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(image), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def semitrain_Ratio(self, net, sever_round):
        T1 = 50
        T2 = 150

        af = 0.15
        self.ldr_masktrain2 = DataLoader(DatasetSplit_mask(self.ds, self.id2, mask_vector=self.mask),batch_size=self.args.local_bs, shuffle=True)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, ((image1, labels1),(image2,labels2)) in enumerate(zip(self.ldr_masktrain,self.ldr_masktrain2)):

                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()
                image_u, label_p = image2.to(self.args.device), labels2.to(self.args.device).long()

                net.zero_grad()
                # labeled forward pass
                optimizer.zero_grad()

                output1 = net(image_l)  # labeled data 预测
                output2 = net(image_u)  # unlabeled data 预测
                output1 = output1.to(self.args.device)
                output2 = output2.to(self.args.device)
                # unlabeled forward pass

                loss1 = self.loss_func(output1, label_l)
                loss1 = loss1.to(self.args.device)

                # unlabeled lossfunc weight
                Iteration = int(iter + 5 * (sever_round))
                if Iteration < T1:
                    alpha = 0.
                elif Iteration > T2:
                    alpha = af
                else:
                    alpha = (Iteration - T1) / (T2 - T1) * af
                # end_if
                
#                 if self.args.swa:
#                     if Iteration > self.args.swa_start and Iteration%self.args.swa_freq == 0 :
#                         optimizer.swap_swa_sgd()
#                         if pseudo_labeled_num > 0:
#                             optimizer.bn_update(self.ldr_semitrain,net, torch.device("cuda"))
#                         else:
#                             optimizer.bn_update(self.ldr_train, net, torch.device("cuda"))
#                         optimizer.swap_swa_sgd()
                        
                loss2 = self.loss_func(output2, label_p)
                loss2 = loss2.to(self.args.device)

                loss = (loss1 + alpha * loss2)/2
                loss = loss.to(self.args.device)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(image1), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def init_train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_masktrain):
                 #                print(batch_idx)
                img, label = img.to(self.args.device), label.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(img)
                loss = self.loss_func(log_probs, label)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(img), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def phase1_train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):
                 #                print(batch_idx)
                img, label = img.to(self.args.device), label.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(img)
                loss = self.loss_func(log_probs, label)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(img), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def phase2_train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_masktrain):
                 #                print(batch_idx)
                img, label = img.to(self.args.device), label.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(img)
                loss = self.loss_func(log_probs, label)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(img), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def fine_tune(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):
                 # print(batch_idx)
                img, label = img.to(self.args.device), label.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(img)
                loss = self.loss_func(log_probs, label)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(img), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    
class LocalUpdate_fedmatch(object):#local
    def __init__(self, args, dataset_strong=None, dataset_weak=None, pseudo_label=None, idxs=set(), idxs_labeled=set()):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.selected_clients = []
        self.pseudo_label=pseudo_label

        idxs_train = get_train_idxs(idxs, idxs_labeled, args, 'with_unlabel')
        self.ldr_train = DataLoader(
            DatasetSplit_match(dataset = dataset_weak, dataset_ema = dataset_strong, idxs = idxs_train, pseudo_label = pseudo_label),
            batch_size=self.args.local_bs
            )

    def train(self, net, net_helper_1, net_helper_2):

        net.train()
        net_helper_1.train()
        net_helper_2.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=False)
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index= -1 ) 
        epoch_loss = []

        for iter in range(self.args.local_ep):

            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_train):   
                img, img_ema, label = img[0].to(self.args.device), img[1].to(self.args.device), label.to(self.args.device)
                input_var = torch.autograd.Variable(img)
                ema_input_var = torch.autograd.Variable(img_ema)
                target_var = torch.autograd.Variable(label)
                minibatch_size = len(target_var)
                labeled_minibatch_size = target_var.data.ne(-1).sum()    
                ema_model_out = net(ema_input_var)
                model_out = net(input_var)
                model_out_helper_1 = net_helper_1(input_var)
                model_out_helper_2 = net_helper_2(input_var)
                if isinstance(model_out, Variable):
                    logit1 = model_out
                    ema_logit = ema_model_out
                else:
                    assert len(model_out) == 2
                    assert len(ema_model_out) == 2
                    logit1, logit2 = model_out
                    ema_logit, _ = ema_model_out     
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
                class_logit, cons_logit = logit1, logit1
                class_loss = class_criterion(class_logit, target_var) / minibatch_size
                pseudo_label1 = torch.softmax(model_out.detach_(), dim=-1)
                pseudo_label2 = torch.softmax(model_out_helper_1.detach_(), dim=-1)
                pseudo_label3 = torch.softmax(model_out_helper_2.detach_(), dim=-1)
                max_probs1, targets_u1 = torch.max(pseudo_label1, dim=-1)
                max_probs2, targets_u2 = torch.max(pseudo_label2, dim=-1)
                max_probs3, targets_u3 = torch.max(pseudo_label3, dim=-1)
                print('1:',targets_u1.size())
                print(targets_u1)
                print('2:',targets_u2.size())
                print(targets_u2)
                print('3:',targets_u3.size())
                print(targets_u3)
                if torch.equal(targets_u1, targets_u2) and torch.equal(targets_u1, targets_u3):
                    max_probs = torch.max(max_probs1, max_probs2)
                    max_probs = torch.max(max_probs, max_probs3)
                else: 
                    max_probs = max_probs1 - 0.2
                targets_u = targets_u1
                mask = max_probs.ge(self.args.threshold_pl).float()
                Lu = (F.cross_entropy(ema_logit, targets_u, reduction='none') * mask).mean()
                loss = class_loss + Lu 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)