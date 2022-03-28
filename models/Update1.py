#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics
from itertools import cycle




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


class LocalUpdate(object):
    def __init__(self, args, maskv, dataset=None, idxs=None, idxs2=None,idxs3 = None,stage=1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1).cuda()
        self.loss_semifunc = nn.CrossEntropyLoss(ignore_index=-1,reduction='none').cuda()
        self.selected_clients = []
        self.dataset = dataset
        self.mask = maskv
        self.idxs3 = idxs3
#         self.sampler = RandomSampler(data_source=dataset,replacement=True)
        if stage == 0:
            self.ldr_label = DataLoader(DatasetSplit_mask(dataset, idxs, mask_vector=maskv), batch_size=self.args.local_bs_label, shuffle=True)
        elif stage == 1:
            self.ldr_label = DataLoader(DatasetSplit_mask(dataset, idxs, mask_vector=maskv), batch_size=self.args.local_bs_label, shuffle=True)
            self.ldr_unlabel = DataLoader(DatasetSplit_mask(dataset, idxs2, mask_vector=maskv),batch_size=self.args.local_bs_unlabel, shuffle=True)
        
            
            
       


    def Phase2_train(self, C, D,img_size, sever_round):
        T1 = 50
        T2 = 150

        af = 0.15
        alpha_P = 0.5
        ldr_pseudo = DataLoader(DatasetSplit_mask(self.dataset, self.idxs3, mask_vector=self.mask),batch_size=self.args.local_bs_pseudo, shuffle=True)
        # label preprocess
        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1
        
        C.train()
        D.train()
        # train and update
        optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerC = torch.optim.Adam(C.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        
        
        
        BCE_loss = torch.nn.BCELoss().cuda()
        CE_loss = self.loss_func
        
        epoch_Closs = []
        epoch_Dloss = []
        for iter in range(self.args.local_ep):
            batch_Closs = []
            batch_Dloss = []
            for batch_idx, ((image1, labels1),(image2,labels2),(image3,labels3)) in enumerate(zip(cycle(self.ldr_label),self.ldr_unlabel,ldr_pseudo)):
                print(batch_idx)
                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()
                image_u, label_u = image2.to(self.args.device), labels2.to(self.args.device).long()
                image_p, label_p = image3.to(self.args.device), labels3.to(self.args.device).long()
                
                label_ld = fill[label_l].cuda()
                label_ud = fill[label_u].cuda()
                
                mini_batch_l = image_l.size()[0]
                mini_batch_u = image_u.size()[0]

           
                y_real_l = torch.ones(mini_batch_l).float()
                y_fake_l = torch.ones(mini_batch_l).float()
                y_real_l, y_fake_l = y_real_l * 0.9, y_fake_l * 0.1
                y_real_l, y_fake_l = y_real_l.to(self.args.device).float(), y_fake_l.to(self.args.device).float()


                y_real_u = torch.ones(mini_batch_u).float()
                y_fake_u = torch.ones(mini_batch_u).float()
                y_real_u, y_fake_u = y_real_u * 0.9, y_fake_u * 0.1
                y_real_u, y_fake_u = y_real_u.to(self.args.device).long(), y_fake_u.to(self.args.device).float()
            


                C.zero_grad()
                D.zero_grad()
                # labeled forward pass
                optimizerC.zero_grad()
               
                optimizerD.zero_grad()

                ######################
                # train Discriminator by labeled data
                ######################
                D.zero_grad()
                log_probsD_real = D(image_l, label_ld)
                D_loss_real = torch.mean(BCE_loss(log_probsD_real, y_real_l))
                
                
                 ######################
                # utilizing of unlabeled data
                ######################
                pseudo_label = C(image_u)
#                 print(pseudo_label)
                pseudo_label = F.softmax(pseudo_label, dim=1)
                max_c = torch.argmax(pseudo_label).float()
                _ = torch.argmax(pseudo_label, dim=1).long()
#                 print(_)
                pseudo_labeld = fill[_].cuda()
                log_probsD_fake = D(image_u, pseudo_labeld)

                D_loss_cla = torch.mean(BCE_loss(log_probsD_fake, y_fake_u))

                C_loss_dis = torch.mean(max_c * self.loss_func(log_probsD_fake, y_real_u))
                
                output1 = C(image_l)  # labeled data 预测
                
                output1 = output1.to(self.args.device)
                output2 = C(image_p).to(self.args.device)
                # unlabeled forward pass

                Closs1 = self.loss_func(output1, label_l)
                Closs1 = Closs1.to(self.args.device)

                # unlabeled lossfunc weight
                Iteration = int(iter + 5 * (sever_round))
                if Iteration < T1:
                    alpha = 0.
                elif Iteration > T2:
                    alpha = af
                else:
                    alpha = (Iteration - T1) / (T2 - T1) * af
                # end_if

                Closs2 = self.loss_func(output2, label_p)
                Closs2 = Closs2.to(self.args.device)

                Closs = Closs1 + alpha * Closs2 + 0.01 * alpha_P * C_loss_dis
                Closs = Closs.to(self.args.device)
                Closs.backward(retain_graph=True)
                optimizerC.step()
                
                Dloss = D_loss_real + alpha_P * D_loss_cla
                Dloss.backward()
                optimizerD.step()
                
                

                batch_Closs.append(Closs.item())
                batch_Dloss.append(Dloss.item())
            epoch_Closs.append(sum(batch_Closs)/len(batch_Closs))
            epoch_Dloss.append(sum(batch_Dloss)/len(batch_Dloss))
        return C.state_dict(), D.state_dict(), sum(epoch_Closs) / len(epoch_Closs),sum(epoch_Dloss) / len(epoch_Dloss)

    
    def Phase1_train(self, C, D, img_size):
        
        alpha_P = 0.5
        
        # label preprocess
        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1
        
        C.train()
        D.train()
        # train and update
        optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        optimizerC_pre = torch.optim.SGD(filter(lambda p: p.requires_grad, C.parameters()), lr=0.01, momentum=0.5)
        
        
        BCE_loss = torch.nn.BCELoss().cuda()
        CE_loss = self.loss_func
        
        epoch_Closs = []
        epoch_Dloss = []
        for iter in range(self.args.local_ep):
            batch_Closs = []
            batch_Dloss = []
            for batch_idx, ((image1, labels1),(image2,labels2)) in enumerate(zip(self.ldr_label,self.ldr_unlabel)):

                image_l, label_l = image1.to(self.args.device), labels1.to(self.args.device).long()
                image_u, label_u = image2.to(self.args.device), labels2.to(self.args.device).long()
                label_ld = fill[label_l].cuda()
                label_ud = fill[label_u].cuda()
                
                mini_batch_l = image_l.size()[0]
                mini_batch_u = image_u.size()[0]

           
                y_real_l = torch.ones(mini_batch_l).float()
                y_real_l = y_real_l * 0.9
                y_real_l = y_real_l.to(self.args.device).float()


                
                y_fake_u = torch.ones(mini_batch_u).float()
                y_fake_u = y_fake_u * 0.1
                y_fake_u =  y_fake_u.to(self.args.device).float()

                C.zero_grad()
                D.zero_grad()
                # labeled forward pass
                
                optimizerC_pre.zero_grad()
                optimizerD.zero_grad()

                ######################
                # train Discriminator by labeled data
                ######################
                D.zero_grad()
                log_probsD_real = D(image_l, label_ld)
                D_loss_real = torch.mean(BCE_loss(log_probsD_real, y_real_l))
                
                
                 ######################
                # utilizing of unlabeled data
                ######################
                pseudo_label = C(image_u)
#                 print(pseudo_label)
                max_c = torch.argmax(pseudo_label).float()
                _ = torch.argmax(pseudo_label, dim=1).long()
#                 print(_)
                pseudo_labeld = fill[_].cuda()
                log_probsD_fake = D(image_u, pseudo_labeld)

                D_loss_cla = torch.mean(BCE_loss(log_probsD_fake, y_fake_u))

                
                
                output = C(image_l)  # labeled data 预测
                
                output = output.to(self.args.device)
                
                Closs = self.loss_func(output, label_l)
                Closs = Closs.to(self.args.device)


                Closs.backward(retain_graph=True)
                optimizerC_pre.step()
                
                Dloss = D_loss_real + alpha_P * D_loss_cla
                Dloss.backward()
                optimizerD.step()
                
                
                batch_Closs.append(Closs.item())
                batch_Dloss.append(Dloss.item())
            epoch_Closs.append(sum(batch_Closs)/len(batch_Closs))
            epoch_Dloss.append(sum(batch_Dloss)/len(batch_Dloss))
        return C.state_dict(), D.state_dict(), sum(epoch_Closs) / len(epoch_Closs),sum(epoch_Dloss) / len(epoch_Dloss)

    def init_train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (img, label) in enumerate(self.ldr_label):
                 #                print(batch_idx)
                img, label = img.to(self.args.device), label.to(self.args.device).long()
                net.zero_grad()
                log_probs = net(img)
                loss = self.loss_func(log_probs, label)
                loss.backward()
                optimizer.step()
                
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
            for batch_idx, (img, label) in enumerate(self.ldr_label):
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

    