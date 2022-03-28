#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # semi-supervised setting
#     parser.add_argument('--label_rate', type=float, default=0.0137*3, help="the fraction of labeled data")
    parser.add_argument('--label_rate', type=float, default=0.01, help="the fraction of labeled data")
    # federated arguments
    parser.add_argument('--p1epochs', type=int, default=40, help="rounds of training")
    parser.add_argument('--p2epochs', type=int, default=60, help="rounds of training")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--local_bs_pseudo', type=int, default=50, help="local batch size: B")
    parser.add_argument('--local_bs_label', type=int, default=50, help="local batch size: B")
    parser.add_argument('--local_bs_unlabel', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    
    
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--initial-lr', default=0.02, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--threshold_pl', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--lambda-u', default=1, type=float,help='coefficient of unlabeled loss')
    # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    
    parser.add_argument('--data_argument', type=str, default='True', help="data argumentation")
    parser.add_argument('--swa', action='store_false', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=200, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--finetune', action='store_false', help='Apply fine-tune')
    parser.add_argument('--tp', action='store_false', help='Apply threshold protection')
    parser.add_argument('--pseudo_select', action='store_false', help='Apply pseudo selection')
    parser.add_argument('--compare_key', action='store_true', help='w or w tuned')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    
    parser.add_argument('--layer_select', type=int, default=1, help='number of conv blocks updated for clients')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=str, default='iid', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args
