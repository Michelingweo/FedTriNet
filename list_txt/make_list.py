#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random


def make_list(label_rate, labeled_file=''):
    '''
    make two data list of labeled data and unlabeled data respectively
    '''
    # the number of images in each class
    num = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949];
    #    total = np.sum(num);
    #    print('the total of data is %d' % total);

    # setting the file name
    if labeled_file == '':
        labeled_file = 'list_txt/mnist_labeled_list.txt';

    labeled_file = open(labeled_file, 'w');

    # random sample
    for i in range(10):
        list_ = list(range(num[i]));
        #        print(int(label_rate*num[i]))
        #        print(label_rate)
        #        print(num[i])
        labeled_list = random.sample(list_, int(label_rate * num[i]));

        for sample in list_:
            img = 'training/' + str(i) + '/' + str(sample) + '.jpg';

            if sample in labeled_list:
                path = img + ' ' + str(i) + '\n';
            else:
                path = img + ' ' + str(-1) + '\n';
            labeled_file.write(path);
            # end_if
        # end_for
    # end_for

    labeled_file.close();


# end_func


def relabel(label, labeled_file=''):
    if labeled_file == '':
        labeled_file = 'list_txt/mnist_labeled_list.txt';
    i = 0
    # 将文件读取到内存中
    with open(labeled_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # 写的方式打开文件
    with open(labeled_file, "w", encoding="utf-8") as f_w:
        for line in lines:
            if "-1" in line:
                # 替换
                line = line.replace("-1", str(label[i]))
            i = i + 1

            f_w.write(line)

    f_w.close();
    f.close();

def maskvector_init(dataset,args,dicuser,dicuser_label):
    mask_vector = np.zeros((int(dataset.__len__()), 4),
                           dtype=int)  # 0: 1-labeled -1-unlabeled 1:user's belonging 2: label/pseudo-label  3: data idx
    # unlabel_vector = []
    label_refer = []
    for i in range(dataset.__len__()):
        mask_vector[i][2] = dataset[i][1]
        label_refer.append(dataset[i][1])
        mask_vector[i][0] = 1
        mask_vector[i][3] = i
    label_idx = []
    for key in dicuser_label.keys():
        for labelidx in dicuser_label[key]:
            label_idx.append(labelidx)
    all_idx = np.arange(dataset.__len__())
    unlabel_idx = list(set(all_idx) - set(label_idx))
    for id in unlabel_idx:
        mask_vector[id][0] = -1
        mask_vector[id][2] = -1

    # user id assign
    for user_id in dicuser.keys():
        for id in dicuser[user_id]:
            mask_vector[id][1] = user_id
    # for i in range(len(mask_vector)):
    #     if mask_vector[i][0] == 1:
    #         label_vector[mask_vector[i][3]] = mask_vector[i]
    #     elif mask_vector[i][0] == -1:
    #         unlabel_vector[mask_vector[i][3]] = mask_vector[i]
    # return  mask_vector, label_vector, unlabel_vector
    return  mask_vector, label_refer


def dictrainUpdate(dic_user_label,dic_pseudo,arg):
    for i in range(arg.num_users):
        dic_user_label[i]+=dic_pseudo[i]

    return dic_user_label


if __name__ == '__main__':
    make_list(200, 'mnist_labeled_list.txt', 'mnist_unlabeled_list.txt');
