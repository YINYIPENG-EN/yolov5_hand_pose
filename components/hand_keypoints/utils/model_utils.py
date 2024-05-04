#-*-coding:utf-8-*-
# date:2020-04-11
# Author: Eric.Lee
# function: model utils

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / float(total)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_seed(seed = 666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True

def split_trainval_datasets(ops):
    print(' --------------->>> split_trainval_datasets ')
    train_split_datasets = []
    train_split_datasets_label = []

    val_split_datasets = []
    val_split_datasets_label = []
    for idx,doc in enumerate(sorted(os.listdir(ops.train_path), key=lambda x:int(x.split('.')[0]), reverse=False)):
        # print(' %s label is %s \n'%(doc,idx))

        data_list = os.listdir(ops.train_path+doc)
        random.shuffle(data_list)

        cal_split_num = int(len(data_list)*ops.val_factor)

        for i,file in enumerate(data_list):
            if '.jpg' in file:
                if i < cal_split_num:
                    val_split_datasets.append(ops.train_path+doc + '/' + file)
                    val_split_datasets_label.append(idx)
                else:
                    train_split_datasets.append(ops.train_path+doc + '/' + file)
                    train_split_datasets_label.append(idx)

                print(ops.train_path+doc + '/' + file,idx)

    print('\n')
    print('train_split_datasets len {}'.format(len(train_split_datasets)))
    print('val_split_datasets len {}'.format(len(val_split_datasets)))

    return train_split_datasets,train_split_datasets_label,val_split_datasets,val_split_datasets_label
