#-*-coding:utf-8-*-
# date:2023-12-07
# Author: yinyipeng
# function : classify

import os
import torch
import cv2
import numpy as np
import json

import torch
import torch.nn as nn

import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F
from components.classify_imagenet.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
#
class classify_imagenet_model(object):
    def __init__(self,
        model_path = './components/classify_imagenet/weights/imagenet_size-256_20210409.pth',
        model_arch = "resnet_50",
        img_size= 256,
        num_classes = 1000,
        ):

        f = open("./components/classify_imagenet/imagenet_msg.json",  encoding='utf-8')#读取 json文件
        dict_ = json.load(f)
        f.close()
        self.classify_dict = dict_
        # print("-------------->>\n dict_ : \n",dict_)
#
        print("classify model loading : ",model_path)
        # print('use model : %s'%(model_arch))

        if model_arch == 'resnet_18':
            model_=resnet18(num_classes=num_classes, img_size=img_size)
        elif model_arch == 'resnet_34':
            model_=resnet34(num_classes=num_classes, img_size=img_size)
        elif model_arch == 'resnet_50':
            model_=resnet50(num_classes=num_classes, img_size=img_size)
        elif model_arch == 'resnet_101':
            model_=resnet101(num_classes=num_classes, img_size=img_size)
        elif model_arch == 'resnet_152':
            model_=resnet152(num_classes=num_classes, img_size=img_size)
        else:
            print('error no the struct model : {}'.format(model_arch))

        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)
        model_.eval() # 设置为前向推断模式

        # print(model_)# 打印模型结构

        # 加载测试模型
        if os.access(model_path,os.F_OK):# checkpoint
            chkpt = torch.load(model_path, map_location=device)
            model_.load_state_dict(chkpt)
            # print('load classify model : {}'.format(model_path))
        self.model_ = model_
        self.use_cuda = use_cuda
        self.img_size = img_size

    def predict(self, img, vis = False):# img is align img
        with torch.no_grad():

            img_ = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_CUBIC)

            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            pre_ = self.model_(img_.float())

            outputs = F.softmax(pre_,dim = 1)
            outputs = outputs[0]

            output = outputs.cpu().detach().numpy()
            output = np.array(output)

            max_index = np.argmax(output)

            score_ = output[max_index]
            # print("max_index:",max_index)
            # print("name:",self.label_dict[max_index])
            return max_index,self.classify_dict[str(max_index)],score_
