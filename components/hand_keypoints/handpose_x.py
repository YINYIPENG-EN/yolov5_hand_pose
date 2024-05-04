#-*-coding:utf-8-*-
# date:2021-03-09
# Author: Eric.Lee
# function: handpose_x 21 keypoints 2D

import os
import torch
import cv2
import numpy as np
import json

import torch
import torch.nn as nn

import time
import math
from datetime import datetime

from components.hand_keypoints.models.resnet import resnet18,resnet34,resnet50,resnet101
from components.hand_keypoints.models.squeezenet import squeezenet1_1,squeezenet1_0
from components.hand_keypoints.models.resnet_50 import resnet50
from components.hand_keypoints.models.resnet import resnet18,resnet34,resnet50,resnet101
from components.hand_keypoints.models.squeezenet import squeezenet1_1,squeezenet1_0
from components.hand_keypoints.models.shufflenetv2 import ShuffleNetV2
from components.hand_keypoints.models.shufflenet import ShuffleNet
from components.hand_keypoints.models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from components.hand_keypoints.models.rexnetv1 import ReXNetV1


from components.hand_keypoints.utils.common_utils import *

def draw_bd_handpose_c(img_,hand_,x,y,thick=3):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)

#
class handpose_x_model(object):
    def __init__(self,
        model_path = 'E:/dpcas-master/dpcas-master/components/hand_keypoints/weights/ReXNetV1-size-256-wingloss102-0.122.pth',
        img_size= 256,
        num_classes = 42,# 手部关键点个数 * 2 ： 21*2
        model_arch = "rexnetv1"
        ):
        print("handpose_x loading : ",model_path)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu") # 可选的设备类型及序号
        self.img_size = img_size
        self.model_arch = model_arch
        #-----------------------------------------------------------------------
        if model_arch == 'resnet_50':
            model_ = resnet50(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == 'resnet_18':
            model_ = resnet18(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == 'resnet_34':
            model_ = resnet34(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == 'resnet_101':
            model_ = resnet101(num_classes = num_classes,img_size = self.img_size)
        elif model_arch == "squeezenet1_0":
            model_ = squeezenet1_0(pretrained=True, num_classes=num_classes)
        elif model_arch == "squeezenet1_1":
            model_ = squeezenet1_1(pretrained=True, num_classes=num_classes)
        elif model_arch == "shufflenetv2":
            model_ = ShuffleNetV2(ratio=1., num_classes=num_classes)
        elif model_arch == "shufflenet_v2_x1_5":
            model_ = shufflenet_v2_x1_5(pretrained=False,num_classes=num_classes)
        elif model_arch == "shufflenet_v2_x1_0":
            model_ = shufflenet_v2_x1_0(pretrained=False,num_classes=num_classes)
        elif model_arch == "shufflenet_v2_x2_0":
            model_ = shufflenet_v2_x2_0(pretrained=False,num_classes=num_classes)
        elif model_arch == "shufflenet":
            model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=num_classes, groups=3)
        elif model_arch == "mobilenetv2":
            model_ = MobileNetV2(num_classes=num_classes)
        elif model_arch == "rexnetv1":
            model_ = ReXNetV1(num_classes=num_classes)
        else:
            print("model_arch=", model_arch)
            print(" no support the model")
        #-----------------------------------------------------------------------
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)
        #print(model_)
        model_.eval() # 设置为前向推断模式

        # 加载测试模型
        if os.access(model_path,os.F_OK):# checkpoint
            chkpt = torch.load(model_path, map_location=self.device)
            model_.load_state_dict(chkpt)
            print('handpose_x model loading : {}'.format(model_path))

        self.model_handpose = model_

    def predict(self, img, vis = False):
        with torch.no_grad():

            if not((img.shape[0] == self.img_size) and (img.shape[1] == self.img_size)):
                img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_CUBIC)

            img_ = img.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if self.use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            pre_ = self.model_handpose(img_.float())
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            return output
