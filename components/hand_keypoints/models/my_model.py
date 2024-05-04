#-*-coding:utf-8-*-
# date:2020-08-08
# Author: X.L.Eric
# function: my model

import torch
import torch.nn as nn
import torch.nn.functional as F
class MY_Net(nn.Module):
    def __init__(self,num_classes):# op 初始化
        super(MY_Net, self).__init__()
        self.cov = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU(inplace=True)
        layers1 = []
        # Conv2d : in_channels, out_channels, kernel_size, stride, padding
        layers1.append(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,stride=1,padding = 0))
        layers1.append(nn.BatchNorm2d(64,affine=True))
        layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.layers1 = nn.Sequential(*layers1)
        layers2 = []
        layers2.append(nn.Conv2d(64, 128, 3))
        layers2.append(nn.BatchNorm2d(128,affine=True))
        layers2.append(nn.ReLU(inplace=True))
        layers2.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers2 = nn.Sequential(*layers2)
        layers3 = []
        layers3.append(nn.Conv2d(128, 256, 3,stride=2))
        layers3.append(nn.BatchNorm2d(256,affine=True))
        layers3.append(nn.ReLU(inplace=True))
        layers3.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layers3 = nn.Sequential(*layers3)
        layers4 = []
        layers4.append(nn.Conv2d(256, 512, 3,stride=2))
        layers4.append(nn.BatchNorm2d(512,affine=True))
        layers4.append(nn.ReLU(inplace=True))
        layers4.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers4.append(nn.Conv2d(512, 512, 1,stride=1))
        self.layers4 = nn.Sequential(*layers4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))# 自适应均值池化
        self.fc = nn.Linear(in_features = 512 , out_features = num_classes)# 全连接 fc

    def forward(self, x):# 模型前向推断
        x = self.cov(x)
        x = self.relu(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    #输入批次图片（batchsize，channel，height，width）：8 ，3*256*256
    dummy_input = torch.randn([8, 3, 256,256])
    model = MY_Net(num_classes = 100)# 分类数为 100 类
    print('model:\n',model)# 打印模型op
    output = model(dummy_input)# 模型前向推断
    # 模型前向推断输出特征尺寸
    print('model inference feature size: ',output.size())
    print(output)

    output_ = F.softmax(output,dim = 1)
    #
    print(output_)
