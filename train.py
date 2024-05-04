#-*-coding:utf-8-*-
# date:2023-12-07
# Author: yinyipeng
## function: train

import os
import argparse

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('components/hand_keypoints/')
from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *
from components.hand_keypoints.models.resnet import resnet18,resnet34,resnet50,resnet101
from components.hand_keypoints.models.squeezenet import squeezenet1_1,squeezenet1_0
from components.hand_keypoints.models.shufflenetv2 import ShuffleNetV2
from components.hand_keypoints.models.shufflenet import ShuffleNet
from components.hand_keypoints.models.mobilenetv2 import MobileNetV2
from loss.loss import *
import time
import json
from loguru import logger
import torch.nn as nn


def calc_mes(predicted, target):
    squared_diff = np.square(predicted - target)
    mes = np.mean(squared_diff)
    return mes

def trainer(ops,f_log):
    writer = SummaryWriter(log_dir='logs')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
        logger.info("GPUS:", ops.GPUS)
        if ops.log_flag:
            sys.stdout = f_log
        pretrained = ops.pretrained
        set_seed(ops.seed)
        #------------------------------------ 构建模型 ------------------------------------
        if ops.model == 'resnet_50':
            model_ = resnet50(pretrained = pretrained,num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_18':
            model_ = resnet18(pretrained = pretrained,num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_ = resnet34(pretrained = pretrained,num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_101':
            model_ = resnet101(pretrained = pretrained,num_classes = ops.num_classes,img_size = ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == "squeezenet1_0":
            model_ = squeezenet1_0(pretrained=pretrained, num_classes=ops.num_classes,dropout_factor=ops.dropout)
        elif ops.model == "squeezenet1_1":
            model_ = squeezenet1_1(pretrained=pretrained, num_classes=ops.num_classes,dropout_factor=ops.dropout)
        elif ops.model == "shufflenetv2":
            model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes, dropout_factor=ops.dropout)
        elif ops.model == "shufflenet":
            model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=ops.num_classes, groups=3, dropout_factor = ops.dropout)
        elif ops.model == "mobilenetv2":
            model_ = MobileNetV2(num_classes=ops.num_classes , dropout_factor = ops.dropout)

        else:
            print(" no support the model")
        print(model_)
        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")


        # Dataset
        print("Loading Dataset....")
        dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res,vis = False)
        print("handpose done")

        print('len train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last = True)
        # 优化器设计
        optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99), weight_decay=1e-6)
        # optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=ops.momentum, weight_decay=ops.weight_decay)# 优化器初始化
        optimizer = optimizer_Adam
        # 定义学习率调度器
        # scheduler = lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
        # 加载 finetune 模型
        if os.access(ops.fintune_model, os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location='cpu')
            print(model_.load_state_dict(chkpt))
            print('load fintune model : {}'.format(ops.fintune_model))
            del chkpt
        model_ = model_.to(device)
        print('/**********************************************/')
        # 损失函数
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0 # 学习率更新计数器
        init_lr = ops.init_lr # 学习率

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            if loss_mean != 0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 50:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0


            loss_mean = 0.  # 损失均值
            loss_idx = 0.  # 损失计算计数器
            mse_list = []
            for i, (imgs_, pts_) in enumerate(dataloader):
                # print('imgs_, pts_',imgs_.size(), pts_.size())
                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda()  # shape [batch_size,42]

                output = model_(imgs_.float())  # output 是21个点位(42个坐标)  shape [batch_size, 42]
                mse = calc_mes(output.cpu().detach().numpy(), pts_.cpu().float().numpy())  # 计算当前batch_size的mse
                # 将MSE指标写入TensorBoard日志
                writer.add_scalar('MSE', mse, global_step=epoch * len(dataloader) + i)

                mse_list.append(mse)
                
                if ops.loss_define == 'wing_loss':
                    loss = got_total_wing_loss(output, pts_.float())
                else:
                    loss = criterion(output, pts_.float())
                loss_mean += loss.item()
                loss_idx += 1.
                writer.add_scalar('mean_loss', loss_mean/loss_idx, global_step=epoch * len(dataloader) + i)
                # if i%10 == 0:
                #     loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                #     print('  %s - %s - epoch [%s/%s] (%s/%s):'%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                #     'Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()),\
                #     ' lr : %.8f'%init_lr,' bs :',ops.batch_size,\
                #     ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss: %.6f'%best_loss)

                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('  %s - %s - epoch [%s/%s] (%s/%s):' % (loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                'Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()),\
                ' lr : %.8f'%init_lr, ' bs :',ops.batch_size,\
                ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]))
                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1
            # 所有batch的mse求平均
            avg_mes = np.mean(mse_list)
            print("{}_avg_mes {}".format(epoch, avg_mes))
            torch.save(model_.state_dict(), ops.model_exp + '{}-size-{}-model_epoch-{}-avg_mse-{}.pth'.format(ops.model, ops.img_size[0], epoch, avg_mes))
        writer.close()
    except Exception as e:
        print('Exception : ',e) # 打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
        print('Exception  line : ', e.__traceback__.tb_lineno)# 发生异常所在的行数

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Hand Train')
    parser.add_argument('--seed', type=int, default=126673, help='seed')  # 设置随机种子
    parser.add_argument('--model_exp', type=str, default='./model_exp', help='model_exp')  # 模型输出文件夹
    parser.add_argument('--model', type=str, default='resnet_50', help = 'model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42, help = 'num_classes') #  landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default='0', help='GPUS') # GPU选择
    parser.add_argument('--train_path', type=str,default = "F:/BaiduNetdiskDownload/handpose_datasets_v1-2021-01-31/handpose_datasets_v1/",help = 'datasets path')  # 训练集标注信息
    parser.add_argument('--pretrained', type=bool, default=False, help = 'imageNet_Pretrain')
    parser.add_argument('--fintune_model', type=str, default='resnet_50-size-256-loss-0.0642.pth', help = 'fintune_model')  # fintune model
    parser.add_argument('--loss_define', type=str, default = 'mse', help = 'define_loss wing_loss or mse')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 5e-4, help = 'init learning Rate')  # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default=0.1, help = 'learningRate_decay')  # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 1e-6, help = 'weight_decay')  # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default = 0.9, help = 'momentum')  # 优化器动量
    parser.add_argument('--batch_size', type=int, default = 4,  help = 'batch_size')  # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default = 0.5, help = 'dropout')  # dropout
    parser.add_argument('--epochs', type=int, default=100,help = 'epochs')  # 训练周期
    parser.add_argument('--num_workers', type=int, default = 4,help = 'num_workers')  # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (256,256),help = 'img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=False,help = 'data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False,help = 'fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default = False,help = 'clear_model_exp')  # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default=False, help='log flag')  # 是否保存训练 log

    #--------------------------------------------------------------------------
    args = parser.parse_args()  # 解析添加参数
    logger.info(args)
    #--------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)  # 建立文件夹model_exp
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp+'/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S",loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    unparsed = vars(args)  # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json',"w",encoding='utf-8')  # 读取训练train_ops.json
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    trainer(ops = args,f_log = f_log)# 模型训练

    if args.log_flag:
        sys.stdout = f_log
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
