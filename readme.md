# YOLOV5_HANDPOSE

之前是做的**yolov3手势物体识别**，最近几天我将该项目进行了重新的整理和升级，实现了**yolov5手势物体识别**。

同时为了方便更多的人直接拿来应用，我生成了支持windows系统的应用小程序，即便你电脑上**没有安装pytorch,没有安装cuda、python，都可以使用**~！

**相关资料：**

[yolov3手势物体识别](https://blog.csdn.net/z240626191s/article/details/123289979?spm=1001.2014.3001.5502)

**应用程序效果如下：**

<iframe id="ceXRG6sL-1714752095865" frameborder="0" src="https://player.bilibili.com/player.html?aid=1454111830" allowfullscreen="true" data-mediaembed="bilibili" style="box-sizing: border-box; outline: 0px; margin: 0px; padding: 0px; font-weight: normal; font-synthesis-style: auto; overflow-wrap: break-word; display: block; width: 730px; height: 365px;"></iframe>

yolov5手势[物体识别](https://so.csdn.net/so/search?q=物体识别&spm=1001.2101.3001.7020)



# 环境说明

```
torch                     1.7.0

tensorboard               1.15.0

protobuf                  3.20.0

Pillow                    9.5.0

opencv-python             4.4.0.44
```



# 技术说明

本项目使用了三个算法模型进行的功能实现。yolov5做手部目标检测，ReXNet(支持Resnet系列)做手部21关键点回归检测，Resnet50做物体分类识别。(其实就是三个算法做的级联)

## yolov5手部目标检测

使用yolov5s进行训练，数据集共有3W+，因本地训练环境受限，我只训练到mAP 64%左右，因此准确率并不是很高，大家有需要的可以自行再去训练~

### 数据集说明

**数据集链接：**

(ps:这里的数据集采用的公共数据集，没有做过数据清洗)

链接：https://pan.baidu.com/s/1jnXH3yBuGJ8_DRXu-gKtNg 
提取码：yypn 

**数据集格式：**

images:存放所有的数据集

labels:已经归一化后的label信息

train.txt:训练集划分，25934张

val.txt:验证集划分，3241张

test.txt：测试集划分，3242张图



**训练实验记录**
采用马赛克数据增强


 **评价指标：**

(我这里只训练了大该十多个epoch，没服务器训练太慢了~(T^T)~,所以准确率比较低，有需要的可以自己训练一下)

(训练好的权重见文末项目链接)

| P       | R       | mAP 0.5 | mAP 0.5:0.95 |
| ------- | ------- | ------- | ------------ |
| 0.75396 | 0.59075 | 0.64671 | 0.27652      |

## 手部21关键点检测

手部关键点识别采用的网络为ReXNet(支持Resnet系列)，这里需要说明的是关键点预测并没有采用openpose网络！而是采用的坐标回归方法，这个问题需要强调一下，不然总有小伙伴问我，而且还很质疑~ 在本任务中，由于有yolo作为前置滤波器算法将手部和背景进行分离，分离后的图像前景和背景相对平衡，而且前景(手部)占主要部分，因此任务其实相对简单，可以采用坐标回归方法。

网络的定义在yolov5_hand_pose/components/hand_keypoints/models/。

21个关键点，那就是有42个坐标(x,y坐标)。因此代码中num_classes=42.

### 数据集说明

(ps:这里的数据集采用的公共数据集，没有做过数据清洗)

**数据集链接：**

链接：https://pan.baidu.com/s/129aFPmhHq3lWmAFkuBI3BA 
提取码：yypn 
整个数据集共有49062张图像。

### 训练

训练代码在train.py中。

(训练好的权重见文末项目链接)

可采用提供的预权重进行fine tune训练。

输入以下命令开始训练：

```
python train.py --model resnet_50 --train_path [数据集路径] --fintune_model 【fine tune模型路径】--batch_size 16
```

如果是fine tune训练，建议初始学习率(init_lr)设置为5e-4，否则建议设置为1e-3。

损失函数此次采用的是MSE，还可支持wing loss。

训练好的权重会保存在model_exp中，对应的tensorboard会存储在logs中

## 分类网络

这里的分类网络采用是resnet50网络，权重为ImageNet数据集上的(1000个类)，可以根据自己任务需求去训练。(权重见文末项目链接)

网络定义在yolov5_hand_pose/components/classify_imagenet/models。

那具体是如何分类的呢？

首先触发分类模型的手势是食指和大拇指捏合。主要是计算两个关键点的欧式距离，当距离小于阈值则为触发状态click_state=True，表示手势触发成功。

当两个手都触发捏合动作，那么判断是有效动作，同时**将左右手选定的区域截出来(和yolo的操作类似)，送入分类网络进行分类识别**。

## 语音播报

当手势动作触发成功后会触发语音播报函数，此时会自动语音播放"正在识别物体请等待"，如果成功识别，并且也有该物体的语音包(需要自己录制)，那么会说“您识别的物体为。。。”

如果需要自己录制语音(mp3格式)，可以将录制好的语音放在materials/audio/imagenet_2012/

# 如何使用本项目

使用方法很简单，clone本项目到本地后，只需要运行predict.py并搭配参数即可。

（提前下载好权重~）

你可能会用到如下参数：

```
--hand_weight  【yolov5权重路径】，默认为best.pt
--handpose_model_path  【关键点权重】，默认components/hand_keypoints/weights/ReXNetV1-size-256-wingloss102-0.122.pth
--handpose_name 【关键点模型】，默认rexnetv1
--classify_model_path 【分类网络权重】，默认components/classify_imagenet/weights/imagenet_size-256_20210409.pth
--classify_model_name 【分类网络模型名称】，默认resnet_50
--conf 【yolo置信度阈值】，默认0.5
--video_path 【视频路径】，默认本机摄像头
--device 【推理设备】，默认GPU

```

例如：

```
python predict.py --conf 0.3 --video_path 0 --hand_weight best.pt --device cuda
```

# 手势物体识别应用程序

为了可以让更多的进行使用，花费了两天的时间导出了exe应用程序，即便你的电脑没有安装pytorch和cuda都可以直接运行(暂时只支持windows系统，linux应该是需要wine来帮助运行)。

ps:博主只是在一些电脑上进行了测试还是可以成功运行的~

应用程序链接：

链接：https://pan.baidu.com/s/1wPpg2v4h2Zlkr5SgzCGgVw 
提取码：yypn 

运行方式：

1.直接双击predict.exe可直接运行程序

2.在cmd运行predict.exe可直接运行程序，推荐这种方式，因为可以搭配命令使用，同时有报错可以看到。

可搭配的命令如下：

```
--hand_weight  【yolov5权重路径】，默认为best.pt
--handpose_model_path  【关键点权重】，默认components/hand_keypoints/weights/ReXNetV1-size-256-wingloss102-0.122.pth
--handpose_name 【关键点模型】，默认rexnetv1
--classify_model_path 【分类网络权重】，默认components/classify_imagenet/weights/imagenet_size-256_20210409.pth
--classify_model_name 【分类网络模型名称】，默认resnet_50
--conf 【yolo置信度阈值】，默认0.5
--video_path 【视频路径】，默认本机摄像头
--device 【推理设备】，默认GPU
```

输入命令样例：

```bash
predict.exe --conf 0.3 --video 0
```

# 权重链接

链接：https://pan.baidu.com/s/1WS3Nb5MkqMGhCKjM7DYsgg 
提取码：yypn 

云盘中有三个权重：

best.pt是yolov5训练的权重

ReXNetV1-size-256-wingloss102-0.122.pth是21关键点权重

imagenet_size-256_20210409.pth是分类网络权重

