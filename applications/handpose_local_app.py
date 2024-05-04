#-*-coding:utf-8-*-
# date:2023-12-07
# Author: yinyipeng
# function: handpose demo

import os
import cv2
import time

from multiprocessing import Process
from multiprocessing import Manager

import cv2
import numpy as np
import random
import time
import sys

import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
# 加载模型组件库
# from hand_detect.yolo_v3_hand import yolo_v3_hand_model
from components.hand_keypoints.handpose_x import handpose_x_model
from components.classify_imagenet.imagenet_c import classify_imagenet_model
from utils.plots import Annotator, colors, save_one_box
# 加载工具库

# from lib.hand_lib.cores.handpose_fuction import handpose_track_keypoints21_pipeline
# from lib.hand_lib.cores.handpose_fuction import hand_tracking,audio_recognize,judge_click_stabel,draw_click_lines
# from lib.hand_lib.utils.utils import parse_data_cfg
from playsound import playsound

from lib.hand_lib.cores.handpose_fuction import hand_tracking, handpose_track_keypoints21_pipeline, draw_click_lines, \
    judge_click_stabel, audio_recognize

def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh
def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RG25
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def audio_process_dw_edge_cnt(info_dict): #dw是down，下降沿（dw_edge）

    while (info_dict["handpose_procss_ready"] == False): # 等待 模型加载 判断手关键点进程是否运行
        time.sleep(2)

    gesture_names = ["click"]
    gesture_dict = {}

    for k_ in gesture_names:
        gesture_dict[k_] = None

    reg_cnt = 0
    while True:
        time.sleep(0.01)
        try:
            reg_cnt = info_dict["click_dw_cnt"]
            for i in range(reg_cnt):

                playsound("./materials/audio/sentences/welldone.mp3")
            info_dict["click_dw_cnt"] = info_dict["click_dw_cnt"] - reg_cnt
        except Exception as inst:
            print(type(inst),inst)    # exception instance


        if info_dict["break"] == True:
            break

def audio_process_up_edge_cnt(info_dict): #语音播放进程上升沿有效

    while (info_dict["handpose_procss_ready"] == False): # 等待 模型加载
        time.sleep(2)

    gesture_names = ["click"]
    gesture_dict = {}

    for k_ in gesture_names:
        gesture_dict[k_] = None

    reg_cnt = 0
    while True:
        time.sleep(0.01)
        # print(" --->>> audio_process")
        try:
            reg_cnt = info_dict["click_up_cnt"]
            for i in range(reg_cnt):

                playsound("./materials/audio/sentences/Click.mp3")
            info_dict["click_up_cnt"] = info_dict["click_up_cnt"] - reg_cnt
        except Exception as inst:
            print(type(inst),inst)    # the exception instance


        if info_dict["break"] == True:
            break

def audio_process_dw_edge(info_dict):

    while (info_dict["handpose_procss_ready"] == False): # 等待 模型加载
        time.sleep(2)

    gesture_names = ["click"]
    gesture_dict = {}

    for k_ in gesture_names:
        gesture_dict[k_] = None
    while True:
        time.sleep(0.01)
        # print(" --->>> audio_process")
        try:
            for g_ in gesture_names:
                if gesture_dict[g_] is None:
                    gesture_dict[g_] = info_dict[g_]
                else:

                    if ("click"==g_):
                        if (info_dict[g_]^gesture_dict[g_]) and info_dict[g_]==False:# 判断Click手势信号为下降沿，Click动作结束
                            playsound("./materials/audio/cue/winwin.mp3")


                    gesture_dict[g_] = info_dict[g_]

        except Exception as inst:
            print(type(inst),inst)    # the exception instance


        if info_dict["break"] == True:
            break

def audio_process_up_edge(info_dict):

    while (info_dict["handpose_procss_ready"] == False): # 等待 模型加载
        time.sleep(2)

    gesture_names = ["click"]
    gesture_dict = {}

    for k_ in gesture_names:
        gesture_dict[k_] = None
    while True:
        time.sleep(0.01)
        # print(" --->>> audio_process")
        try:
            for g_ in gesture_names:
                if gesture_dict[g_] is None:
                    gesture_dict[g_] = info_dict[g_]
                else:

                    if ("click"==g_):
                        if (info_dict[g_]^gesture_dict[g_]) and info_dict[g_]==True:# 判断Click手势信号为上升沿，Click动作开始
                            playsound("./materials/audio/cue/m2.mp3")
                            # playsound("./materials/audio/sentences/clik_quick.mp3")

                    gesture_dict[g_] = info_dict[g_]

        except Exception as inst:
            print(type(inst),inst)    # the exception instance


        if info_dict["break"] == True:
            break
'''
    启动识别语音进程  该项目中主要用到下面的自定义函数
'''
def audio_process_recognize_up_edge(info_dict):

    while (info_dict["handpose_procss_ready"] == False): # 等待 模型加载
        time.sleep(2)

    gesture_names = ["double_en_pts"] # 姿态列表，
    gesture_dict = {}

    for k_ in gesture_names:#k_= double_en_pts
        gesture_dict[k_] = None #gesture_dict[double_en_pts]=None

    while True:
        time.sleep(0.01)
        # print(" --->>> audio_process")
        try:
            for g_ in gesture_names:  # 输出 double_en_pts ，因为gesture_name列表内容为str，所以g_也是str,输出的g_=double_en_pts
                if gesture_dict[g_] is None:  # gesture_dict[double_en_pts] 为真
                    gesture_dict[g_] = info_dict[g_]  #info_dict[g_]为False
                else:

                    if ("double_en_pts"==g_):
                        if (info_dict[g_]^gesture_dict[g_]) and info_dict[g_]==True:# 判断Click手势信号为上升沿，Click动作开始
                            playsound("./materials/audio/sentences/IdentifyingObjectsWait.mp3")
                            playsound("./materials/audio/sentences/ObjectMayBeIdentified.mp3")
                            if info_dict["reco_msg"] is not None:
                                print("process - (audio_process_recognize_up_edge) reco_msg : {} ".format(info_dict["reco_msg"]))
                                doc_name = info_dict["reco_msg"]["label_msg"]["doc_name"]
                                reco_audio_file = "./materials/audio/imagenet_2012/{}.mp3".format(doc_name)
                                if os.access(reco_audio_file,os.F_OK):# 判断语音文件是否存在
                                    playsound(reco_audio_file)

                                info_dict["reco_msg"] = None

                    gesture_dict[g_] = info_dict[g_]

        except Exception as inst:
            print(type(inst),inst)    # exception instance

        if info_dict["break"] == True:
            break
'''
/*****************************************/
                算法 pipeline
/*****************************************/
'''
def handpose_x_process(info_dict, config):
    # 模型初始化
    print("load model component  ...")
    # yolo v5 手部检测模型初始化
    device = torch.device('cuda' if config.device == 'cuda' else 'cpu')
    hand_detect_model = DetectMultiBackend(config.hand_weight, device=device, dnn=False, data=config.data, fp16=False)
    stride, names, pt = hand_detect_model.stride, hand_detect_model.names, hand_detect_model.pt
    # handpose_x 21 关键点回归模型初始化
    handpose_model = handpose_x_model(model_arch = config.handpose_name,model_path = config.handpose_model_path)
    #
    gesture_model = None # 目前缺省
    #
    object_recognize_model = classify_imagenet_model(model_arch=config.classify_model_name, model_path = config.classify_model_path,
        num_classes = int(config.num_class)) # 识别分类模型

    #
    img_reco_crop = None
    video_path = config.video_path
    if video_path.isnumeric():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path) # 开启摄像机
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fps = 0.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 获取视频的宽和高
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('test.avi', fourcc, 25, size)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0) # 设置相机曝光，（注意：不是所有相机有效）

    print("start handpose process ~")

    info_dict["handpose_procss_ready"] = True #多进程间的开始同步信号

    gesture_lines_dict = {} # 点击使能时的轨迹点

    hands_dict = {} # 手的信息
    hands_click_dict = {} #手的按键信息计数
    track_index = 0 # 跟踪的全局索引

    while True:
        ret, img = cap.read()# 读取相机图像
        if ret:# 读取相机图像成功
            # img = cv2.flip(img,-1)
            algo_img = img.copy()
            st_ = time.time()
            with torch.no_grad():
                img0 = process_data(img, config.input_shape)
                img0 = torch.from_numpy(img0)
                img0 = img0[None]
                img0 = img0.to(config.device)
                pred = hand_detect_model(img0)  # 检测手，获取手的边界框
            hand_bbox = []
            annotator = Annotator(algo_img, line_width=3, example=str(names))
            det = non_max_suppression(pred, config.conf, config.iou, None, False, max_det=1000)[0]
            det[:, :4] = scale_boxes(img0.shape[2:], det[:, :4], algo_img.shape).round()
            for *xyxy, conf, cls in reversed(det):  # 遍历每个目标
                xmin = int(xyxy[0])
                ymin = int(xyxy[1])
                xmax = int(xyxy[2])
                ymax = int(xyxy[3])
                w = xmax - xmin
                h = ymax - ymin
                if w * h > 200:
                    hand_bbox.append((xmin, ymin, xmax, ymax, conf))
                    label = (names[0])
                    annotator.box_label(xyxy, label, color=colors(0, True))
            img = annotator.result()
            hands_dict, track_index = hand_tracking(data=hand_bbox,hands_dict = hands_dict,track_index = track_index) # 手跟踪，目前通过IOU方式进行目标跟踪
            # 检测每个手的关键点及相关信息
            handpose_list = handpose_track_keypoints21_pipeline(img, hands_dict = hands_dict,hands_click_dict = hands_click_dict,track_index = track_index,algo_img = algo_img,
                handpose_model = handpose_model,gesture_model = gesture_model,
                icon = None,vis = True)
            et_ = time.time()
            fps_ = 1./(et_-st_+1e-8)
            #------------------------------------------ 跟踪手的 信息维护
            #------------------ 获取跟踪到的手ID
            id_list = []
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_list.append(dict_["id"])
            # print(id_list)
            #----------------- 获取需要删除的手ID
            id_del_list = []
            for k_ in gesture_lines_dict.keys():
                if k_ not in id_list:#去除过往已经跟踪失败的目标手的相关轨迹
                    id_del_list.append(k_)
            #----------------- 删除无法跟踪到的手的相关信息
            for k_ in id_del_list:
                del gesture_lines_dict[k_]
                del hands_click_dict[k_]

            #----------------- 更新检测到手的轨迹信息,及手点击使能时的上升沿和下降沿信号
            double_en_pts = []
            for i in range(len(handpose_list)):
                _,_,_,dict_ = handpose_list[i]
                id_ = dict_["id"]
                if dict_["click"]:
                    if  id_ not in gesture_lines_dict.keys():
                        gesture_lines_dict[id_] = {}
                        gesture_lines_dict[id_]["pts"]=[]
                        gesture_lines_dict[id_]["line_color"] = (random.randint(100,255),random.randint(100,255),random.randint(100,255))
                        gesture_lines_dict[id_]["click"] = None
                    #判断是否上升沿
                    if gesture_lines_dict[id_]["click"] is not None:
                        if gesture_lines_dict[id_]["click"] == False:#上升沿计数器
                            info_dict["click_up_cnt"] += 1
                    #获得点击状态
                    gesture_lines_dict[id_]["click"] = True
                    #---获得坐标
                    gesture_lines_dict[id_]["pts"].append(dict_["choose_pt"])
                    double_en_pts.append(dict_["choose_pt"])
                else:
                    if  id_ not in gesture_lines_dict.keys():
                        gesture_lines_dict[id_] = {}
                        gesture_lines_dict[id_]["pts"]=[]
                        gesture_lines_dict[id_]["line_color"] = (random.randint(100,255),random.randint(100,255),random.randint(100,255))
                        gesture_lines_dict[id_]["click"] = None
                    elif  id_ in gesture_lines_dict.keys():

                        gesture_lines_dict[id_]["pts"]=[]# 清除轨迹
                        #判断是否上升沿
                        if gesture_lines_dict[id_]["click"] == True:#下降沿计数器
                            info_dict["click_dw_cnt"] += 1
                        # 更新点击状态
                        gesture_lines_dict[id_]["click"] = False

            #绘制手click 状态时的大拇指和食指中心坐标点轨迹
            draw_click_lines(img,gesture_lines_dict,vis = bool(config.vis_gesture_lines))
            # 判断各手的click状态是否稳定，且满足设定阈值
            flag_click_stable = judge_click_stabel(img,handpose_list,int(config.charge_cycle_step))
            # 判断是否启动识别语音,且进行选中目标识别
            img_reco_crop,reco_msg = audio_recognize(img,algo_img,img_reco_crop,object_recognize_model,info_dict,double_en_pts,flag_click_stable)

            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0),5)
            cv2.putText(img, 'HandNum:[{}]'.format(len(hand_bbox)), (5,25),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))

            #cv2.namedWindow("image",0)
            out.write(img)
            cv2.imshow("image",img)
            if cv2.waitKey(1) == 27:
                info_dict["break"] = True
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def main_handpose_x(cfg):  # 配置文件中主要是模型路径，图片大小等
    print(" loading handpose_x local demo ...")
    g_info_dict = Manager().dict()  # 多进程共享字典初始化：用于多进程间的 key：value 操作 给g_info_dict字典各状态初始为False
    g_info_dict["handpose_procss_ready"] = False # 进程间的开启同步信号
    g_info_dict["break"] = False # 进程间的退出同步信号
    g_info_dict["double_en_pts"] = False  # 双手选中动作使能信号

    g_info_dict["click_up_cnt"] = 0
    g_info_dict["click_dw_cnt"] = 0

    g_info_dict["reco_msg"] = None

    print(" multiprocessing dict key:\n")
    for key_ in g_info_dict.keys():
        print( " -> ",key_)
    print()

    #-------------------------------------------------- 初始化各进程
    process_list = []
    t = Process(target=handpose_x_process, args=(g_info_dict, cfg,))
    process_list.append(t)

    t = Process(target=audio_process_recognize_up_edge,args=(g_info_dict,)) # 上升沿播放
    process_list.append(t)

    for i in range(len(process_list)):
        process_list[i].start()

    for i in range(len(process_list)):
        process_list[i].join()# 设置主线程等待子线程结束

    del process_list
