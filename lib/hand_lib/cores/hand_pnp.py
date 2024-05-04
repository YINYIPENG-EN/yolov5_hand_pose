#-*-coding:utf-8-*-
'''
 DpCas-Light
||||      |||||        ||||         ||       |||||||
||  ||    ||   ||    ||    ||      ||||     ||     ||
||    ||  ||    ||  ||      ||    ||  ||     ||
||    ||  ||   ||   ||           ||====||     ||||||
||    ||  |||||     ||      ||  ||======||         ||
||  ||    ||         ||    ||  ||        ||  ||     ||
||||      ||           ||||   ||          ||  |||||||

/--------------------- HandPose_X ---------------------/
'''
# date:2019-12-10
# Author: Eric.Lee
# function: handpose :rotation & translation

import cv2
import numpy as np
# 人脸外轮廓
def get_face_outline(img_crop,face_crop_region,obj_crop_points,face_w,face_h):
    face_mask = np.zeros((1,27,2),dtype = np.int32)
    for m in range(obj_crop_points.shape[0]):
        if m <=16:
            x = int(face_crop_region[0]+obj_crop_points[m][0]*face_w)
            y = int(face_crop_region[1]+obj_crop_points[m][1]*face_h)
            # face_mask.append((x,y))
            face_mask[0,m,0]=x
            face_mask[0,m,1]=y

    for k in range(16,26):
        m = 42-k
        x = int(face_crop_region[0]+obj_crop_points[m][0]*face_w)
        y = int(face_crop_region[1]+obj_crop_points[m][1]*face_h)
        # face_mask.append((x,y))
        face_mask[0,k+1,0]=x
        face_mask[0,k+1,1]=y
        # print(x,y)
    return face_mask

# 人脸公共模型三维坐标
object_pts = np.float32([
                         [0., 0.4,0.],#掌心
                         [0., 5.,0.],#hand 根部
                         # [-2, 2.5,0.],#thumb 第一指节
                         # [-4, 0.5,0.],#thumb 第二指节
                         [-2.7, -4.5, 0.],# index 根部
                         [0., -5., 0.],# middle 根部
                         [2.6, -4., 0.], # ring 根部
                         [5.2, -3., 0.],# pink 根部
                         ]
                         )

# object_pts = np.float32([[-2.5, -7.45, 0.5],# pink 根部
#
#                          [-1.2, -7.45, 0.5], # ring 根部
#
#
#                          [1.2, -7.5, 0.5],# middle 根部
#
#                          [2.5, -7.45, 0.5],# index 根部
#                          [4.2, -3.45, 0.5],# thumb 第二指节
#                          [2.5, -2.0, 0.5],# thumb 根部
#                          [0.00, -0.0,0.5],#hand 根部
#                          ]
#                          )

# xyz 立体矩形框
# reprojectsrc = np.float32([[3.0, 11.0, 2.0],
#                            [3.0, 11.0, -4.0],
#                            [3.0, -7.0, -4.0],
#                            [3.0, -7.0, 2.0],
#                            [-3.0, 11.0, 2.0],
#                            [-3.0, 11.0, -4.0],
#                            [-3.0, -7.0, -4.0],
#                            [-3.0, -7.0, 2.0]])

reprojectsrc = np.float32([[5.0, 8.0, 2.0],
                           [5.0, 8.0, -2.0],
                           [5.0, -8.0, -2.0],
                           [5.0, -8.0, 2.0],
                           [-5.0, 8.0, 2.0],
                           [-5.0, 8.0, -2.0],
                           [-5.0, -8.0, -2.0],
                           [-5.0, -8.0, 2.0]])

# reprojectsrc = np.float32([[6.0, 4.0, 2.0],
#                            [6.0, 4.0, -4.0],
#                            [6.0, -3.0, -4.0],
#                            [6.0, -3.0, 2.0],
#                            [-6.0, 4.0, 2.0],
#                            [-6.0, 4.0, -4.0],
#                            [-6.0, -3.0, -4.0],
#                            [-6.0, -3.0, 2.0]])

# reprojectsrc = np.float32([[6.0, 6.0, 6.0],
#                            [6.0, 6.0, -6.0],
#                            [6.0, -6.0, -6.0],
#                            [6.0, -6.0, 6.0],
#                            [-6.0, 6.0, 6.0],
#                            [-6.0, 6.0, -6.0],
#                            [-6.0, -6.0, -6.0],
#                            [-6.0, -6.0, 6.0]])

# 立体矩形框连线，连接组合
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_hand_pose(shape,img,vis = True):
    h,w,_=img.shape
    K = [w, 0.0, w//2,
         0.0, w, h//2,
         0.0, 0.0, 1.0]
    # Assuming no lens distortion
    D = [0, 0, 0.0, 0.0, 0]

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)# 相机矩阵
    # dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)#相机畸变矩阵，默认无畸变
    dist_coeffs = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    # 选取的人脸关键点的二维图像坐标
    # image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
    #                         shape[39], shape[42], shape[45],
    #                         shape[27],shape[31], shape[35],shape[30],shape[33]])

    image_pts = np.float32([shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]
                            ]
                            )

    # PNP 计算图像二维和三维实际关系，获得旋转和偏移矩阵
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # _, rotation_vec, translation_vec = cv2.solvePnPRansac(object_pts, image_pts, cam_matrix, dist_coeffs)


    # print("translation_vec:",translation_vec)
    #print('translation_vec : {}'.format(translation_vec))

    # 映射矩形框
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)#旋转向量转为旋转矩阵
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))# 拼接操作 旋转 + 偏移
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)#欧拉角估计

    if vis:
        for i,line_pair in enumerate(line_pairs):# 显示立体矩形框
            x1 = int(reprojectdst[line_pair[0]][0])
            y1 = int(reprojectdst[line_pair[0]][1])

            x2 = int(reprojectdst[line_pair[1]][0])
            y2 = int(reprojectdst[line_pair[1]][1])

            if line_pair[0] in [0,3,4,7] and line_pair[1] in [0,3,4,7]:
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
            elif line_pair[0] in [1,2,5,6] and line_pair[1] in [1,2,5,6]:
                cv2.line(img,(x1,y1),(x2,y2),(250,150,0),2)
            else:
                cv2.line(img,(x1,y1),(x2,y2),(0,90,255),2)

    return reprojectdst, euler_angle,translation_vec
