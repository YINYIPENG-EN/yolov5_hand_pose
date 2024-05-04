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
import copy
def compute_iou_tk(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles

    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def tracking_bbox(data,hand_dict,index,iou_thr = 0.5):

    track_index = index
    reg_dict = {}
    Flag_ = True if hand_dict else False
    if Flag_ == False:
        # print("------------------->>. False")
        for bbox in data:
            x_min,y_min,x_max,y_max,score = bbox
            reg_dict[track_index] = (x_min,y_min,x_max,y_max,score,0.,1,1)
            track_index += 1

            if track_index >= 65535:
                track_index = 0
    else:
        # print("------------------->>. True ")
        for bbox in data:
            xa0,ya0,xa1,ya1,score = bbox
            is_track = False
            for k_ in hand_dict.keys():
                xb0,yb0,xb1,yb1,_,_,cnt_,bbox_stanbel_cnt = hand_dict[k_]

                iou_ = compute_iou_tk((ya0,xa0,ya1,xa1),(yb0,xb0,yb1,xb1))
                # print((ya0,xa0,ya1,xa1),(yb0,xb0,yb1,xb1))
                # print("iou : ",iou_)
                if iou_ > iou_thr: # 跟踪成功目标
                    UI_CNT = 1
                    if iou_ > 0.888:
                        UI_CNT = bbox_stanbel_cnt + 1
                    reg_dict[k_] = (xa0,ya0,xa1,ya1,score,iou_,cnt_ + 1,UI_CNT)
                    is_track = True
                    # print("is_track : " ,cnt_ + 1)
            if is_track == False: # 新目标
                reg_dict[track_index] = (xa0,ya0,xa1,ya1,score,0.,1,1)
                track_index += 1
                if track_index >=65535: #索引越界归零
                    track_index = 0

                if track_index>=100:
                    track_index = 0

    hand_dict = copy.deepcopy(reg_dict)

    # print("a:",hand_dict)

    return hand_dict,track_index
