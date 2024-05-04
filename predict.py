import multiprocessing

import torch
import torchvision
import argparse
from applications.handpose_local_app import main_handpose_x
def demo_logo():
    print("\n/*********************************/")
    print("/---------------------------------/\n")
    print("             WELCOME      ")
    print("             yinyipeng    ")
    print("        wechat:y24065939s  ")
    print("\n/---------------------------------/")
    print("/*********************************/\n")
if __name__ == '__main__':
    multiprocessing.freeze_support()
    demo_logo()
    parse = argparse.ArgumentParser("yolov5 hand pose")
    parse.add_argument('--hand_weight', type=str, default='best.pt', help='hand detect model path')
    parse.add_argument('--input_shape', type=int, default=640, help='yolov5 input shape')
    parse.add_argument('--conf', type=float, default=0.5, help='detect conf')
    parse.add_argument('--iou', type=float, default=0.45)
    parse.add_argument('--data', type=str, default='components/hand_detect/data/coco128.yaml')
    parse.add_argument('--handpose_model_path', type=str, default='components/hand_keypoints/weights/ReXNetV1-size-256-wingloss102-0.122.pth', help='hand 21 keys model path')
    parse.add_argument('--handpose_name', type=str, default='rexnetv1', help='handpose arch name')

    parse.add_argument('--classify_model_path', type=str, default='components/classify_imagenet/weights/imagenet_size-256_20210409.pth', help='classify model path')
    parse.add_argument('--classify_model_name', type=str, default='resnet_50', help='classify model name')
    parse.add_argument('--num_class', type=int, default=1000)

    parse.add_argument('--video_path', default='0', help='video path')
    parse.add_argument('--vis_gesture_lines', action='store_false')
    parse.add_argument('--charge_cycle_step', type=int, default=18)
    parse.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    # args = parse.parse_args()
    args, unparsed = parse.parse_known_args()
    print(args)

    main_handpose_x(args)

# 导出exe:pyinstaller --onefile --hidden-import torch._C  --hidden-import torch.nn.functional predict.py

