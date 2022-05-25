'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-10-27 上午12:56
@desc: 基于Yolov5的推理预测box结合grabcut对目标进行图像分割
'''

import os
import cv2
from yolov5_master.models.experimental import attempt_load
from yolov5_master.utils import torch_utils, GrabCut
from yolov5_master.utils.augmentations import letterbox
from yolov5_master.utils.datasets import *
from yolov5_master.utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_requirements
from yolov5_master.utils.plots import plot_one_box, colors, plot_one_box_circle
from yolov5_master.utils.torch_utils import load_classifier


def detector(frame, model, device, conf_threshold=0.4,half=True):
    '''
    检测函数主体
    :param frame: 图像
    :param model: 模型
    :param device: 设备类型
    :param conf_threshold: 置信度阈值
    :param half: 是否使用F16精度推理
    :return:
    '''
    img_size = 640
    frame = cv2.resize(frame, (640, 480))
    img0 = frame
    imgcut = frame.copy()
    img = letterbox(img0, new_shape=img_size)[0]

    img = img[:, :, :].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    hide_labels = False
    hide_conf = False
    line_thickness = 4

    with torch.no_grad():
        # 前向推理
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=0.1)

        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                info_list = []
                points_list = []
                boxes = []
                # 保存预测结果
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).view(-1).tolist()
                    info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls)]
                    box = info[:4]
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    # 画框预览效果
                    plot_one_box(xyxy, frame, label=label, color=colors(c, True), line_thickness=line_thickness)
                    # 绘制中心点
                    x_pt,y_pt = plot_one_box_circle(xyxy, frame, label=label, color=colors(c, True), line_thickness=line_thickness)

                    h,w,c = frame.shape
                    if (h > 4000 or w > 4000):
                        frame = cv2.resize(frame, (640, 480))

                    cv2.imshow('frame',frame)
                    info_list.append(info)
                    boxes.append(box)
                    points_list.append((int(x_pt),int(y_pt)))

                idxs = info[4]

                # 绘制中心点连线
                for i in range(0,len(points_list)):
                    pt1 = points_list[i]
                    if i == len(points_list)-1:
                        pt2 = points_list[i]
                    else:
                        pt2 = points_list[i+1]
                        delta_y = abs(pt2[1] - pt1[1])
                        delta_x = abs(pt2[0] - pt1[0])
                        if delta_x == 0:
                            delta_x = 0.00000001
                        line_k = delta_y / delta_x
                        print('line_k', line_k, 'delta_x', delta_x, 'delta_y', delta_y)
                        if line_k > 1 and delta_x < 100:
                            cv2.line(frame,pt1,pt2,color=colors(i, True),thickness=4, lineType=2)
                            cv2.imshow('frame', frame)
                            cv2.waitKey(0)

                return info_list
            else:
                return None


def start_log():
    print('开始检测和分割')
    print('正在进行模型读取,请稍候...')

if __name__ == '__main__':
    start_log()
    # 参数设置
    # weights = 'yolov5s.pt'
    weights = '/media/hxzh02/SB@home/hxzh/MyGithub/Yolov5_Efficient/yolov5_master/weights/yolov5s.pt'
    # 设置图片路径
    imgdir = '/media/hxzh02/SB@home/hxzh/MyGithub/Yolov5_Efficient/yolov5_master/data/images'
    # 输出xml标注文件
    outdir = '/media/hxzh02/SB@home/hxzh/MyGithub/Yolov5_Efficient/yolov5_master/dst'

    if (os.path.exists(imgdir)):
        # 选择设备类型
        device = torch_utils.select_device(device='0')
        half = device.type != 'cpu'
        # Load model
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
        w = weights[0] if isinstance(weights, list) else weights
        classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if pt:
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            # 获取类别名字
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                # 设置Float16
                model.half()  # to FP16
            if classify:  # second-stage classifier
                # 设置第二次分类，默认不使用
                modelc = load_classifier(name='resnet50', n=2)  # initialize
                modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        elif onnx:
            check_requirements(('onnx', 'onnxruntime'))

            import onnxruntime

            session = onnxruntime.InferenceSession(w, None)

        names = model.module.names if hasattr(model, 'module') else model.names
        IMAGES_LIST = os.listdir(imgdir)

        for image_name in IMAGES_LIST:
            # print(image_name)
            # 判断后缀只处理jpg文件
            if image_name.endswith('.jpg') or image_name.endswith('.JPG'):
                image = cv2.imread(os.path.join(imgdir, image_name))
                # 进行检测并将预测信息存入list
                conf_threshold = 0.4
                coordinates_list = detector(image, model, device,conf_threshold,half)
                # print(coordinates_list)
                if coordinates_list:
                    print(image_name)
                    # for coordinate in coordinates_list:
                    #     label_id = coordinate[4]
                        # print(label_id)
                        # print(names[label_id])

        # image = cv2.imread(os.path.join(imgdir, imgdir))
        # # 进行检测并将预测信息存入list
        # conf_threshold = 0.4
        # coordinates_list = detector(image, model, device, conf_threshold, half)
        # # print(coordinates_list)
        # if coordinates_list:
        #     print(imgdir)
        #     for coordinate in coordinates_list:
        #         label_id = coordinate[4]
        #         print(label_id)
        #         # print(names[label_id])

    else:
        print('imgdir not exist!')


