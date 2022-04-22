# -*- coding:utf-8  -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-10-16 上午09:56
@desc: 自动标注xml信息
       通过训练好的模型对新样本图片进行预测并生成xml信息
'''

import os
from os import getcwd
from xml.etree import ElementTree as ET
# from lxml import etree as ET
# from yolov5_master.detect import detect_parse_opt
from yolov5_master.models.experimental import attempt_load
from yolov5_master.utils import torch_utils
from yolov5_master.utils.datasets import *
from yolov5_master.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5_master.utils.plots import plot_one_box, colors, plot_one_box_circle
from yolov5_master.utils.torch_utils import load_classifier


def mk(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("make dirs in %s" % (path))
    else:
        print("There are %d files in %s" % (len(os.listdir(path)), path))

def detector(frame, model, device, conf_threshold=0.4,half=True, debug = True):
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
    img0 = frame
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
        # pred = model(img, augment=opt['augment'])[0]
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=conf_threshold, iou_thres=0.1)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                info_list = []
                # 保存预测结果
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).view(-1).tolist()
                    info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls)]

                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    if debug:
                        # 画框预览效果
                        plot_one_box(xyxy, frame, label=label, color=colors(c, True), line_thickness=line_thickness)
                        cv2.imshow('frame',frame)
                        cv2.waitKey(0)

                    info_list.append(info)
                return info_list
            else:
                return None


def create_object(root, xi, yi, xa, ya, obj_name):
    '''
    定义一个创建一级分支object的函数
    :param root:树根
    :param xi:xmin
    :param yi:ymin
    :param xa:xmax
    :param ya:ymax
    :param obj_name:
    :return:
    '''
    # 创建一级分支object
    _object = ET.SubElement(root, 'object')
    # 创建二级分支
    name = ET.SubElement(_object, 'name')
    # print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # 创建bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


def create_tree(image_name, h, w, imgdir):
    '''
    创建xml文件的函数
    :param image_name:
    :param h:
    :param w:
    :param imgdir:
    :return:
    '''
    global annotation
    # 创建树根annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    folder = ET.SubElement(annotation, 'folder')
    # 添加folder标签内容
    folder.text = (imgdir)

    # 创建一级分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # 创建一级分支path
    path = ET.SubElement(annotation, 'path')

    path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用于返回当前工作目录

    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 创建一级分支size
    size = ET.SubElement(annotation, 'size')
    # 创建size下的二级分支图像的宽、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 创建一级分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

def start_log():
    print('开始自动标注')
    print('正在进行模型读取,请稍候...')

def pretty_xml(element, indent, newline, level=0):  # ，参数indent用于缩进，newline用于换行
    '''
    xml格式美化修饰
    :param element: elemnt为传进来的Elment类
    :param indent: indent用于缩进
    :param newline: newline用于换行
    :param level: level设置层级
    :return:
    '''
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def auto_label_main(weights, imgdir, outdir):
    '''
    自动标注主运行函数
    :param weights:
    :param imgdir:
    :param outdir:
    :return:
    '''
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
            print(image_name)

            # 判断后缀只处理jpg文件
            if image_name.endswith('.jpg') or image_name.endswith('.JPG'):
                image = cv2.imread(os.path.join(imgdir, image_name))
                # 进行检测并将预测信息存入list
                conf_threshold = 0.4
                coordinates_list = detector(image, model, device,conf_threshold,half,False)

                (h, w) = image.shape[:2]
                create_tree(image_name, h, w, imgdir)
                if coordinates_list:
                    print(image_name)
                    for coordinate in coordinates_list:
                        label_id = coordinate[4]
                        create_object(annotation, int(coordinate[0]), int(coordinate[1]), int(coordinate[2]),
                                      int(coordinate[3]), names[label_id])

                    # 将树模型写入xml文件
                    tree = ET.ElementTree(annotation)
                    root = tree.getroot()
                    pretty_xml(root, '\t', '\n')

                    # 设置去除文件后缀名，避免与.xml冲突
                    # image_name = image_name.strip('.JPG')
                    # image_name = image_name.strip('.jpg')
                    # print('pp',image_name)

                    # Windows
                    if outdir.find('\\') != -1:
                        print('image_name', image_name)
                        tree.write('{}\{}.xml'.format(outdir, image_name), encoding='utf-8')
                    # Mac、Linux、Unix
                    if outdir.find('/') != -1:
                        print('image_name', image_name)
                        tree.write('{}/{}.xml'.format(outdir, image_name), encoding='utf-8')

                else:
                    print(image_name)
    else:
        print('imgdir not exist!')



if __name__ == '__main__':
    start_log()
    # 参数设置
    weights = '/media/hxzh02/SB@home/hxzh/MyGithub/Yolov5_Efficient/yolov5_master/weights/yolov5s.pt'
    # 设置图片路径
    imgdir = '/media/hxzh02/Double/数据集/images/'
    # 输出xml标注文件
    outdir = '/media/hxzh02/Double/数据集/xml'

    auto_label_main(weights, imgdir, outdir)



