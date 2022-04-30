# -*- coding:utf-8 -*-

"""
@ Author: Yutian Lu
@ Contact: physicoada@gmail.com
@ Date: 2022/4/29 下午3:11
@ Software: PyCharm
@ File: YOLOdataset.py
@ Desc: 
"""

# 第一步，进行训练集的划分
import os
from os import getcwd
import random
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import xml.etree.cElementTree as et
from collections import OrderedDict
import yaml

from kmeans import kmeans, avg_iou

# 需要修改的参数：
# 1.数据集的划分
TRAINVAL_PENCENT = 1.0  # 训练集与验证集所占全体比例。 这里没有划分测试集所以前者1.0
TRAIN_PERCENT = 0.9  # 训练集所占的比例，可自己进行调整
IMG_PATH = "/home/hxzh/视频/YOLOV5/VOCData/"  # 【这里要改成数据集的绝对地址】，记的后面要加个/！！
# 2.【类别参数的设置】：（这里输入对应检测类别，取决于label名字，多个加逗号）
CLASS_NAMES = ["person"]
CLASS_NUM = 1

# 需要注意的参数：【无需修改但是可以知道】
ANNOTATION_PATH = XML_LOCATION = './Annotations'  # 存放数据集标签的地址
TXT_LOCATION = './ImageSets/Main'  # 输出数据集划分后位置的地址
# 1.聚类结果地址：
ANCHORS_TXT_PATH = "./anchors.txt"  # anchors文件保存位置
# 2.选取模型配置：
MODELCONFIG = '../models/yolov5s.yaml'
# 3.数据配置：
DATACONFIG = "../data/myyolov5.yaml"


# 第一步，对训练集进行划分，并将xml转换成txt：
def dataset_split(xml_location=XML_LOCATION, txt_location=TXT_LOCATION):
    parser = argparse.ArgumentParser()
    # xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
    parser.add_argument('--xml_path', default=xml_location, type=str, help='input xml label path')
    # 数据集的划分，地址选择自己数据下的ImageSets/Main
    parser.add_argument('--txt_path', default=txt_location, type=str, help='output txt label path')
    opt = parser.parse_args()
    global TRAINVAL_PENCENT, TRAIN_PERCENT
    trainval_percent = TRAINVAL_PENCENT
    train_percent = TRAIN_PERCENT  # 训练集所占比例，可自己进行调整
    xmlfilepath = opt.xml_path
    txtsavepath = opt.txt_path
    total_xml = os.listdir(xmlfilepath)
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    file_trainval = open(txtsavepath + '/trainval.txt', 'w')
    file_test = open(txtsavepath + '/test.txt', 'w')
    file_train = open(txtsavepath + '/train.txt', 'w')
    file_val = open(txtsavepath + '/val.txt', 'w')

    for i in list_index:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


# 第二步，对训练集进行划分，并将xml转换成txt：
def xml_yolotxt(classes_=CLASS_NAMES):
    sets = ['train', 'val', 'test']
    classes = classes_
    abs_path = os.getcwd()
    print(abs_path)

    def convert(size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(image_id):
        in_file = open('Annotations/%s.xml' % (image_id), encoding='UTF-8')
        out_file = open('labels/%s.txt' % (image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            # difficult = obj.find('Difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    wd = getcwd()
    for image_set in sets:
        if not os.path.exists('labels/'):
            os.makedirs('labels/')
        image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()

        if not os.path.exists('dataSet_path/'):
            os.makedirs('dataSet_path/')

        list_file = open('dataSet_path/%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            list_file.write(IMG_PATH + '/images/%s.jpg\n' % (image_id))
            convert_annotation(image_id)
        list_file.close()


# 第三步，形成聚类信息：
# 数据加载函数
def load_data(anno_dir, class_names):
    xml_names = os.listdir(anno_dir)
    boxes = []
    for xml_name in xml_names:
        xml_pth = os.path.join(anno_dir, xml_name)
        tree = et.parse(xml_pth)

        width = float(tree.findtext("./size/width"))
        height = float(tree.findtext("./size/height"))

        for obj in tree.findall("./object"):
            cls_name = obj.findtext("name")
            if cls_name in class_names:
                xmin = float(obj.findtext("bndbox/xmin")) / width
                ymin = float(obj.findtext("bndbox/ymin")) / height
                xmax = float(obj.findtext("bndbox/xmax")) / width
                ymax = float(obj.findtext("bndbox/ymax")) / height

                box = [xmax - xmin, ymax - ymin]
                boxes.append(box)
            else:
                continue
    return np.array(boxes)


# anchor计算过程
def calculate_anchors(anchors_tet_path=ANCHORS_TXT_PATH, annotation_path=ANNOTATION_PATH, class_names_=CLASS_NAMES):
    anchors_txt = open(anchors_tet_path, "w")
    CLUSTERS = 9
    train_boxes = load_data(annotation_path, class_names_)
    count = 1
    best_accuracy = 0
    best_anchors = []
    best_ratios = []

    for i in range(10):  # 可以修改，不要太大，否则时间很长
        anchors_tmp = []
        clusters = kmeans(train_boxes, k=CLUSTERS)
        idx = clusters[:, 0].argsort()
        clusters = clusters[idx]
        # print(clusters)

        for j in range(CLUSTERS):
            anchor = [round(clusters[j][0] * 640, 2), round(clusters[j][1] * 640, 2)]
            anchors_tmp.append(anchor)
            print(f"Anchors:{anchor}")

        temp_accuracy = avg_iou(train_boxes, clusters) * 100
        print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

        ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
        ratios.sort()
        print("Ratios:{}".format(ratios))
        print(20 * "*" + " {} ".format(count) + 20 * "*")

        count += 1

        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_anchors = anchors_tmp
            best_ratios = ratios

    anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
    anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
    anchors_txt.write("Best Ratios = " + str(best_ratios))
    anchors_txt.close()


# 结束：转换anchor为整数类型。
def anchor_changes_int():
    list_ = []
    with open("./anchors.txt") as f2:
        lines = f2.readlines()
    for line in lines:
        list_.append(line)
    # 数组处理
    str_ = list_[1][14:-1]
    list_ = eval(str_)
    new_list = [int(x) for row in list_ for x in row]
    # 拆分成对应的数据
    new_list_p3 = new_list[0:6]
    new_list_p4 = new_list[6:12]
    new_list_p5 = new_list[12:18]
    return new_list_p3, new_list_p4, new_list_p5


# 结束：改写yolo配置文件。
def ordered_yaml_load(yaml_path, Loader=yaml.Loader,
                      object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    with open(yaml_path) as stream:
        return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data, filename, Dumper=yaml.SafeDumper):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    ## 这里是 把 生成文件里的 “null” 转为 “”
    def represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', '')

    stream = None
    with open(filename, "w") as stream:
        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        OrderedDumper.add_representer(type(None), represent_none)
        yaml.dump(data,
                  stream,
                  OrderedDumper,
                  default_flow_style=None,
                  encoding='utf-8',
                  allow_unicode=True)


def yaml_load_dump_model(yaml_file_path=MODELCONFIG, class_num_=CLASS_NUM):
    new_list_p3, new_list_p4, new_list_p5 = anchor_changes_int()
    kv_conf_tmpl = ordered_yaml_load(yaml_file_path)
    kv_conf_tmpl["nc"] = class_num_
    kv_conf_tmpl["anchors"][0] = new_list_p3
    kv_conf_tmpl["anchors"][1] = new_list_p4
    kv_conf_tmpl["anchors"][2] = new_list_p5
    ordered_yaml_dump(kv_conf_tmpl, yaml_file_path)  ###  使用


def yaml_load_dump_data(yaml_file_path=DATACONFIG, class_num_=CLASS_NUM):
    kv_conf_tmpl = ordered_yaml_load(yaml_file_path)
    kv_conf_tmpl["train"] = IMG_PATH + "dataSet_path/train.txt"
    kv_conf_tmpl["val"] = IMG_PATH + "dataSet_path/val.txt"
    kv_conf_tmpl["nc"] = class_num_
    kv_conf_tmpl["names"] = CLASS_NAMES
    ordered_yaml_dump(kv_conf_tmpl, yaml_file_path)


if __name__ == '__main__':
    # 第一步拆分数据集
    dataset_split()
    print("拆分数据集成功！")
    # 第二步转换标签
    xml_yolotxt()
    print("转换标签成功！")
    # 第三步聚类分析
    calculate_anchors()
    print("聚类分析成功！")
    # 第四步配置改写
    yaml_load_dump_model()
    yaml_load_dump_data()
    print("配置改写成功！")
