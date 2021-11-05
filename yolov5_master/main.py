
import cv2
import sys
import os

class PackageProjectUtil:
    @staticmethod
    def project_root_path(project_name=None, print_log=True):
        """
        获取当前项目根路径
        :param project_name: 项目名称
        :param print_log: 是否打印日志信息
        :return: 指定项目的根路径
        """
        p_name = 'Yolov5_Efficient' if project_name is None else project_name
        project_path = os.path.abspath(os.path.dirname(__file__))
        # Windows
        if project_path.find('\\') != -1: separator = '\\'
        # Mac、Linux、Unix
        if project_path.find('/') != -1: separator = '/'

        root_path = project_path[:project_path.find(f'{p_name}{separator}') + len(f'{p_name}{separator}')]
        if print_log: print(f'当前项目名称：{p_name}\r\n当前项目根路径：{root_path}')
        return root_path

# 将当前项目目录添加至Python编译器路径(兼容python命令行运行方式)
sys.path.append(PackageProjectUtil.project_root_path())

# 当前目录
rpath = sys.path[0]

from yolov5_master.train import train_main,train_parse_opt
from yolov5_master.detect import detect_main,detect_parse_opt

def train_(object_name):
    # 初始化参数列表
    t_opt = train_parse_opt()
    """
    重设自定义参数,进行模型训练
    Usage-命令行使用方式:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 数据集配置文件
    t_opt.data = rpath + '/data/coco128.yaml'
    # 模型配置文件
    t_opt.cfg = rpath  + '/models/yolov5s.yaml'
    # 预训练权重
    # weights/yolov5l.pt,yolov5l6.pt,yolov5m.pt,yolov5m6.pt,yolov5s6.pt,yolov5x.pt,yolov5x6.pt
    t_opt.weights = rpath + '/weights/yolov5s.pt'
    # 设置单次训练所选取的样本数
    t_opt.batch_size = 16
    # 设置训练样本训练的迭代次数
    t_opt.epochs = 500
    # 设置线程数
    t_opt.workers = 4
    # 训练结果的文件名称
    t_opt.name = object_name

    """开始训练"""
    train_main(t_opt)

def detect_(object_name):
    # 初始化参数列表
    d_opt = detect_parse_opt()
    """
    重设自定义参数,进行预测推理
    Usage-命令行使用方式:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 图像/图像集合/视频的源路径,内部自动文件类型进行判断
    d_opt.source = rpath +'/data/images/bus.jpg'
    # 设置进行预测推理使用的权重模型文件
    d_opt.weights = rpath + '/runs/train/' + object_name + '/weights/best.pt'
    # 设置是否需要预览
    d_opt.view_img = False
    # 置信度设置
    d_opt.conf_thres = 0.5
    # 边界框线条粗细
    d_opt.line_thickness = 4
    cv2.waitKey()
    """开始预测推理"""
    detect_main(d_opt)

if __name__ == '__main__':
    # 设置训练任务/生成模型名称
    object_name = 'coco128_yolov5s'
    # 模型训练
    train_(object_name)

    # 模型预测
    # detect_(object_name)
