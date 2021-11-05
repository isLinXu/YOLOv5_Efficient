# Yolov5_Efficient
![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105103947692-337341194.png)

Use yolov5 efficiently(高效地使用Yolo v5)
---
## 1.Introduction-介绍

The repository is reconstructed and annotated based on UltralyTICS / YOLOV5, and other functions are added thereto, such as automatic annotation with the Grab Cut, and the pointing center point.

该存储库基于Ultralytics/yolov5进行重构与注释，并且在此基础上加入其他功能，例如自动标注与漫水填充GrabCut，以及绘制中心点进行连线。

---
## 2.Performance-效果表现

### 2.1 Yolo detect+GrabCut

| detect预测结果 | GrabCut分割结果 |
| -------------- | --------------- |
|  ![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105110409690-1455179243.png)     |      ![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105110418858-940040936.png)           |
|![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105110426763-812179903.png) |![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105110435881-328445319.png)      |

**demo video**

<iframe src="//player.bilibili.com/player.html?aid=336331236&bvid=BV1zR4y177sa&cid=432234631&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>


### 2.2 Yolo center point line 中心点连线
| demo1                                                        | demo2                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105110651730-834327083.png) | ![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105110705075-1370353606.png) |

### 2.3 autolabel半自动标注

<iframe src="//player.bilibili.com/player.html?aid=933566990&bvid=BV1ET4y1o7ZR&cid=425923550&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>



## 3.How to Use-用法

### 3.1 Train 训练自定义数据集

可参考Ultralytics/yolov5的训练用法：

#### 3.1.1命令行方式：

单独使用：

```shell
python yolov5_master/train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

集成使用：

```shell
python yolov5_master/main.py
```
![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105111817968-708151746.png)

#### 3.1.2 IDE方式：
![](https://img2020.cnblogs.com/blog/1571518/202111/1571518-20211105112013475-1447251119.png)

### 3.2 detect 推理预测

#### 3.2.1命令行方式：

单独使用-测试图片

```shell
python yolov5_master/detect.py --source ./testfiles/img1.jpg --weights runs/train/bmyolov5s/weights/best.pt 
```

单独使用-测试视频

```shell
python yolov5_master/detect.py --source ./testfiles/video.mp4 --weights runs/train/bmyolov5s/weights/best.pt 
```

集成使用

```shell
python yolov5_master/main.py
```
集成方式中包含参数设置与训练、预测的配置选择，可自行更换。

### 3.3 Yolo detect+GrabCut使用

直接在IDE中修改配置即可。

### 3.4 Yolo center point line使用

直接在IDE中修改配置即可。

### 3.5 autolabel自动标注

细节在注释中介绍，直接在IDE中修改配置即可。

