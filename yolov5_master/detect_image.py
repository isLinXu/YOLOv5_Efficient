import cv2
import torch
from PIL import Image

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load('../YOLO/yolov5_master', 'custom', path='../weight/planeall.pt', source='local',force_reload=True)  # local repo
# model = torch.hub.load('../YOLO/yolov5_master', 'custom', path='weight/planeall.pt',force_reload=True)  # force reload

# Images
img1 = Image.open('/home/hxzh02/文档/test_image/towerupdown/DJI_0650_towerupdown.JPG')  # PIL image

# Inference
results = model(img1, size=640)  # includes NMS

# Results
results.print()
results.save()  # or .show()

