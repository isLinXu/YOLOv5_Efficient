
import numpy as np
import cv2
from skimage import segmentation

def runGrabCut(_image, boxes, indices):
    imgs = []
    image = _image.copy()
    # ensure at least one detection exists
    indices = len(boxes)
    if indices > 0:
        # loop over the indices we are keeping
        for i in range(0,indices):
            image = _image.copy()
            mask = np.zeros(_image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgbModel = np.zeros((1, 65), np.float64)
            # extract the bounding box coordinates
            rect = (int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]))
            # outline = segmentation.slic(_image, n_segments=100,enforce_connectivity=False)
            print('rect',rect)
            # print(boxes)

            # apply GrabCut
            cv2.grabCut(image, mask, rect, bgdModel, fgbModel, 3, cv2.GC_INIT_WITH_RECT)

            # 0和2做背景
            grab_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            # 使用蒙板来获取前景区域
            image = image * grab_mask[:, :, np.newaxis]
            # regions = outline*grab_mask
            # segmented = np.unique(regions)
            # print('segmented',segmented)
            # cv2.imshow('segmented',segmented)
            # segmented = segmented[1:len(segmented)]
            # pxtotal = np.bincount(outline.flatten())
            # pxseq = np.bincount(regions.flatten())
            #
            #
            # pxseg = np.bincount(regions.flatten())
            # seg_mask = np.zeros(_image.shape[:2], np.uint8)
            # label = (pxseg[segmented] / pxtotal[segmented].astype(float)) < 0.75
            # for j in range(0, len(label)):
            #     if label[j] == 0:
            #         temp = outline == segmented[j]
            #         seg_mask = seg_mask + temp
            # mask = seg_mask > 0
            # mask = np.where((mask == 1), 255, 0).astype("uint8")
            # mask = cv2.bitwise_not(mask)
            # cv2.imshow('mask', mask)
            # cv2.waitKey()

            imgs.append(image)
            # imgs = image|image
            # cv2.bitwise_xor(image, image)

    return imgs

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-y", "--yolo", required=True,
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.25,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.45,
        help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    import yolo

    img, boxes, idxs = yolo.runYOLOBoundingBoxes(args)

    images = runGrabCut(img, boxes, idxs)

    # show the output images
    #cv.namedWindow("Image", cv.WINDOW_NORMAL)
    #cv.resizeWindow("image", 1920, 1080)
    for i in range(len(images)):
        cv2.imshow("Image{}".format(i), images[i])
        cv2.imwrite("grabcut{}.jpg".format(i), images[i])
    cv2.waitKey(0)

