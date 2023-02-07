#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import keyboard
import pyautogui
import pyscreenshot as ImageGrab
import argparse
import os
import random
import cv2
import numpy as np
import onnxruntime
import time
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from pyHM import mouse

pyautogui.FAILSAFE = True
def yupiaoshibie(imgpath):
    input_shape = tuple(map(int, ['640', '640']))
    origin_img = cv2.imread(imgpath)

    img, ratio = preprocess(origin_img, input_shape)


    session = onnxruntime.InferenceSession('./diaoyunew.onnx')

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    box=[]
    score=[]
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        count=0
        for i in final_scores:
            if i>0.4:
                box.append(final_boxes[count])
                score.append(final_scores[count])
            count=count+1
    if len(box)>1 or len(score)>1:
        print("error")
        return None,None
    return box,score

if __name__ == '__main__':
    try:
        print('请输入a开始')
        keyboard.wait('a')
        while(True):
            pyautogui.press('6')
            time.sleep(0.5)
            quyu = (420,0,1500,800)
            img = ImageGrab.grab(quyu)
            img.save('yu.png')
            box,score=yupiaoshibie('./yu.png')    
            # box,score=yupiaoshibie('./assets/a67.png')
            x1=box[0][0]
            y1=box[0][1]
            x2=box[0][2]
            y2=box[0][3]
            x=random.randint(int(x1),int(x2))
            y=random.randint(int(y1),int(y2))
            x=int(1920*x/1080)
            y=int(1080*x/800)
            speed=random.uniform(0.5,1.5)
            mouse.move(x, y, multiplier=speed)
            # pyautogui.moveTo(x, y, speed, pyautogui.easeInOutQuad)
            keyboard.wait('space')
            t=random.uniform(1,1.5)
            print('sleep',t)
            time.sleep(t)
    except SyntaxError:
        print('error')
        pass

  