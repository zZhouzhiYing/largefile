#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

#开始使用的时候必须目标中只有一个鱼漂，而不能有2个，否则list会报错（2个的话需要和前面图片中的鱼漂做iou才能确定哪个是新的鱼漂，上来就2个的话做不了iou直接报错）
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
import pyaudio
import audioop
import math
import collections 
from collections import deque

pyautogui.FAILSAFE = True

def listen():
    print('Well, now we are listening for loud sounds...')
    CHUNK = 1024  # CHUNKS of bytes to read each time from mic
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    THRESHOLD = 1000  # The threshold intensity that defines silence
    # and noise signal (an int. lower than THRESHOLD is silence).
    SILENCE_LIMIT = 1  # Silence limit in seconds. The max ammount of seconds where
    # only silence is recorded. When this time passes the
    # recording finishes and the file is delivered.
    # Open stream
    p = pyaudio.PyAudio()
    for index in range(0, p.get_device_count()):
        print(p. get_device_info_by_index(index))

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=0,
                    frames_per_buffer=CHUNK)

    success = False
    listening_start_time = time.time()
    while True:
        try:
            cur_data = stream.read(CHUNK)
            audio_data = np.fromstring(cur_data, dtype=np.short)
            temp = np.max(audio_data)
            print(temp)
            if (temp>=THRESHOLD):
                print('I heart something!')
                success = True
                break
            if time.time() - listening_start_time > 20:
                print('I don\'t hear anything already 20 seconds!')
                break
        except IOError:
            break

    # print "* Done recording: " + str(time.time() - start)
    stream.close()
    p.terminate()
    return success

def compute_iou(rec_1,rec_2):
    '''
    rec_1:左上角(rec_1[0],rec_1[1])    右下角：(rec_1[2],rec_1[3])
    rec_2:左上角(rec_2[0],rec_2[1])    右下角：(rec_2[2],rec_2[3])

    '''
    rec_1[0]=int(rec_1[0])
    rec_1[1]=int(rec_1[1])
    rec_1[2]=int(rec_1[2])
    rec_1[3]=int(rec_1[3])
    rec_2[0]=int(rec_2[0])
    rec_2[1]=int(rec_2[1])
    rec_2[2]=int(rec_2[2])
    rec_2[3]=int(rec_2[3])

    s_rec1=(rec_1[2]-rec_1[0])*(rec_1[3]-rec_1[1])   #第一个bbox面积 = 长×宽
    s_rec2=(rec_2[2]-rec_2[0])*(rec_2[3]-rec_2[1])   #第二个bbox面积 = 长×宽
    sum_s=s_rec1+s_rec2                              #总面积
    left=max(rec_1[0],rec_2[0])                      #并集左上角顶点横坐标
    right=min(rec_1[2],rec_2[2])                     #并集右下角顶点横坐标
    bottom=max(rec_1[1],rec_2[1])                    #并集左上角顶点纵坐标
    top=min(rec_1[3],rec_2[3])                       #并集右下角顶点纵坐标
    if left >= right or top <= bottom:               #不存在并集的情况
        return 0
    else:
        inter=(right-left)*(top-bottom)              #求并集面积
        iou=(inter/(sum_s-inter))*1.0                #计算IOU
        return iou

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
    return len(box),box,score

if __name__ == '__main__':
    t=random.uniform(1,2)
    qlist = collections.deque()
    try:
        print('请输入o开始')
        keyboard.wait('o')
        while(True):
            pyautogui.press('6')
            time.sleep(1.5)
            quyu = (420,0,1500,800)
            img = ImageGrab.grab(quyu)
            img.save('yu.png')
            lenn,box,score=yupiaoshibie('./yu.png')    
            # box,score=yupiaoshibie('./assets/a67.png')
            if(lenn==1):
                image = cv2.imread('yu.png')
                cv2.rectangle(image, (int(box[0][0]), int(box[0][1])), (int(box[0][2]), int(box[0][3])), (0, 0, 255), 2)  
                cv2.imwrite('1.png', image)
                qlist.append(box[0])
                x1=box[0][0]
                y1=box[0][1]
                x2=box[0][2]
                y2=box[0][3]
                x=int((int(x2)-int(x1))/2)+int(x1)
                y=int((int(y2)-int(y1))/2)+int(y1)
                x=x+420
                speed=random.uniform(0.5,1.5)
                mouse.move(x, y, multiplier=speed)
                # pyautogui.moveTo(x, y, speed, pyautogui.easeInOutQuad) 
                if not listen():
                    print('If we didn\' hear anything, lets try again')
                mouse.right_click()
                time.sleep(t)
            if(lenn==2):
                #此处分别计算第二次截图中两个鱼漂box和第一次鱼漂box的iou，iou大的即为第一次的鱼漂坐标，则选iou小的那个作为输出并保留其box（放进jilu里取代之前的）
                image = cv2.imread('yu.png')
                cv2.rectangle(image, (int(box[0][0]), int(box[0][1])), (int(box[0][2]), int(box[0][3])), (0, 0, 255), 2)  
                cv2.rectangle(image, (int(box[1][0]), int(box[1][1])), (int(box[1][2]), int(box[1][3])), (0, 0, 255), 2)  
                cv2.imwrite('1.png', image)
                i0=compute_iou(box[0],qlist[0])
                i1=compute_iou(box[1],qlist[0])
                qlist.popleft()
                if(i0<=i1):
                    qlist.append(box[0])
                    x1=box[0][0]
                    y1=box[0][1]
                    x2=box[0][2]
                    y2=box[0][3]
                    x=int((int(x2)-int(x1))/2)+int(x1)
                    y=int((int(y2)-int(y1))/2)+int(y1)
                    x=x+420
                    speed=random.uniform(0.5,1.5)
                    mouse.move(x, y, multiplier=speed)
                    # pyautogui.moveTo(x, y, speed, pyautogui.easeInOutQuad) 
                    if not listen():
                        print('If we didn\' hear anything, lets try again')
                    mouse.right_click()
                    time.sleep(t)
                else:
                    qlist.append(box[1])
                    x1=box[1][0]
                    y1=box[1][1]
                    x2=box[1][2]
                    y2=box[1][3]
                    x=int((int(x2)-int(x1))/2)+int(x1)
                    y=int((int(y2)-int(y1))/2)+int(y1)
                    x=x+420
                    speed=random.uniform(0.5,1.5)
                    mouse.move(x, y, multiplier=speed)
                    # pyautogui.moveTo(x, y, speed, pyautogui.easeInOutQuad) 
                    if not listen():
                        print('If we didn\' hear anything, lets try again')
                    mouse.right_click()                    
                    time.sleep(t)
            if(lenn>2):
                print('gg')
                keyboard.wait('space')

    except SyntaxError:
        print('error')
        pass

  