# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:09:02 2021

@author: babymlin
"""
import cv2 as cv
import argparse
import sys
import os
import glob
 
parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera', default='1.jpg')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', default='deploy.prototxt')
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', default='hed_pretrained_bsds.caffemodel')
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
args = parser.parse_args()
 
#! [CropLayenr]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
 
    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
 
        #self.ystart = (inputShape[2] - targetShape[2]) / 2
        #self.xstart = (inputShape[3] - targetShape[3]) / 2
 
 
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
 
        self.yend = self.ystart + height
        self.xend = self.xstart + width
 
        return [[batchSize, numChannels, height, width]]
 
    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]
#! [CropLayer]
 
#! [Register]
cv.dnn_registerLayer('Crop', CropLayer)
#! [Register]
 
# Load the model.
net = cv.dnn.readNet(cv.samples.findFile(args.prototxt), cv.samples.findFile(args.caffemodel))

# base_path = sys.argv[1]
# output_path = sys.argv[2]
# orange_list = glob.glob(base_path+"*.*g")
# if not os.path.exists(output_path):
#     os.mkdir(output_path)

path = input("請輸入批次轉檔路徑：")
orange_list = glob.glob(path + "*.*g")
if not os.path.exists("..\\result"):
    os.mkdir("..\\result")

count = 1
print("預計處理相片數：", len(orange_list))
for img in orange_list:
    print("正在處理第" + str(count) + "張相片：" + img.split('\\')[-1])
    frame=cv.imread(img) # input img
    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(args.width, args.height),
                                   mean=(104.00698793, 116.66876762, 122.67891434),
                                   swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    out = (255*out).astype("uint8")
    cv.imwrite("..\\result\\" + img.split('\\')[-1] + "_hed.png", out)
    count += 1

"""
版权声明：本文为CSDN博主「玖耿」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/jkjj2015/article/details/87714921
"""
