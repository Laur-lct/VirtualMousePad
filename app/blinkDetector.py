import numpy as np
import cv2
import time
import math
import caffe

# we want to use cpu
caffe.set_mode_cpu()


# performs eye state detection and maps the state to blink events
class BlinkDetector:
    def __init__(self, nn_definition_file, nn_weights_file):
        self.__net = caffe.Net(nn_definition_file, nn_weights_file, caffe.TEST)
        self.__transformer = caffe.io.Transformer({'data': self.__net.blobs['data'].data.shape})

        # prepare the net
        self.__net.forward()

    # outputs 'openness' probabilities for eye images
    def __predictEyeStates(self, leftEyeImg, rightEyeImg):
        # prepare images
        leftEyeImg = cv2.equalizeHist(cv2.resize(leftEyeImg, (32, 32)))
        rightEyeImg = cv2.equalizeHist(cv2.resize(rightEyeImg, (32, 32)))

        # transform scale and unit variance
        img1 = cv2.subtract(leftEyeImg.astype('float'), np.mean(leftEyeImg))
        img2 = cv2.subtract(rightEyeImg.astype('float'), np.mean(rightEyeImg))
        img1 *= 1.0 / 255
        img2 *= 1.0 / 255

        self.__net.blobs['data'].data[0] = self.__transformer.preprocess('data', img1)
        self.__net.blobs['data'].data[1] = self.__transformer.preprocess('data', img2)

        # millisStart = int(round(time.time() * 1000))
        self.__net.forward()
        # obtain the output probabilities
        output_probL = self.__net.blobs['softmax'].data[0][1]
        output_probR = self.__net.blobs['softmax'].data[1][1]

        # print "Classification delay ", int(round(time.time() * 1000)) - millisStart, ' ms'

        return output_probL, output_probR

    def predict_states(self, grayImg, leftEyeArea, rightEyeArea):
        roiL = grayImg[leftEyeArea[1]:leftEyeArea[3], leftEyeArea[0]:leftEyeArea[2]]
        roiR = grayImg[rightEyeArea[1]:rightEyeArea[3], rightEyeArea[0]:rightEyeArea[2]]
        return self.__predictEyeStates(roiL.copy(), roiR.copy())
