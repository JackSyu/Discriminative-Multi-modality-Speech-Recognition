# encoding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
import dlib
import imp
import numpy as np
#import face_detector as fd
import skvideo
import argparse
import torch
import cv2
skvideo.setFFmpegPath('/monchickey/ffmpeg/bin')
print(skvideo.getFFmpegPath())
import skvideo.io
from keras import backend as K
from scipy import ndimage
import detect_face
import cv2 as cv

# Thresholds
minsize = 10  # minimum size of face
threshold = [0.6, 0.7, 0.9]  # three steps's threshold
factor = 0.709  # scale factor

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_mouth_frame(shape1, frame, jjj):
    i = -1
    mouth_points = []
    low = 0
    high = 0
    left = 0
    right = 0
    landmarks = np.matrix([[p.x, p.y] for p in shape1.parts()])
    # print"the %d frame's landmarks"%jjj,landmarks.shape
    for part in shape1.parts():
        i += 1

        if i == 34:
            low = part.y
        if i == 9:
            high = part.y
        if i == 5:
            left = part.x
        if i == 13:
            right = part.x
        # xyx modify.

        if i < 48:
            continue
        mouth_points.append((part.x, part.y))

    width = (right - left) // 2
    hight = (high - low) // 2
    # print width, hight
    para = max(width, hight)
    np_mouth_points = np.array(mouth_points)
    mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0).astype(int)
    if mouth_centroid[0]-30 < para and mouth_centroid[1]-30 < para:
        mouth_crop_image = frame[0:mouth_centroid[1] + para, 0:mouth_centroid[0] + para]
    elif mouth_centroid[0]-30 < para:
        mouth_crop_image = frame[mouth_centroid[1] - para-30:mouth_centroid[1] + para,
                           0:mouth_centroid[0] + para]
    elif mouth_centroid[1]-30 < para:
        mouth_crop_image = frame[0:mouth_centroid[1] + para,
                           mouth_centroid[0] - para-30:mouth_centroid[0] + para]
    else:
        mouth_crop_image = frame[mouth_centroid[1] - para-30:mouth_centroid[1] + para,
                           mouth_centroid[0] - para-30:mouth_centroid[0] + para]
    # mouth_crop_image = sk.resize(mouth_crop_image, [112, 112])
    mouth_crop_image = cv.resize(mouth_crop_image, (112, 112), interpolation=cv.INTER_LINEAR)
    return mouth_crop_image


class Video(object):
    def __init__(self, vtype='mouth', face_predictor_path=None, detector=None, pnet=None, rnet=None, onet=None):
        if vtype == 'face' and face_predictor_path is None:
            raise AttributeError('Face video need to be accompanied with face predictor')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype
        self.detector = detector
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet
        self.missing_Detect = None
        #self.error_path = error_path
    def from_frames(self, path):
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        # print "frames_path= ",frames_path
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        self.handle_type(frames)
        return self

    def from_video(self, path):
        # print 'path=',path
        frames = self.get_video_frames(path)
        self.handle_type(frames)
        return self

    def from_array(self, frames):
        self.handle_type(frames)
        return self

    def handle_type(self, frames):
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Video type not found')

    def process_frames_face(self, frames):
        #detector = dlib.get_frontal_face_detector()

        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(self.detector, predictor, frames)
        # self.face = np.array(frames)
        # print"mouth_frames type", mouth_frames.dtype
        # print"mouth_frames type", mouth_frames
        self.mouth = np.array(mouth_frames)

        #print"mouth type", (self.mouth).dtype
        # self.set_data(mouth_frames)

    def process_frames_mouth(self, frames):
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        # self.set_data(frames)

    def get_frames_mouth(self, detector, predictor, frames):
        jjj = 0
        mouth_frames = []
        nxb = 0
        print(frames.shape)
        for frame in frames:
            nxb += 1
            (R, G, B) = cv2.split(frame)
            img_test = cv2.merge((B, G, R))
            #cv2.imwrite('./test_frames/test_{}.jpg'.format(nxb), img_test, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            boundings, points = detector.detect_face(frame, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
            shape = None
                #cv2.imwrite('test_{}.jpg'.format(nxb), img_test, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            for b in boundings:
                # score = bounding[0]
                # print('Bounding: ', bounding)
                #bounding_rec = bounding_rec.astype(np.int32)
                #cv2.rectangle(img_test, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
                #cv2.imwrite('./test_frames/test_{}.jpg'.format(nxb), img_test, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                #if len(bounding) > 2:  # There are landmarks here
                # print('Draw landmarks!')
                #for ptr in bounding[2]:
                #cv2.circle(frame, tuple(ptr), 1, (0, 0, 255), 2)
                d = dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                #dets = detector(frame, 1)
                shape = predictor(frame, d)
            if shape is None:
                detector1 = dlib.get_frontal_face_detector()
                dets = detector1(frame, 1)
                for k, d in enumerate(dets):
                    shape = predictor(frame, d)
            i = -1
            low = 0
            high = 0
            left = 0
            right = 0
            # print"dets type", dets.dtype
            # print"shape type", shape.dtype
            if shape is None:  # Detector doesn't detect face, just return as is
                # return frames
                self.missing_Detect = True
                return frames
                #if not os.path.exists(self.error_path):
                #    os.mkdir(self.error_path)
                #cv2.imwrite(os.path.join(self.error_path, '{}.jpg'.format(nxb)), img_test, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                #startX = 0
                #startY = 0
                #endX = 224
                #endY = 224
                #rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                #shape1 = predictor(frame, rect)
                #mouth_crop_image = get_mouth_frame(shape1, frame, jjj)
                # print"mouth_crop_image type",mouth_crop_image.dtype
                #gray_mouth_crop_image = rgb2gray(mouth_crop_image)
                # print"gray_mouth_crop_image type",gray_mouth_crop_image.dtype
                #gray_mouth_crop_image = gray_mouth_crop_image.astype(np.uint8)
                #mouth_frames.append(gray_mouth_crop_image)
                #cv2.imwrite('./test_mouth_crop/mouth_crop_{}.jpg'.format(nxb), gray_mouth_crop_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                #jjj = jjj + 1
                #continue

            mouth_points = []
            # print"face landmark",shape.parts()
            # landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
            # print"the %d frame's landmarks"%jjj,landmarks.shape
            for part in shape.parts():
                i += 1

                if i == 34:
                    low = part.y
                if i == 9:
                    high = part.y
                if i == 5:
                    left = part.x
                if i == 13:
                    right = part.x
                # xyx modify.

                if i < 48:
                    continue
                mouth_points.append((part.x, part.y))
            cv2.rectangle(img_test, (int(left), int(low)), (int(right), int(high)), (0, 255, 0))
            for i in range(68):
                cv2.circle(img_test, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
            cv2.imwrite('./test_frames/test_{}.jpg'.format(nxb), img_test, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            width = (right - left) // 2
            hight = (high - low) // 2
            # print width, hight
            para = max(width, hight)
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0).astype(int)

            #print"mouth_centroid[0]&para", mouth_centroid[0], para
            #print"mouth_centroid[1]&para", mouth_centroid[1], para

            if mouth_centroid[0] < para and mouth_centroid[1] < para:
                mouth_crop_image = frame[0:mouth_centroid[1] + para, 0:mouth_centroid[0] + para]
            elif mouth_centroid[0] < para:
                mouth_crop_image = frame[mouth_centroid[1] - para:mouth_centroid[1] + para,
                                   0:mouth_centroid[0] + para]
            elif mouth_centroid[1] < para:
                mouth_crop_image = frame[0:mouth_centroid[1] + para,
                                   mouth_centroid[0] - para:mouth_centroid[0] + para]
            else:
                mouth_crop_image = frame[mouth_centroid[1] - para:mouth_centroid[1] + para,
                                   mouth_centroid[0] - para:mouth_centroid[0] + para]

            # print"mouth_crop_image shape",(mouth_crop_image.shape)[1]
            # print"frame type", frame.dtype
            # print"mouth_crop_image type", mouth_crop_image.dtype
            # mouth_crop_image = sk.resize(mouth_crop_image, [112, 112])
            #print"mouth_crop_image",mouth_crop_image.shape
            mouth_crop_image = cv.resize(mouth_crop_image, (112, 112), interpolation=cv.INTER_LINEAR)
            # print"resize mouth_crop_image ", mouth_crop_image
            # print"mouth_crop_image type",mouth_crop_image.dtype
            gray_mouth_crop_image = rgb2gray(mouth_crop_image)
            gray_mouth_crop_image = gray_mouth_crop_image.astype(np.uint8)
            mouth_frames.append(gray_mouth_crop_image)
            #cv2.imwrite('./test_mouth_crop/mouth_crop_{}.jpg'.format(nxb), gray_mouth_crop_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # print"mouth_crop_image type", mouth_crop_image.dtype
            # print"gray_mouth_crop_image type", gray_mouth_crop_image.dtype
            jjj = jjj + 1
            # xyx modify.
        # print"frame sample",mouth_frames[5]
        # print"frame shape",mouth_frames[5].shape
        # print"mouth_frames type", mouth_frames.dtype
        return mouth_frames

    def get_video_frames(self, path):
        # 原来使用skvideo读取视频文件
        videogen = skvideo.io.vread(path)
        # print(videogen)
        # print"videogen type",videogen.dtype
        frames = np.array([frame for frame in videogen])
        #for frame in frames:
            #cv2.imwrite('test_.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        # print "original frames type",frames.dtype
        print("original frames shape",frames.shape)
        return frames

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0, 1)  # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)  # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames)  # T x W x H x C
        if K.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3)  # C x T x W x H
        self.data = data_frames
        self.length = frames_n
