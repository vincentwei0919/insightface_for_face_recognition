# -*- coding: utf-8 -*-


import face_model

import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import time
from PIL import Image

class Args():
    def __init__(self):
        self.image_size = '112,112'
        self.gpu = 0
        self.model = './models/model-r50-am-lfw/model,0000'
        self.ga_model = './models/gamodel-r50/model,0000'
        self.threshold = 1.24
        self.flip = 0
        self.det = 0


args = Args()
model = face_model.FaceModel(args)

path  = ''
dst_path = ''
# imgs = os.listdir(path)
cnt = 1


for root, dirs, files in os.walk(path):
    for dir in dirs:
        img_root = os.path.join(root, dir)
        images = os.listdir(img_root)
        for image in images:
            print(os.path.join(img_root, image))
            img = cv2.imread(os.path.join(img_root, image))
            out, points = model.get_input(img)  # 3x112x112
            new_image = np.transpose(out, (1, 2, 0))[:, :, ::-1]
            out = Image.fromarray(new_image)
            out = out.resize((112, 112))
            out = np.asarray(out)

        # for point in points:
        #     cv2.circle(out, (point[0], point[1]), 2, (0, 0, 255), -1)
            #     cv2.putText(image, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            if not os.path.exists(os.path.join(dst_path, dir)):
                os.mkdir(os.path.join(dst_path, dir))
            cv2.imwrite(os.path.join(dst_path, dir, str(cnt) + '.jpg'), out)
            cnt += 1
