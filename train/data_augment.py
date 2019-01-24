# coding:utf-8

import os
import tensorflow as tf
import cv2
import numpy as np

def data_aug(data_dir):
    folders = os.listdir(data_dir)
    for folder in folders:
        face_folder = os.path.join(data_dir, folder)
        face_imgs = os.listdir(face_folder)
        for face in face_imgs:
            face_root_path = os.path.join(face_folder, face)
            pic = cv2.imread(face_root_path)
            R = pic[:,:,2]
            G = pic[:,:,1]
            B = pic[:,:,0]
            rgb_pic = cv2.merge([R, G, B])
