#coding:utf-8

import face_model
import argparse
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--data_dir', default='', help='path to dataset.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gallery_dir', default='', help='path to face gallery.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--num_per_class', default=10, type=int, help='image number per class')
args = parser.parse_args()


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


image_set = get_dataset(args.data_dir)
image_list, label_list = get_image_paths_and_labels(image_set)
path_exp = os.path.expanduser(args.data_dir)
classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]

label_strings = [name for name in classes if os.path.isdir(os.path.join(path_exp, name))]

# pdb.set_trace()


# 初始化人脸检测模型
model = face_model.FaceModel(args)

# embedding_array 是一个 类别数*每个类别的图片数*每张图片特征向量长度 的numpy，这里每个类别有50张，可以更改。
embedding_array = np.zeros((len(classes), args.num_per_class, 512))
for i in range(0, len(classes)):
    for j in range(args.num_per_class):
        try:
            pic = cv2.imread(image_list[i*args.num_per_class+j])
            print(image_list[i*args.num_per_class+j])
            if pic is None:
                print(image_list[j])
            else:
                pic = model.get_input(pic)
                feature = model.get_feature(pic)
                embedding_array[i, j, :] = feature
        except:
            print(j,image_list[j])

label_strings = np.array(label_strings)
np.save(os.path.join(args.gallery_dir,'face_gallery.npy'), embedding_array)
np.save(os.path.join(args.gallery_dir,'face_labels.npy'), label_strings)



