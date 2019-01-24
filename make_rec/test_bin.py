#coding:utf-8

import pickle
import os
import sys
import mxnet as mx
from mxnet import ndarray as nd
import pdb
import matplotlib.pyplt as plt

img_dir = '/media/dhao/系统/05-weiwei/FR/dataset/validdata'
path ='/media/dhao/系统/05-weiwei/FR/dataset/faces_zhibo_112x112/lfw.bin'
image_size = (112, 112)

img_pairs = read_pairs((os.path.join(img_dir, 'all_in_one.txt'))
lfw_paths, issame_list = get_paths(img_pairs)
lfw_data_list = []
for flip in [0,1]:
lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
lfw_data_list.append(lfw_data)
i = 0
for path in lfw_paths:
with open(path, 'rb') as fin:
  _bin = fin.read()
  img = mx.image.imdecode(_bin)
  img = nd.transpose(img, axes=(2, 0, 1))
  for flip in [0,1]:
    if flip==1:
      img = mx.ndarray.flip(data=img, axis=2)
    lfw_data_list[flip][i][:] = img
  i+=1
  if i%1000==0:
    print('loading lfw', i)
print(lfw_data_list[0].shape)
print(lfw_data_list[1].shape)
return (lfw_data_list, issame_list)
