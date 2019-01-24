# coding: utf-8
# YuanYang
import math
import cv2
import numpy as np


def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    """
      boxs  [1x_1, 1y_1, 1x_2, 1y_2, score1]
            [2x_1, 2y_1, 2x_2, 2y_2, score2]
            .....
            [nx_1, ny_1, nx_2, ny_2, scoren]
      """
    # grab the coordinates of the bounding boxes

    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    """
        x1 = [1x_1, 2x_1, 3x_1, ..., nx_1]
        y1 = [1y_1, 2y_1, 3y_1, ..., ny_1] 
        x2 = [1x_2, 2x_2, 3x_2, ..., nx_2]
        y2 = [1y_2, 2y_2, 3y_2, ..., ny_2]
        score = [score1, score2, ..., scoren]
    """

    # 计算每个人脸框的面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 给出从小到大排序后score在原先列表中的索引
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        # 最大的面积在最后一个，其他的框与它求交并比
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return pick

def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5)*0.0078125
    return out_data


def generate_bbox(map, reg, scale, threshold):
     """
        # map 表示每个检测到人脸位置的置信度score，reg表示检测到人脸的坐标框 ，scale表示当前的尺度，threshold表示产生box的阈值
         generate bbox from feature map
     Parameters:
     ----------
         map: numpy array , n x m x 1
             detect score for each position
         reg: numpy array , n x m x 4
             bbox
         scale: float number
             scale of this detection
         threshold: float number
             detect threshold
     Returns:
     -------
         bbox array
     """
     stride = 2
     cellsize = 12
    # 找到得分大于阈值的人脸索引，该索引表示结果是人脸
     t_index = np.where(map>threshold)

     # find nothing
     if t_index[0].size == 0:
         return np.array([])

     dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

     reg = np.array([dx1, dy1, dx2, dy2])
     score = map[t_index[0], t_index[1]]
     # 由于在卷积时进行了一次下采样，那么获得到的人脸框需要重新映射回原图
     boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                              np.round((stride*t_index[0]+1)/scale),
                              np.round((stride*t_index[1]+1+cellsize)/scale),
                              np.round((stride*t_index[0]+1+cellsize)/scale),
                              score,
                              reg])

     return boundingbox.T


def detect_first_stage(img, net, scale, threshold):
    """
        run PNet for first stage
    
    Parameters:
    ----------
        img: numpy array, bgr order
            input image
        scale: float number
            how much should the input image scale
        net: PNet
            worker
    Returns:
    -------
        total_boxes : bboxes
    """
    height, width, _ = img.shape
    # 缩放一次之后的图像长、宽
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))
    
    im_data = cv2.resize(img, (ws,hs))
    
    # adjust for the network input
    # 将图像进行通道扩充，并进行中心化，从而可以喂给网络[batchs, channels, rows, cols]
    input_buf = adjust_input(im_data)
    output = net.predict(input_buf)
    # output的形式为：[置信度、人脸框位置]
    # output[1] 表示每个检测到人脸位置的置信度score，output[0]表示检测到人脸的坐标框 ，scale表示当前的尺度，threshold表示产生box的阈值
    boxes = generate_bbox(output[1][0,1,:,:], output[0], scale, threshold)
    """
    boxs  [x1, y1, x2, y2, score, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
          [x1, y1, x2, y2, score, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
          .....
          [x1, y1, x2, y2, score, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
    """

    if boxes.size == 0:
        return None

    # nms
    pick = nms(boxes[:,0:5], 0.5, mode='Union')
    boxes = boxes[pick]
    return boxes

def detect_first_stage_warpper(args):
    return detect_first_stage(*args)
