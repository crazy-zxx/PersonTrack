# -*- coding: utf-8 -*-
"""
下载完VOC2007的两个数据集压缩包后，解压合并两个文件夹
将该文件放在数据集文件夹下，通过命令运行该脚本即可将数据集转为YOLOv5格式
"""

import xml.etree.ElementTree as ET
import os
from os import getcwd

# 三个数据集名称
sets = ['train', 'val', 'test']

# 指定需要训练识别的类：person
classes = ['person']

# 获取当前目录的绝对路径
abs_path = os.getcwd()


# 转换标记的区域尺寸
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


# 转换标记文件格式
def convert_annotation(image_id):
    # 读取VOC的xml标记文件
    in_file = open('./Annotations/%s.xml' % (image_id), encoding='UTF-8')
    # 输出为yolov5的txt格式文件
    out_file = open('./labels/%s.txt' % (image_id), 'w')

    # 解析xml结构
    tree = ET.parse(in_file)
    root = tree.getroot()
    # 获取字段值
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        # 保存到txt文件
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 创建标签文件保存的文件夹labels
if not os.path.exists('./labels/'):
    os.makedirs('./labels/')

# 遍历VOC的'train', 'val', 'test'三个文件
for image_set in sets:
    # 读取VOC的train, val, test文件列表文件
    image_ids = open('./ImageSets/Main/%s.txt' %
                     (image_set)).read().strip().split()
    # 保存为yolov5的列表文件
    list_file = open('./%s.txt' % (image_set), 'w')
    # 处理列表中每个文件
    for image_id in image_ids:
        # 在yolov5列表文件中写入每个图片文件路径
        list_file.write(abs_path + '/images/%s.jpg\n' % (image_id))
        # 转换标记文件
        convert_annotation(image_id)
    list_file.close()

os.rename('./JPEGImages', './images')
