# -*- coding: utf-8 -*-
# @Time    : 2021-11-03 16:24
# @Author  : Wu You
# @File    : txt2xml.py
# @Software: PyCharm

import os
import cv2
from lxml import etree


# ######txt转xml#########
class Gen_Annotations:
    def __init__(self, filename, path):
        try:
            self.root = etree.Element('annotation')
            foldern = etree.SubElement(self.root, 'folder')
            foldern.text = 'VOC2007'

            filenamen = etree.SubElement(self.root, 'filename')
            filenamen.text = filename
            pathn = etree.SubElement(self.root, 'path')
            pathn.text = path

        except Exception as error:
            print("error in init is %s" % str(error))

    def set_size(self, width, height, depth):
        try:
            sizen = etree.SubElement(self.root, 'size')
            widthn = etree.SubElement(sizen, 'width')

            widthn.text = str(width)
            heightn = etree.SubElement(sizen, 'height')
            heightn.text = str(height)

            depthn = etree.SubElement(sizen, 'depth')
            depthn.text = str(depth)
        except Exception as error:
            print("error in set_size is %s" % str(error))

    def add_object(self, bbox):
        Object = etree.SubElement(self.root, 'object')
        name = etree.SubElement(Object, 'name')
        name.text = bbox[0]
        pose = etree.SubElement(Object, 'pose')
        pose.text = 'Unspecified'
        truncated = etree.SubElement(Object, 'truncated')
        truncated.text = '0'
        Difficult = etree.SubElement(Object, 'Difficult')
        Difficult.text = '0'

        x_min = bbox[1]
        y_min = bbox[2]
        x_max = bbox[3]
        y_max = bbox[4]

        bndbox = etree.SubElement(Object, 'bndbox')
        xminn = etree.SubElement(bndbox, 'xmin')
        xminn.text = str(int(x_min))
        yminn = etree.SubElement(bndbox, 'ymin')
        yminn.text = str(int(y_min))
        xmaxn = etree.SubElement(bndbox, 'xmax')
        xmaxn.text = str(int(x_max))
        ymaxn = etree.SubElement(bndbox, 'ymax')
        ymaxn.text = str(int(y_max))

    def save_xml(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')


def TXT2Voc(txt_path, xml_path, jpg_path):
    frame = cv2.imread(jpg_path)

    width = frame.shape[1]
    height = frame.shape[0]
    depth = frame.shape[2]

    # ####写VOC格式的xml
    anno = Gen_Annotations(xml_path, jpg_path)
    anno.set_size(width, height, depth)

    fd = open(txt_path, "r")  # 打开文件
    lines = fd.readlines()  # 读取行

    ObjNames = {'0': 'crack'}  # 改成自己的类别

    # ##txt文件中name、xmin等之间是空格连接
    obj_list = []
    for line in lines:  # 遍历每一行
        line1 = line.replace('\n', '').replace('\r', '')

        words = line1.split(' ')  # 空格连接
        sizeX = len(words)

        if sizeX < 6:
            objNumber = words[0]  # txt文件中的类别

            ObjName = ObjNames[objNumber]  # 实际类别
            obj_list.append(ObjName)

            center_x = float(words[1]) * width
            center_y = float(words[2]) * height
            bbox_width = float(words[3]) * width
            bbox_height = float(words[4]) * height

            minx = int(center_x - bbox_width // 2)
            maxx = int(center_x + bbox_width // 2)
            miny = int(center_y - bbox_height // 2)
            maxy = int(center_y + bbox_height // 2)

            obj_list.append(minx)
            obj_list.append(miny)
            obj_list.append(maxx)
            obj_list.append(maxy)
            anno.add_object(obj_list)

    anno.save_xml(xml_path)


if __name__ == '__main__':
    input_jpg_dir = 'D:/work files/WeChat Files/wxid_c3rnw9wjo7rv22/FileStorage/File/2021-11/1/'
    input_txt_dir = 'D:/work files/WeChat Files/wxid_c3rnw9wjo7rv22/FileStorage/File/2021-11/2/'
    output_xml_dir = 'D:/work files/WeChat Files/wxid_c3rnw9wjo7rv22/FileStorage/File/2021-11/3/'
    if not os.path.exists(output_xml_dir):
        os.mkdir(output_xml_dir)
    files = os.listdir(input_txt_dir)
    for file in files:
        print(file)
        input_txt_path = os.path.join(input_txt_dir, file)
        jpg_file = file.split('.')[0] + '.jpg'
        input_jpg_path = os.path.join(input_jpg_dir, jpg_file)
        xml_file = file.split('.')[0] + '.xml'
        output_xml_file = os.path.join(output_xml_dir, xml_file)

        TXT2Voc(input_txt_path, output_xml_file, input_jpg_path)
