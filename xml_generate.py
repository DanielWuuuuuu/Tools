# _*_ coding : utf-8 _*_
# @Time      : 2020/11/25 16:13
# @Author    : WuYou
# @File      : xml_generate.py

from numpy import *
import os
import cv2
import glob
import darknet
import Config
import numpy as np
from lxml import etree


class Gen_Annotations:
    def __init__(self, filename, path):
        try:
            self.root = etree.Element('annotation')
            foldern = etree.SubElement(self.root, 'folder')
            # foldern.text = 'VOC2007'

            filenamen = etree.SubElement(self.root, 'filename')
            filenamen.text = filename
            pathn = etree.SubElement(self.root, 'path')
            pathn.text = path

        except Exception as error:
            print("error in init is %s" % str(error))
        # sourcen = etree.SubElement(self.root, 'source')
        # database = etree.SubElement(sourcen, 'database')
        # database.text = 'Unknown'

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

    def add_object(self, image, results, resize, width, height, channel):
        if results:
            for detection in results:
                # 当前检测框的标签和置信度
                label = detection[0]
                confidence = detection[1]
                confidence = "{}%".format(confidence)

                # bbox信息
                bounds = detection[2]
                x = int(bounds[0] * resize[0])
                y = int(bounds[1] * resize[1])
                w = int(bounds[2] * resize[0])
                h = int(bounds[3] * resize[1])

                # 左上角的坐标
                x_coord = int(x - w / 2)
                y_coord = int(y - h / 2)
                bbox = [[x_coord, y_coord], [x_coord, y_coord + h], [x_coord + w, y_coord + h], [x_coord + w, y_coord]]

                # 矫正bbox
                if bbox[0][0] < 0:
                    bbox[0][0] = 0
                if bbox[0][1] < 0:
                    bbox[0][1] = 0
                if bbox[1][0] < 0:
                    bbox[1][0] = 0
                if bbox[1][1] > image.shape[0]:
                    bbox[1][1] = image.shape[0]
                if bbox[2][0] > image.shape[1]:
                    bbox[2][0] = image.shape[1]
                if bbox[2][1] > image.shape[0]:
                    bbox[2][1] = image.shape[0]
                if bbox[3][0] > image.shape[1]:
                    bbox[3][0] = image.shape[1]
                if bbox[3][1] < 0:
                    bbox[3][1] = 0

                object = etree.SubElement(self.root, 'object')
                namen = etree.SubElement(object, 'name')
                namen.text = label
                posen = etree.SubElement(object, 'pose')
                posen.text = 'Unspecified'
                truncatedn = etree.SubElement(object, 'truncated')
                truncatedn.text = '0'
                Difficultn = etree.SubElement(object, 'Difficult')
                Difficultn.text = '0'

                bndbox = etree.SubElement(object, 'bndbox')
                xminn = etree.SubElement(bndbox, 'xmin')
                xminn.text = str(int(bbox[0][0]))
                yminn = etree.SubElement(bndbox, 'ymin')
                yminn.text = str(int(bbox[0][1]))
                xmaxn = etree.SubElement(bndbox, 'xmax')
                xmaxn.text = str(int(bbox[2][0]))
                ymaxn = etree.SubElement(bndbox, 'ymax')
                ymaxn.text = str(int(bbox[2][1]))

    def save_xml(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')


def check_parameter_errors(config):
    """
    检查参数
    :param config: Yolo参数
    :return:
    """
    assert 0 < config['thresh'] < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config['config_file']):
        raise (ValueError("Invalid cfg file path {}".format(os.path.abspath(config['config_file']))))
    if not os.path.exists(config['data_file']):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(config['data_file']))))
    if not os.path.exists(config['weight']):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(config['weight']))))


class Network:
    def __init__(self):
        """
        初始化模型及参数
        """
        # 检查参数
        check_parameter_errors(Config.Darknet)
        # 网络模型初始化
        self.network, self.class_names, self.class_colors = darknet.load_network(
            Config.Darknet['config_file'],
            Config.Darknet['data_file'],
            Config.Darknet['weight'],
            Config.Darknet['batch_size']
        )
        # 阈值
        self.thresh = Config.Darknet['thresh']
        # 输入路径
        self.input_path = Config.input_path
        if not os.path.exists(self.input_path):
            raise (ValueError("Invalid input path {}".format(os.path.abspath(self.input_path))))

    def load_images(self):
        """
        根据指定的路径返回文件路径列表
        对单张图像的路径：直接返回该路径
        对txt：返回该txt下存储的所有文件路径
        对文件夹：返回该文件夹下的所有文件路径
        :return: 文件路径列表
        """
        input_path_extension = self.input_path.split('.')[-1]
        if input_path_extension in ['jpg', 'jpeg', 'png']:
            return [self.input_path]
        elif input_path_extension == "txt":
            with open(self.input_path, "r") as f:
                return f.read().splitlines()
        else:
            return glob.glob(
                os.path.join(self.input_path, "*.jpg")) + \
                   glob.glob(os.path.join(self.input_path, "*.png")) + \
                   glob.glob(os.path.join(self.input_path, "*.jpeg"))

    def image_detection(self, image):
        """
        检测单张图片
        :param image: cv2格式的图片
        :return: 检测结果[(label, confidence, (x, y, w, h)), ()]，原图尺寸和网络尺寸之比
        """
        # 网络尺寸和原图尺寸
        width = darknet.network_width(self.network)
        height = darknet.network_height(self.network)
        height_ori, width_ori = image.shape[:2]
        if len(image.shape) == 3:
            image_channel = image.shape[2]
        else:
            image_channel = 1

        # 将cv2格式的图片转为Yolo需要的格式
        if image_channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        darknet_image = darknet.make_image(width, height, image_channel)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

        # 获取检测结果
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
        darknet.free_image(darknet_image)

        return detections, (width_ori / width, height_ori / height)


def save_result(image, results, resize, save_crop=""):
    """
    在原图上根据检测结果绘制检测框，并根据类别保存检测框内的目标
    :param image: cv2格式的图片
    :param results: 检测结果
    :param resize: 原图尺寸和网络尺寸之比
    :param save_crop: 保存检测框内目标的路径
    :return: 绘制好检测框的原图
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if results:
        for detection in results:
            # 当前检测框的标签和置信度
            label = detection[0]
            confidence = detection[1]
            confidence = "{}%".format(confidence)

            # bbox信息
            bounds = detection[2]
            x = int(bounds[0] * resize[0])
            y = int(bounds[1] * resize[1])
            w = int(bounds[2] * resize[0])
            h = int(bounds[3] * resize[1])

            # 左上角的坐标
            x_coord = int(x - w / 2)
            y_coord = int(y - h / 2)
            bbox = [[x_coord, y_coord], [x_coord, y_coord + h], [x_coord + w, y_coord + h], [x_coord + w, y_coord]]

            # 矫正bbox
            if bbox[0][0] < 0:
                bbox[0][0] = 0
            if bbox[0][1] < 0:
                bbox[0][1] = 0
            if bbox[1][0] < 0:
                bbox[1][0] = 0
            if bbox[1][1] > image.shape[0]:
                bbox[1][1] = image.shape[0]
            if bbox[2][0] > image.shape[1]:
                bbox[2][0] = image.shape[1]
            if bbox[2][1] > image.shape[0]:
                bbox[2][1] = image.shape[0]
            if bbox[3][0] > image.shape[1]:
                bbox[3][0] = image.shape[1]
            if bbox[3][1] < 0:
                bbox[3][1] = 0

            # 按类别保存检测框内的图片
            if save_crop != "":
                # if label == "1":
                if True:
                    save_dir_crop_now = save_crop + "/" + label
                    if not os.path.exists(save_dir_crop_now):
                        os.makedirs(save_dir_crop_now)
                    global count
                    image_crop = image[bbox[0][1]:bbox[2][1], bbox[0][0]:bbox[2][0]]
                    # image_crop = cv2.resize(image[bbox[0][1]:bbox[2][1], bbox[0][0]:bbox[2][0]], (64, 64))
                    save_dir_crop_now_ = os.path.join(save_dir_crop_now, "{}.png".format(str(count).zfill(6)))
                    cv2.imencode('.png', image_crop)[1].tofile(save_dir_crop_now_)
                    count += 1

            # 绘制检测框
            cv2.rectangle(image_bgr, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 5)
            # 在检测框右下角显示类别和置信度信息：
            text = str(label) + ":" + confidence
            cv2.putText(image_bgr, text, tuple(bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
    return image_bgr


if __name__ == '__main__':
    count = 0
    coordinates_list = []  # 创建坐标列表
    boxes = []
    confidences = []
    classIDs = []

    # 网络模型初始化
    network = Network()
    # 获取输入图像列表
    image_list = network.load_images()

    # 检测结果保存路径
    save_path_crop = os.path.join(Config.save_path, "crop")
    if not os.path.exists(save_path_crop):
        os.makedirs(save_path_crop)
    save_path_whole = os.path.join(Config.save_path, "whole")
    if not os.path.exists(save_path_whole):
        os.makedirs(save_path_whole)

    # 逐图进行检测
    for file_name in image_list:
        # 读图
        img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), 0)
        print("正在检测：{}".format(file_name))
        # 检测
        detect_results, resize_rate = network.image_detection(img)
        # 绘制并保存检测结果
        result_img = save_result(img, detect_results, resize_rate, save_path_crop)
        cv2.imencode('.jpg', result_img)[1].tofile(os.path.join(save_path_whole, os.path.basename(file_name)))
        # 生成xml文件
        if detect_results:
            anno = Gen_Annotations(os.path.basename(file_name), path=file_name)
            if not os.path.exists(Config.xml_save_path):
                os.makedirs(Config.xml_save_path)
            Outname = Config.xml_save_path + os.path.basename(file_name)
            (h, w) = img.shape[:2]
            anno.set_size(w, h, 1)
            anno.add_object(img, detect_results, resize_rate, w, h, 1)
            name_xml = Outname[:-4] + '.xml'
            anno.save_xml(name_xml)
